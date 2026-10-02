"""Re-evaluating a finished run with newer evaluation code, without retraining.

A rerun has to score the saved adapters under the settings they were trained with, not under
whatever configs/ holds today, and it has to write somewhere the original run's results are not.

    python -m src.rerun --run-dir results/<run_id> --list   # what was saved, no model loaded
    python -m src.rerun --run-dir results/<run_id>          # base model, every checkpoint, both folds

1. `load_run_settings` reads the settings.json a run wrote into its results folder (the merged
   settings, after "extends" and defaults) and puts it through the same checks `config.load`
   applies to a file in configs/, so a field today's code no longer accepts stops here rather
   than midway through an 8B evaluation. A settings.json that only "extends" a config, written
   to score the base model with no run behind it, is labelled by the files it extends.
2. `settings_drift` lists every field where the saved settings differ from a settings file as it
   loads today, so a changed config is seen rather than silently rerun under.
3. `fresh_results_dir` gives the rerun its own results folder. The new evaluation code adds
   columns, so `append_results` refuses the original results.csv, and a separate folder also
   keeps the rerun from overwriting the original run's per-checkpoint files, which share tags.
4. `find_checkpoints` lists the adapters the run saved, by stage and step.
5. `rerun` loads the base model once and evaluates, in order:
   - the base model with no adapter, as a decision row at checkpoint_step 0. It is the
     common-sense control: whatever it scores, the model had before any training.
   - every decision checkpoint, on all personas, as the pipeline did.
   - each introspection fold on the half of the personas it was NOT trained on (fold 1 trained
     on the first half, fold 2 on the second), then the two combined into the "both" row,
     exactly as pipeline.run does. `evaluate_checkpoint` scores every persona, which for a fold
     adapter includes the personas it was trained to report on, so it is not used here.
   Adapters are swapped on the one loaded base rather than reloading 8B weights per checkpoint.
   The base is loaded for inference only, without the kbit training preparation, as
   `evaluate_checkpoint` also does.

Reads all eight blocks through config.load, and model_training.instances for the folds.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src import data as D
from src.config import load, resolve

STAGE_DIRS = {"decision": ("decision", 0), "introspection-fold1": ("introspection", 1), "introspection-fold2": ("introspection", 2)}


def load_run_settings(run_dir):
    """The settings a finished run was trained with, validated by today's code.

    A settings.json the pipeline wrote carries the `_source` of the run that trained. One with
    none was written by hand to score the base model alone, as the notebook's control does, so
    its label names the files it extends instead. The label lands in every row's config_file.
    """
    path = Path(run_dir) / "settings.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no settings.json in {run_dir}; the pipeline writes one into results/<run_id>/ at the "
            "start of every run, so pull the run's results folder back first"
        )
    original = json.loads(path.read_text()).get("_source")
    cfg = load(str(path))
    if original:
        cfg["_source"] = f"{path} (trained from {original})"
    else:
        # cfg["_source"] is this file first, then each file it extends, as paths taken through
        # the run folder ("results/x/../../configs/..."); normpath folds them back to the repo's.
        extended = [os.path.normpath(p) for p in cfg["_source"].split(" <- ")[1:]]
        cfg["_source"] = f"{path} (base model, extends {' <- '.join(extended) or 'nothing'})"
    return cfg


def settings_drift(cfg, current_path):
    """Fields where `cfg` differs from `current_path` as it loads today: [(field, saved, current)]."""
    current = load(str(current_path))
    drift = []
    for block in sorted(set(cfg) | set(current)):
        if block.startswith("_"):
            continue
        saved_block, current_block = cfg.get(block, {}), current.get(block, {})
        for field in sorted(set(saved_block) | set(current_block)):
            saved, now = saved_block.get(field, "<absent>"), current_block.get(field, "<absent>")
            if saved != now:
                drift.append((f"{block}.{field}", saved, now))
    return drift


def fresh_results_dir(run_dir, results_dir="results_rerun"):
    """A results folder for the rerun, refused if it is the folder the original run wrote to."""
    original = Path(run_dir).resolve().parent
    target = Path(results_dir)
    if target.resolve() == original:
        raise ValueError(
            f"{results_dir} is where run {Path(run_dir).name} wrote its results; a rerun there would "
            "hit the old results.csv's columns and overwrite the original per-checkpoint files"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_checkpoints(checkpoint_root, run_id):
    """{stage folder: [(step, path), ...] by step} for every stage folder the run saved."""
    run_root = Path(checkpoint_root) / run_id
    found = {}
    for folder in STAGE_DIRS:
        stage_dir = run_root / folder
        steps = sorted(
            (int(p.name.split("-")[-1]), p)
            for p in stage_dir.glob("step-*")
            if p.is_dir() and p.name.split("-")[-1].isdigit()
        ) if stage_dir.is_dir() else []
        if steps:
            found[folder] = steps
    return found


def describe_checkpoints(found, cfg):
    """Lines saying what was saved, and whether anything exists before the first scheduled checkpoint."""
    every = cfg["model_hyperparameters"]["checkpoint_every"]
    lines = []
    for folder in STAGE_DIRS:
        steps = [s for s, _ in found.get(folder, [])]
        lines.append(f"  {folder}: {', '.join(map(str, steps)) if steps else 'none found'}")
    decision = [s for s, _ in found.get("decision", [])]
    if decision:
        early = [s for s in decision if s < every]
        lines.append(
            f"  decision checkpoints before step {every}: {', '.join(map(str, early))}" if early else
            f"  no decision checkpoint before step {every} (checkpoint_every is {every}); "
            "the base-model row at step 0 is the only earlier point"
        )
    return lines


def _swap_adapter(model, path, name):
    """`model` with the adapter at `path` active and no other adapter loaded."""
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        return PeftModel.from_pretrained(model, str(path), adapter_name=name)
    previous = model.active_adapter
    model.load_adapter(str(path), adapter_name=name)
    model.set_adapter(name)
    model.delete_adapter(previous)
    return model


def rerun(run_dir, checkpoint_root="checkpoints", results_dir="results_rerun", base=True, decision=True,
          introspection=True, steps=None):
    """Evaluate the base model and a finished run's adapters with today's evaluation code. Returns the rows."""
    from src.evaluate import append_results, evaluate_model, summarize
    from src.plots import make_plots
    from src.train import load_model_and_tokenizer

    run_dir = Path(run_dir)
    run_id = run_dir.name
    cfg = load_run_settings(run_dir)
    if cfg["model_hyperparameters"]["finetune"] != "lora":
        raise ValueError("rerun swaps LoRA adapters on one base; for a full fine-tune use evaluate_checkpoint per step")
    out = fresh_results_dir(run_dir, results_dir)
    found = find_checkpoints(checkpoint_root, run_id)
    print(f"rerun of {run_id} into {out}/, settings {cfg['_source']}")
    print("checkpoints under " + str(Path(checkpoint_root) / run_id) + ":")
    print("\n".join(describe_checkpoints(found, cfg)))
    if not found and not base:
        raise FileNotFoundError(f"no checkpoints under {Path(checkpoint_root) / run_id} and base is off; nothing to evaluate")

    comps = resolve(cfg)
    comps.backend.setup()
    device = comps.backend.get_device()
    data = D.load_data(cfg, comps.rule)
    seed = cfg["model_hyperparameters"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(cfg, device)
    all_personas = list(range(cfg["model_training"]["instances"]))
    rows = []

    def evaluate(stage, fold, step, personas):
        new, detail = evaluate_model(model, tok, cfg, comps, data, device, run_id, stage, fold, step, personas, out)
        rows.extend(new)
        return detail

    # the control: no adapter at all
    if base:
        evaluate("decision", 0, 0, all_personas)

    if decision:
        for step, path in found.get("decision", []):
            if steps is None or step in steps:
                model = _swap_adapter(model, path, f"decision_{step}")
                evaluate("decision", 0, step, all_personas)

    # each fold on the half it did not train on, then both folds as one row
    folds = [folder for folder in ("introspection-fold1", "introspection-fold2") if folder in found]
    if introspection and folds:
        if len(folds) < 2:
            print(f"only {folds[0]} was found; its rows are written but no combined 'both' row is")
        half = len(all_personas) // 2
        held_out = {1: all_personas[half:], 2: all_personas[:half]}
        details, last_step = [], None
        for folder in folds:
            _, fold = STAGE_DIRS[folder]
            last_step, path = found[folder][-1]
            model = _swap_adapter(model, path, f"introspection_fold{fold}")
            details.append(evaluate("introspection", fold, last_step, held_out[fold]))
        if len(details) == 2:
            combined = summarize(details, cfg, comps, data, run_id, "introspection", "both", last_step)
            append_results(combined, cfg, out / "results.csv")
            rows.extend(combined)

    make_plots(run_id, out)
    print(f"{len(rows)} rows appended to {out / 'results.csv'}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="the run's results folder, results/<run_id>")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--results-dir", default="results_rerun")
    parser.add_argument("--compare", default="configs/a4_8b.json", help="settings file to check for drift")
    parser.add_argument("--list", action="store_true", help="check settings and list checkpoints, load no model")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="decision steps to evaluate; default all")
    parser.add_argument("--no-base", action="store_true", help="skip the base model control")
    parser.add_argument("--no-decision", action="store_true", help="skip the decision checkpoints")
    parser.add_argument("--no-introspection", action="store_true", help="skip the introspection folds")
    args = parser.parse_args()

    cfg = load_run_settings(args.run_dir)
    print(f"settings load and validate: {cfg['_source']}")
    drift = settings_drift(cfg, args.compare)
    if drift:
        print(f"{len(drift)} fields differ from {args.compare} today; the rerun uses the saved values:")
        for field, saved, now in drift:
            print(f"  {field}: saved {saved!r}, today {now!r}")
    else:
        print(f"no fields differ from {args.compare}")
    if args.list:
        found = find_checkpoints(args.checkpoint_root, Path(args.run_dir).name)
        print("checkpoints under " + str(Path(args.checkpoint_root) / Path(args.run_dir).name) + ":")
        print("\n".join(describe_checkpoints(found, cfg)))
        return
    rerun(args.run_dir, args.checkpoint_root, args.results_dir, base=not args.no_base,
          decision=not args.no_decision, introspection=not args.no_introspection, steps=args.steps)


if __name__ == "__main__":
    main()
