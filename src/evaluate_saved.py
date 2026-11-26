"""Evaluate a finished run's saved checkpoints again: decision trials, the recovered rule and the report questions,
with no training. Driven by evaluate_checkpoints.ipynb.

    from src import evaluate_saved as ES
    out_dir, timing, loaded = ES.evaluate(run, steps=[0, 1500, 3000], name="recheck")

The base model loads once; each step applies that step's saved LoRA adapter (checkpoints/<run_id>/decision/step-<N>),
and step 0 is the base model with no adapter. Each step goes through evaluate.evaluate_model, the function the
training run evaluates its checkpoints with, so the files are the same ones a run writes: selections_, recovered_,
weight_reports_, reports_raw_, persona_scores_, learned_vs_reported_, design_correlation_, screen_fit_, and results
rows. Settings are the run's own results/<run_id>/settings.json, except:

  questions_from / only / show_neither / reply_prefix   another report question set, as ask_only takes them
  overrides    {"block.field": value}, e.g. {"compute.decision_eval_batch_size": 16} for a smaller card, or
               {"model_estimating.estimation_trials_per_persona": 200} for a quicker pass
  flip_ab      True shows every trial, decision and report, with its two options in each other's place: the run's own
               trials, A's values under B and B's under A (model_estimating.flip_ab in the folder's settings.json).
               `flip_check` then says whether each answer kept its letter or followed the option.

Writes only into results/<run_id>/evaluate_only/<name>/ (results.csv there holds these rows alone). Never trains,
never appends to results/results.csv, never changes a file of the run.

`reproduction(run_dir, out_dir, step)` compares a step's decisions with the run's own selections_ file at that step:
with the run's settings they should be the same, decision for decision (greedy decoding, the same trials), up to
what a different card or batch size changes in the last bit of a logit.

How this differs from its neighbours: ask_only asks report questions only and refits the rule from the run's saved
decisions; this asks the decision trials again too. evaluate.evaluate_checkpoint does one checkpoint, loading the
base model each time, and writes into the run's folder and results/results.csv.
"""
import copy
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from src import ask_only as A
from src import config as C
from src import cutoff_flags, data_paths, faithfulness_by_kind, step_answers
from src import data as D
from src.evaluate import evaluate_model
from src.train import stage_label

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "base_model"   # pipeline.BASE_MODEL: the stage name the training run gives its step-0 rows


# Settings a run's settings.json may not hold and an override may still set: config.OPTIONAL's, and the optional
# fields of the blocks config.validate lists no fields for (each is read with .get(), and nothing fills it in).
SETTABLE = {**{block: set(fields) for block, fields in C.OPTIONAL.items()},
            "model_estimating": {"bootstrap_draws", "flip_ab"},
            "report_schema": {"reply_prefix", "show_neither"}}


def apply_overrides(cfg, overrides):
    """Set {"block.field": value} entries in cfg; an unknown block or field is refused, not added. A field the run's
    settings do not hold is still set when its block takes it (SETTABLE); validate then checks the value."""
    for key, value in (overrides or {}).items():
        block, _, field = key.partition(".")
        if not field or block not in cfg or (field not in cfg[block] and field not in SETTABLE.get(block, ())):
            raise KeyError(f"override {key!r}: no such setting (expected 'block.field', e.g. compute.eval_batch_size)")
        cfg[block][field] = value
    C.validate(cfg)
    return cfg


def setup_differences(record, cfg, overrides):
    """What differs between the set-up an evaluate-only folder was evaluated under (`record`, its settings.json) and
    the one about to run: the questions and how they are asked (report_schema, which `only` and QUESTIONS_FROM
    change), whether A and B are swapped (flip_ab), and the overrides. [] when they are the same. compute's
    overrides are not compared: batch sizes do not change a result. As ask_only.setup_differences."""
    plain = lambda value: json.loads(json.dumps(value))
    old_rs, new_rs = record.get("report_schema", {}), plain(cfg["report_schema"])
    out = [f"report_schema.{k}: {json.dumps(old_rs.get(k))} there, {json.dumps(new_rs.get(k))} now"
           for k in sorted(set(old_rs) | set(new_rs)) if old_rs.get(k) != new_rs.get(k)]
    was = record.get("_evaluate_only", {})
    flip = bool(cfg["model_estimating"].get("flip_ab", False))
    if bool(was.get("flip_ab", False)) != flip:
        out.append(f"flip_ab: {bool(was.get('flip_ab', False))} there, {flip} now")
    kept = lambda o: {k: v for k, v in plain(o or {}).items() if not k.startswith("compute.")}
    old_o, new_o = kept(was.get("overrides")), kept(overrides)
    out += [f"override {k}: {json.dumps(old_o.get(k))} there, {json.dumps(new_o.get(k))} now"
            for k in sorted(set(old_o) | set(new_o)) if old_o.get(k) != new_o.get(k)]
    return out


def prepare(run, name, questions_from=None, only=None, show_neither=None, reply_prefix=None, overrides=None,
            results_dir=None, flip_ab=None):
    """(run_dir, out_dir, cfg): the run's settings with the changes applied, and the output folder."""
    run_dir = A.find_run(run, results_dir)
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"name must be a plain folder name, got {name!r}")
    cfg = A.run_settings(run_dir, questions_from, only, show_neither, reply_prefix, flip_ab)
    data_dir = Path(cfg["model_training"]["data_dir"])
    cfg["model_training"]["data_dir"] = str(data_dir if data_dir.is_absolute() else (ROOT / data_dir).resolve())
    apply_overrides(cfg, overrides)
    out_dir = run_dir / "evaluate_only" / name
    # One folder holds one set-up: its settings.json describes every step in it, and the summaries are rebuilt
    # over all of them. Until 2026-11-25 a second call with another set-up (FLIP_AB, other questions, an override)
    # rewrote settings.json and added its steps to the first call's, so the folder mixed two set-ups under the
    # description of the last.
    if (out_dir / "settings.json").exists() and any(out_dir.glob("reports_raw_*.jsonl")):
        diff = setup_differences(json.loads((out_dir / "settings.json").read_text(encoding="utf-8")), cfg, overrides)
        if diff:
            raise ValueError(f"{out_dir} already holds steps evaluated with other settings, and one folder must hold "
                             "one set-up. Choose another NAME (or delete that folder). What differs: " + "; ".join(diff))
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**cfg, "_evaluate_only": {"run_id": run_dir.name, "questions_from": str(questions_from) if questions_from else None,
                                        "only": list(only) if only else None, "overrides": overrides or {},
                                        "flip_ab": bool(cfg["model_estimating"].get("flip_ab", False))}}
    # as ask_only.prepare: the record names the data folder relative to the repo, not by this machine's path to it
    record["model_training"] = {**cfg["model_training"], "data_dir": data_paths.recorded(cfg["model_training"]["data_dir"])}
    (out_dir / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return run_dir, out_dir, cfg


def _model_for(loaded, run_id, step, checkpoint_root=None):
    """The model to evaluate at `step`: the base model for 0, else the base with that step's adapter active.

    Returns (model, context): the context disables the adapters for step 0 once any adapter is loaded (loading one
    wraps the base model in place, so the plain base is reached by switching the adapters off)."""
    import contextlib

    if step == 0:
        if loaded["model"] is None:
            return loaded["base"], contextlib.nullcontext()
        return loaded["model"], loaded["model"].disable_adapter()
    return A.apply_adapter(loaded, run_id, step, checkpoint_root), contextlib.nullcontext()


def _move_up(staging, run_id, out_dir):
    """evaluate_model writes into <results_dir>/<run_id>/ and <results_dir>/results.csv: bring both into out_dir."""
    for f in (staging / run_id).iterdir():
        target = out_dir / f.name
        if target.exists():
            target.unlink() if target.is_file() else shutil.rmtree(target)
        shutil.move(str(f), str(target))
    rows = staging / "results.csv"
    if rows.exists():
        new = pd.read_csv(rows)
        old = out_dir / "results.csv"
        if old.exists():
            prev = pd.read_csv(old)
            # a step evaluated again under the same name replaces its earlier rows
            new = pd.concat([prev[~prev["checkpoint_step"].isin(new["checkpoint_step"].unique())], new], ignore_index=True)
        new.to_csv(old, index=False, lineterminator="\n")
    shutil.rmtree(staging)


def summaries(out_dir):
    """Rebuild step_answers, faithfulness_by_trial_kind and cutoff_flags over every step evaluated into out_dir.

    evaluate_model builds them where it writes, the staging folder, which holds one step at a time: each step's
    copy would replace the last when it is moved up. Here they are built again where every step's files are."""
    for name, build in (("step_answers", step_answers.update), ("faithfulness_by_trial_kind", faithfulness_by_kind.run),
                        ("cutoff_flags", cutoff_flags.update)):
        try:
            build(out_dir)
        except Exception as err:    # a summary must never lose the evaluation it summarizes
            print(f"  {name} not written: {err!r}")


def evaluate(run, steps, name, questions_from=None, only=None, show_neither=None, reply_prefix=None, overrides=None,
             results_dir=None, loaded=None, checkpoint_root=None, flip_ab=None):
    """Evaluate every step in `steps` (0 = the base model). `loaded` reuses a base model already in memory; it is
    returned for the next call. `flip_ab=True` shows every trial with A's and B's values swapped. Returns
    (out_dir, timing, loaded)."""
    run_dir, out_dir, cfg = prepare(run, name, questions_from, only, show_neither, reply_prefix, overrides, results_dir,
                                    flip_ab)
    comps = C.resolve(cfg)
    model_name = cfg["model_hyperparameters"]["model_name"]
    if cfg["model_hyperparameters"]["finetune"] != "lora":
        raise ValueError("evaluate_saved applies LoRA adapters; this run saved full weights")
    if loaded is not None and loaded["model_name"] != model_name:
        raise ValueError(f"the loaded base model is {loaded['model_name']}, this run's is {model_name}: pass loaded=None")
    saved = A.available_steps(run_dir.name, checkpoint_root)
    missing = [s for s in steps if s and s not in saved]
    if missing:
        raise FileNotFoundError(f"no adapter saved here for steps {missing} of {run_dir.name} (saved: {saved}); "
                                "download_checkpoint.ipynb fetches them from the Hub")
    if loaded is None:
        from src.train import load_model_and_tokenizer
        comps.backend.setup()
        device = comps.backend.get_device()
        base, tok = load_model_and_tokenizer(cfg, device)
        loaded = {"base": base, "tok": tok, "device": device, "model": None, "adapters": set(), "model_name": model_name}
    data = D.load_data(cfg, comps.rule)
    personas = list(range(cfg["model_training"]["instances"]))
    timing = []
    for step in sorted(steps, key=lambda s: (s != 0, s)):    # the base model first, before any adapter wraps it
        model, context = _model_for(loaded, run_dir.name, step, checkpoint_root)
        model.eval()
        staging = out_dir / "_staging"
        if staging.exists():
            shutil.rmtree(staging)
        # the summaries evaluate_model prints after a step read the persona's rule through settings.json
        (staging / run_dir.name).mkdir(parents=True)
        shutil.copy(out_dir / "settings.json", staging / run_dir.name / "settings.json")
        t = time.time()
        with context:
            evaluate_model(model, loaded["tok"], cfg, comps, data, loaded["device"], run_dir.name,
                           BASE_MODEL if step == 0 else "decision", 0, step, personas, staging)
        _move_up(staging, run_dir.name, out_dir)
        timing.append({"step": step, "seconds": round(time.time() - t, 1)})
        print(f"step {step}: {timing[-1]['seconds']:.0f}s")
    summaries(out_dir)
    pd.DataFrame(timing).to_csv(out_dir / "timing.csv", index=False, lineterminator="\n")
    return out_dir, timing, loaded


def reproduction(run_dir, out_dir, step):
    """The share of decisions at `step` equal to the run's own selections_ file, or None when either is missing."""
    tag = f"{BASE_MODEL if step == 0 else stage_label('decision')}_step-{step}"
    theirs, mine = Path(run_dir) / f"selections_{tag}.csv", Path(out_dir) / f"selections_{tag}.csv"
    if not theirs.exists() or not mine.exists():
        return None
    a, b = pd.read_csv(theirs), pd.read_csv(mine)
    keys = ["persona", "trial", "sample"]
    merged = a.merge(b, on=keys, suffixes=("_run", "_now"))
    same = merged["selection_run"] == merged["selection_now"]
    gap = max((merged[f"{p}_run"] - merged[f"{p}_now"]).abs().max() for p in ("p_A", "p_B", "p_C") if f"{p}_run" in merged)
    return {"step": step, "decisions": len(merged), "identical": int(same.sum()), "match_rate": float(same.mean()),
            "max_probability_gap": float(gap)}


def flip_check(run_dir, out_dir, step):
    """For an evaluation made with flip_ab=True: each decision beside the run's own decision on the same
    trial before its options changed places. None when either selections file is missing.

      mirrored              True when every trial here is the run's trial with A's and B's values swapped. False means
                            this was not a flipped evaluation of the same trials, and the shares below mean nothing.
      run_A/B/C, now_A/B/C  how often each letter was answered, in the run and here
      same_letter           of the decisions the run answered A or B, the share answering that same letter here:
                            1.0 for a model that answers by the letter (or the place), whatever the options are
      followed_option       of the same decisions, the share answering the other letter here, the one the option the
                            run chose now sits under: 1.0 for a model that answers by the options' values
    The two shares add to 1 less the share that moved to C.
    """
    tag = f"{BASE_MODEL if step == 0 else stage_label('decision')}_step-{step}"
    theirs, mine = Path(run_dir) / f"selections_{tag}.csv", Path(out_dir) / f"selections_{tag}.csv"
    if not theirs.exists() or not mine.exists():
        return None
    merged = pd.read_csv(theirs).merge(pd.read_csv(mine), on=["persona", "trial", "sample"], suffixes=("_run", "_now"))
    attributes = [c[len("A_"):-len("_run")] for c in merged.columns if c.startswith("A_attribute_") and c.endswith("_run")]
    mirrored = bool(len(merged)) and all((merged[f"A_{a}_run"] == merged[f"B_{a}_now"]).all()
                                         and (merged[f"B_{a}_run"] == merged[f"A_{a}_now"]).all() for a in attributes)
    run, now = merged["selection_run"], merged["selection_now"]
    on_ab = run.isin(["A", "B"])
    other = run.map({"A": "B", "B": "A"})
    share = lambda hit: float(hit[on_ab].mean()) if on_ab.any() else float("nan")
    return {"step": step, "decisions": len(merged), "mirrored": mirrored,
            **{f"run_{letter}": int((run == letter).sum()) for letter in "ABC"},
            **{f"now_{letter}": int((now == letter).sum()) for letter in "ABC"},
            "same_letter": share(now == run), "followed_option": share(now == other)}
