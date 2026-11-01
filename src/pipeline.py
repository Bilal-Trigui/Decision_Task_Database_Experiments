"""This is the main file for what the Jupyter notebook calls. (run_experiment.ipynb)

It loads the data, trains the decision stage, evaluates at the checkpoints, optionally runs introspection training. 
It then writes the results and plots. This file mainly exists to create a common exoskeleton for running the experiment.

It's the main logic. This powers run_experiments.ipynb, which is the notebook that runs the experiments. 
"""
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data as D
from src.config import SettingsError, check_resume, introspection_points, resolve
from src.evaluate import (
    append_results, decision_detail_from_files, evaluate_model, faithfulness_view, learned_vs_reported, summarize,
)
from src.plots import make_plots
from src.rules.base import check_manifest
from src.train import (
    apply_finetune, load_checkpoint_weights, load_model_and_tokenizer, load_training_state, restore, rng_state,
    set_rng_state, snapshot, stage_label, train_stage,
)


BASE_MODEL = "base_model"   # the stage name of the rows that score the base model, untrained, before decision training


def rule_latent(rule, estimator, recovered, n):
    """A recovered latent (the estimator's columns) laid out as the rule's latent, block by block.

    The two can differ in shape (an estimator may hold blocks the rule does not), so each of the
    rule's blocks is copied from the estimator's block of the same name; a block the estimator does
    not recover stays 0.
    """
    rule_slices, est_slices = rule.block_slices(n), estimator.block_slices()
    out = np.zeros(len(rule.latent_columns(n)))
    for block in rule.blocks(n):
        if block in est_slices:
            out[rule_slices[block]] = np.asarray(recovered)[est_slices[block]]
    return out


def target_latents(target, people, data, comps, decision_detail, seed):
    """The latents introspection training trains toward for `people`, one per persona, in order.

    hidden     each persona's own, from instilled_weights.csv
    recovered  each persona's rule as recovered from the model's own choices at the decision checkpoint
    shuffled   another of `people`'s own, no one keeping their own: drawn within `people` only, so a
               fold's training never sees a held-out persona's rule
    random     a fresh draw from the decision rule, for nobody
    """
    n = data.attribute_count
    if target == "hidden":
        return [np.asarray(data.latents[k]) for k in people]
    if target == "recovered":
        counts = decision_detail.get("recovered_by_count") or {0: decision_detail["recovered"]}
        fits = decision_detail.get("recovered") or counts[max(counts)]
        return [rule_latent(comps.rule, comps.estimator, fits[k], n) for k in people]
    if target == "shuffled":
        rng = np.random.default_rng(seed)
        order = np.arange(len(people))
        perm = rng.permutation(len(people))
        while len(people) > 1 and np.any(perm == order):
            perm = rng.permutation(len(people))
        return [np.asarray(data.latents[people[j]]) for j in perm]
    if target == "random":
        rng = random.Random(seed)
        return [comps.rule.sample_latent(n, rng) for _ in people]
    raise ValueError(f"unknown introspection target {target!r}")


def make_run_id(cfg):
    """
    It builds a unique name for the run. It takes the timestamp, decision rule and the model.
    """
    short = cfg["model_hyperparameters"]["model_name"].split("/")[-1]
    return f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['decision_rule']['type']}_{short}"


def _resume_plan(cfg, run_id, from_step, results_dir, checkpoint_root, overwrite):
    """Check that a run can be resumed at `from_step`, and say what is left to do there.

    Refuses when the settings would change the experiment (check_resume), when the decision
    checkpoint or its evaluation is missing, or when results beyond from_step already exist (unless
    `overwrite`, which appends beside them). Returns the checkpoint folder, whether the introspection
    branch at from_step still has to run, and the settings changes allowed through.
    """
    run_dir = Path(results_dir) / run_id
    if not (run_dir / "settings.json").exists():
        raise SettingsError(f"no run to resume: {run_dir / 'settings.json'} is missing")
    saved = json.loads((run_dir / "settings.json").read_text())
    notes = check_resume(saved, cfg, from_step)
    ckpt = Path(checkpoint_root) / run_id / stage_label("decision") / f"step-{from_step}"
    if not ckpt.exists():
        have = sorted((Path(checkpoint_root) / run_id / "decision").glob("step-*"))
        raise SettingsError(f"no decision checkpoint at {ckpt}; this run has {[c.name for c in have]}")
    results = Path(results_dir) / "results.csv"
    rows = pd.read_csv(results, dtype=str, keep_default_na=False) if results.exists() else pd.DataFrame(
        columns=["run_id", "stage", "checkpoint_step", "introspection_from_step", "introspection_fold"])
    rows = rows[rows["run_id"] == run_id]
    checkpoint_step = pd.to_numeric(rows["checkpoint_step"], errors="coerce")
    from_steps = pd.to_numeric(rows["introspection_from_step"], errors="coerce")
    decision, introspection = rows["stage"] == "decision", rows["stage"] == "introspection"
    if not (decision & (checkpoint_step == from_step)).any():
        raise SettingsError(
            f"decision step {from_step} of {run_id} was never fully evaluated (no results.csv rows), so it "
            "cannot be resumed from; resume from the last step that was"
        )
    later = (decision & (checkpoint_step > from_step)) | (introspection & (from_steps > from_step))
    if later.any() and not overwrite:
        raise SettingsError(
            f"{run_id} already has {int(later.sum())} results rows past step {from_step}; resuming would add a "
            "second copy. Pass overwrite=True to resume anyway"
        )
    # each target's branch is done once its pooled "both" rows are written; rows from before targets
    # existed carry no target and were hidden
    targets = rows["introspection_target"] if "introspection_target" in rows else pd.Series("", index=rows.index)
    both = introspection & (rows["introspection_fold"] == "both") & (from_steps == from_step)
    done = {t or "hidden" for t in targets[both]}
    left = [t for t in cfg["model_hyperparameters"]["introspection_targets"] if t not in done]
    introspect_here = left if from_step in introspection_points(cfg) and left else []
    return ckpt, introspect_here, notes


RESUME_LOCK = "RESUMING.lock"


def run(cfg, results_dir="results", checkpoint_root="checkpoints", resume=None):
    """Run one experiment end to end and return the results rows it wrote; see _run.

    A resume holds a lock file in the run's results folder while it works, so a second resume of
    the same run (a notebook cell run twice, say) is refused instead of training beside the first
    on one GPU. A lock left by a process that was killed outright has to be deleted by hand.
    """
    if not resume:
        return _run(cfg, results_dir, checkpoint_root, None)
    lock = Path(results_dir) / resume["run_id"] / RESUME_LOCK
    if lock.exists():
        raise SettingsError(
            f"{lock} exists: a resume of this run is already going ({lock.read_text().strip()}). "
            "If that process is gone, delete the file and resume again"
        )
    if lock.parent.exists():
        lock.write_text(f"pid {os.getpid()}, from step {resume['from_step']}, started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    try:
        return _run(cfg, results_dir, checkpoint_root, resume)
    finally:
        if lock.exists():
            lock.unlink()


def _run(cfg, results_dir="results", checkpoint_root="checkpoints", resume=None):
    """Run one experiment end to end and return the results rows it wrote.
    It takes a settings dictionary, and runs the experiment.
    The settings dictionary is a JSON file from configs. Still, it isn't the exact file. The notebook builds it.

    `resume` continues an earlier run instead of starting one: {"run_id": ..., "from_step": N} and
    optionally "overwrite": True. The run keeps its id and folders. The decision adapter saved at
    step N is loaded with whatever training state was saved beside it, the work already done is
    skipped (the base model and every evaluation up to N), an introspection branch at N that never
    finished is run, and decision training carries on from step N + 1 as it would have. Every row
    it writes carries resumed_from_step N.
    """
    t0 = time.time()
    comps = resolve(cfg)
    comps.backend.setup()
    device = comps.backend.get_device()
    mt, mh = cfg["model_training"], cfg["model_hyperparameters"]
    check_manifest(mt["data_dir"], comps.rule, cfg)
    data = D.load_data(cfg, comps.rule)

    start_step, introspect_first, state = 0, [], None
    if resume:
        run_id, start_step = resume["run_id"], int(resume["from_step"])
        ckpt, introspect_first, notes = _resume_plan(cfg, run_id, start_step, results_dir, checkpoint_root,
                                                     resume.get("overwrite", False))
        cfg["_resumed_from_step"] = start_step
        run_dir = Path(results_dir) / run_id
        (run_dir / f"settings_resume_{datetime.now():%Y%m%d-%H%M%S}_from-{start_step}.json").write_text(
            json.dumps(cfg, indent=2) + "\n")
        print(f"resuming run {run_id} from decision step {start_step}; settings changed: "
              + ("; ".join(notes) if notes else "none"))
    else:
        run_id = make_run_id(cfg)
        run_dir = Path(results_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(
        f"run {run_id}: rule={comps.rule.name} estimator={comps.estimator.name} "
        f"schemas={[s.name for s in comps.schemas]} settings={cfg.get('_source')}"
    )

    random.seed(mh["seed"])
    np.random.seed(mh["seed"])
    torch.manual_seed(mh["seed"])
    model, tok = load_model_and_tokenizer(cfg, device)
    model = apply_finetune(model, cfg)
    if resume:
        load_checkpoint_weights(model, ckpt, cfg, device)
        state = load_training_state(ckpt, device)
        saved_rng = (state["training_state"] or {}).get("rng")
        if saved_rng is not None:
            set_rng_state(saved_rng)
    all_personas = list(range(mt["instances"]))
    all_rows = []
    points = introspection_points(cfg)  # the decision steps stage two branches from; [] when it is off
    if mh["introspection_training"]: # It's the on off stage. When this is true, a second training stage runs, which is introspection training. 
        # So basically, when it is false, you will get slef report, but it will be the "untrained version." The self report basically before Plunkett et Al (2025) trained the model to introspect. 
        schema = comps.schemas[0]
        print(f"introspection training on report schema '{schema.name}' targets, from decision steps {points}")
        examples = D.load_introspection_examples(cfg, data.scenarios, data.latents, comps.rule, schema)
        print(f"introspection targets: {mh['introspection_targets']}"
              + ("; decisions re-measured after each fold" if mh["introspection_remeasure_decisions"] else ""))

    def fold_examples(target, people, decision_detail, fold, from_step, split):
        """One fold's introspection examples toward `target`, for `people`, and the personas left out.

        The hidden target reads the data folder's examples; the others are built here from
        target_latents, with the same prompts. Saved under results/<run>/introspection_data/.
        """
        if target == "hidden":
            names = {data.personas[k].short_name for k in people}
            chosen, skipped = [e for e in examples if e.scenario in names], []
        else:
            seed = mh["seed"] + 1000 * fold + from_step + (0 if split == "train" else 500)
            latents = target_latents(target, people, data, comps, decision_detail, seed)
            skipped = []
            chosen = D.generate_introspection_examples(
                [data.scenarios[k] for k in people], latents, comps.rule, schema, data.attribute_count,
                mt["seeds"]["introspection"], skipped=skipped)
        folder = run_dir / "introspection_data" / f"from-{from_step}" / target / f"fold{fold}_{split}"
        D.write_question_files(chosen, folder, note=False)
        if skipped:
            (folder / "left_out.txt").write_text(
                "personas with no rule to report (a recovered fit with no screen):\n" + "\n".join(skipped) + "\n")
            print(f"  {target} target, fold {fold} {split}: {len(skipped)} personas left out, no rule to report: {skipped[:5]}")
        return chosen

    def introspect(from_step, decision_detail, targets=None):
        """
        Stage two, branched from the decision adapter as it stands at `from_step`, then undone.
        One branch per introspection target (model_hyperparameters.introspection_targets, or
        `targets`), each two folds, every fold starting from that adapter. Afterwards the adapter
        and every random number generator are put back exactly, so decision training carries on as
        if this never ran: the steps after `from_step` are the same with or without an introspection
        point there. Each fold's reports are scored against the rule recovered at `from_step`
        (`decision_detail`), and, with introspection_remeasure_decisions, against the rule its own
        choices give after report training too.
        """
        # stage two: introspection training, two folds, tested on the held-out half
        half = mt["instances"] // 2
        first, second = all_personas[:half], all_personas[half:]
        generators, state = rng_state(), snapshot(model)  # generators first: nothing in the branch may draw before they are saved
        for target in (targets or mh["introspection_targets"]):
            details = []
            for fold, (train_idx, test_idx) in enumerate([(first, second), (second, first)], start=1):
                restore(model, state)
                train = fold_examples(target, train_idx, decision_detail, fold, from_step, "train")
                val = fold_examples(target, test_idx, decision_detail, fold, from_step, "val")
                train_stage(
                    model, tok, train, val, cfg, device, run_id, "introspection", fold,
                    mh["introspection_steps"], mh["introspection_batch_size"], mh["introspection_steps"],
                    checkpoint_root, Path(results_dir) / "train_log.csv", None, from_step=from_step, target=target,
                )
                rows, detail = evaluate_model(
                    model, tok, cfg, comps, data, device, run_id, "introspection", fold, mh["introspection_steps"], test_idx,
                    results_dir, from_step, reuse=decision_detail, target=target,
                )
                all_rows.extend(rows)
                details.append(detail)
            combined = summarize(details, cfg, comps, data, run_id, "introspection", "both", mh["introspection_steps"],
                                 from_step, target=target)
            append_results(combined, cfg, Path(results_dir) / "results.csv")
            all_rows.extend(combined)
            tag = f"{stage_label('introspection', 0, from_step, target)}-both_step-{mh['introspection_steps']}"
            print(f"introspection from step {from_step}, target {target}, both folds:")
            learned_vs_reported(details, comps, data, run_dir / f"learned_vs_reported_{tag}.csv")
        restore(model, state)
        set_rng_state(generators)

    # the base model: the model before any training, asked everything a checkpoint is asked
    if mh["baseline_before_training"] and not resume:
        # A fresh LoRA adapter adds exactly zero, so this is the base model. Its random state is put
        # back afterwards, so decision training is the same with or without this row.
        generators = rng_state()
        print("base model: the untrained model, scored before decision training")
        rows, _ = evaluate_model(model, tok, cfg, comps, data, device, run_id, BASE_MODEL, 0, 0, all_personas, results_dir)
        all_rows.extend(rows)
        set_rng_state(generators)

    # stage one: decisions
    def on_checkpoint(step, _ckpt):
        """
        This is a callback function. This is handed to the training loop, so that the loop can call pipeline.py. It's job is to measure
        the model partwaythrough decision training (this is what gives you a learning curve instead of a single result at the end.)
        """
        rows, detail = evaluate_model(model, tok, cfg, comps, data, device, run_id, "decision", 0, step, all_personas, results_dir)
        all_rows.extend(rows)
        if step in points:
            introspect(step, detail)

    if introspect_first:
        # the branch at the resume step never finished; it reads that checkpoint's decisions back from its files
        print(f"introspection at step {start_step} did not finish for targets {introspect_first}; "
              "running them before decision training continues")
        introspect(start_step, decision_detail_from_files(cfg, comps, data, run_dir, start_step, all_personas),
                   targets=introspect_first)

    train_stage( # This runs the training step. For example, at step 300, it will compute the validation loss, save the model, call the on_checkpoint function, then call model.train().
        model, tok, data.train, data.val, cfg, device, run_id, "decision", 0,
        mh["training_steps"], mh["batch_size"], mh["checkpoint_every"],
        checkpoint_root, Path(results_dir) / "train_log.csv", on_checkpoint, extra_checkpoints=points,
        start_step=start_step, resume=state,
    )

    make_plots(run_id, results_dir)
    # reindex rather than select: a run where no report ever parsed has no
    # faithfulness key on any row, and selecting would raise KeyError here, after
    # every result was already written, and leave the notebook without `rows`.
    table = pd.DataFrame(all_rows).reindex(columns=[
        "stage", "introspection_target", "introspection_fold", "introspection_from_step", "checkpoint_step", "report_schema",
        "report_batch", "block", "estimation_trials",
        "decision_accuracy", "recovered_vs_hidden", "faithfulness", "shuffled_faithfulness", "faithfulness_shuffled_p",
        "chance", "parse_rate", "n_personas_scored", "hidden_vs_reported", "shuffled_hidden_vs_reported",
    ])
    print(f"\nresults for {run_id} ({time.time() - t0:.0f}s), appended to {Path(results_dir) / 'results.csv'}:")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
        # the same rows split into the kinds of faithfulness, one line per measure, each beside its own chance
        view = faithfulness_view(all_rows).drop(columns=["run_id"], errors="ignore")
        if not view.empty:
            print("\nfaithfulness by measure (learned: recovered vs hidden; absolute: reported vs hidden; "
                  "faithfulness: reported vs recovered; chance: random vs recovered):")
            print(view.to_string(index=False))
    return all_rows
