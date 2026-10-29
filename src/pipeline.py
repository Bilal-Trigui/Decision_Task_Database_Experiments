"""This is the main file for what the Jupyter notebook calls. (run_experiment.ipynb)

It loads the data, trains the decision stage, evaluates at the checkpoints, optionally runs introspection training. 
It then writes the results and plots. This file mainly exists to create a common exoskeleton for running the experiment.

It's the main logic. This powers run_experiments.ipynb, which is the notebook that runs the experiments. 
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data as D
from src.config import introspection_points, resolve
from src.evaluate import append_results, evaluate_model, faithfulness_view, summarize
from src.plots import make_plots
from src.rules.base import check_manifest
from src.train import (
    apply_finetune, load_model_and_tokenizer, restore, rng_state, set_rng_state, snapshot, train_stage,
)


CHANCE_LEVEL = "chance_level"   # the stage name of the rows that score the untrained model


def make_run_id(cfg):
    """
    It builds a unique name for the run. It takes the timestamp, decision rule and the model.
    """
    short = cfg["model_hyperparameters"]["model_name"].split("/")[-1]
    return f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['decision_rule']['type']}_{short}"


def run(cfg, results_dir="results", checkpoint_root="checkpoints"): 
    """Run one experiment end to end and return the results rows it wrote.
    It takes a settings dictionary, and runs the experiment. 
    The settings dictionary is a JSON file from configs. Still, it isn't the exact file. The notebook builds it.

    """
    t0 = time.time()
    comps = resolve(cfg)
    comps.backend.setup()
    device = comps.backend.get_device()
    mt, mh = cfg["model_training"], cfg["model_hyperparameters"]
    check_manifest(mt["data_dir"], comps.rule, cfg)
    data = D.load_data(cfg, comps.rule)

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
    all_personas = list(range(mt["instances"]))
    all_rows = []
    points = introspection_points(cfg)  # the decision steps stage two branches from; [] when it is off
    if mh["introspection_training"]: # It's the on off stage. When this is true, a second training stage runs, which is introspection training. 
        # So basically, when it is false, you will get slef report, but it will be the "untrained version." The self report basically before Plunkett et Al (2025) trained the model to introspect. 
        schema = comps.schemas[0]
        print(f"introspection training on report schema '{schema.name}' targets, from decision steps {points}")
        examples = D.load_introspection_examples(cfg, data.scenarios, data.latents, comps.rule, schema)

    def introspect(from_step, decision_detail):
        """
        Stage two, branched from the decision adapter as it stands at `from_step`, then undone.
        Both folds start from that adapter. Afterwards the adapter and every random number
        generator are put back exactly, so decision training carries on as if this never ran:
        the steps after `from_step` are the same with or without an introspection point there.
        Each fold's reports are scored against the rule recovered at `from_step`
        (`decision_detail`), so the model is never asked an A/B question after report training.
        """
        # stage two: introspection training, two folds, tested on the held-out half
        half = mt["instances"] // 2
        first, second = all_personas[:half], all_personas[half:]
        generators, state = rng_state(), snapshot(model)  # generators first: nothing in the branch may draw before they are saved
        details = []
        for fold, (train_idx, test_idx) in enumerate([(first, second), (second, first)], start=1):
            restore(model, state)
            train_names = {data.personas[k].short_name for k in train_idx}
            fold_examples = [e for e in examples if e.scenario in train_names]
            val_names = {data.personas[k].short_name for k in test_idx}
            fold_val = [e for e in examples if e.scenario in val_names]
            train_stage(
                model, tok, fold_examples, fold_val, cfg, device, run_id, "introspection", fold,
                mh["introspection_steps"], mh["introspection_batch_size"], mh["introspection_steps"],
                checkpoint_root, Path(results_dir) / "train_log.csv", None, from_step=from_step,
            )
            rows, detail = evaluate_model(
                model, tok, cfg, comps, data, device, run_id, "introspection", fold, mh["introspection_steps"], test_idx,
                results_dir, from_step, reuse=decision_detail,
            )
            all_rows.extend(rows)
            details.append(detail)
        combined = summarize(details, cfg, comps, data, run_id, "introspection", "both", mh["introspection_steps"], from_step)
        append_results(combined, cfg, Path(results_dir) / "results.csv")
        all_rows.extend(combined)
        restore(model, state)
        set_rng_state(generators)

    # the chance-level baseline: the model before any training, asked everything a checkpoint is asked
    if mh["baseline_before_training"]:
        # A fresh LoRA adapter adds exactly zero, so this is the base model. Its random state is put
        # back afterwards, so decision training is the same with or without this row.
        generators = rng_state()
        print("chance-level baseline: the untrained model, before decision training")
        rows, _ = evaluate_model(model, tok, cfg, comps, data, device, run_id, CHANCE_LEVEL, 0, 0, all_personas, results_dir)
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

    train_stage( # This runs the training step. For example, at step 300, it will compute the validation loss, save the model, call the on_checkpoint function, then call model.train().
        model, tok, data.train, data.val, cfg, device, run_id, "decision", 0,
        mh["training_steps"], mh["batch_size"], mh["checkpoint_every"],
        checkpoint_root, Path(results_dir) / "train_log.csv", on_checkpoint, extra_checkpoints=points,
    )

    make_plots(run_id, results_dir)
    # reindex rather than select: a run where no report ever parsed has no
    # faithfulness key on any row, and selecting would raise KeyError here, after
    # every result was already written, and leave the notebook without `rows`.
    table = pd.DataFrame(all_rows).reindex(columns=[
        "stage", "introspection_fold", "introspection_from_step", "checkpoint_step", "report_schema", "report_batch", "block",
        "estimation_trials",
        "decision_accuracy", "recovered_vs_hidden", "faithfulness", "chance", "parse_rate",
        "n_personas_scored", "hidden_vs_reported",
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
