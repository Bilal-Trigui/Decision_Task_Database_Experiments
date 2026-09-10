"""The orchestrator the notebook calls: settings -> data -> train -> evaluate every checkpoint -> results -> plots.

Reads every block: it hands each one to the module that owns it. The only
decisions made here are the run id, the order of stages, and the two
introspection folds (Plunkett's Experiment 2: train the report on the first
half of the personas and test on the second, then the reverse).
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
from src.config import resolve
from src.evaluate import append_results, evaluate_model, summarize
from src.plots import make_plots
from src.rules.base import check_manifest
from src.train import apply_finetune, load_model_and_tokenizer, restore, snapshot, train_stage


def make_run_id(cfg):
    short = cfg["model_hyperparameters"]["model_name"].split("/")[-1]
    return f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['decision_rule']['type']}_{short}"


def run(cfg, results_dir="results", checkpoint_root="checkpoints"):
    """Run one experiment end to end and return the results rows it wrote."""
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

    # stage one: decisions
    def on_checkpoint(step, _ckpt):
        rows, _ = evaluate_model(model, tok, cfg, comps, data, device, run_id, "decision", 0, step, all_personas, results_dir)
        all_rows.extend(rows)

    train_stage(
        model, tok, data.train, data.val, cfg, device, run_id, "decision", 0,
        mh["training_steps"], mh["batch_size"], mh["checkpoint_every"],
        checkpoint_root, Path(results_dir) / "train_log.csv", on_checkpoint,
    )

    # stage two: introspection training, two folds, tested on the held-out half
    if mh["introspection_training"]:
        schema = comps.schemas[0]
        print(f"introspection training on report schema '{schema.name}' targets")
        examples = D.load_introspection_examples(cfg, data.scenarios, data.latents, comps.rule, schema)
        half = mt["instances"] // 2
        first, second = all_personas[:half], all_personas[half:]
        state = snapshot(model)
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
                checkpoint_root, Path(results_dir) / "train_log.csv", None,
            )
            rows, detail = evaluate_model(
                model, tok, cfg, comps, data, device, run_id, "introspection", fold, mh["introspection_steps"], test_idx, results_dir
            )
            all_rows.extend(rows)
            details.append(detail)
        combined = summarize(details, cfg, comps, data, run_id, "introspection", "both", mh["introspection_steps"])
        append_results(combined, cfg, Path(results_dir) / "results.csv")
        all_rows.extend(combined)

    make_plots(run_id, results_dir)
    table = pd.DataFrame(all_rows)[
        ["stage", "introspection_fold", "checkpoint_step", "report_schema", "block", "decision_accuracy",
         "recovered_vs_hidden", "faithfulness", "chance", "parse_rate", "hidden_vs_reported"]
    ]
    print(f"\nresults for {run_id} ({time.time() - t0:.0f}s), appended to {Path(results_dir) / 'results.csv'}:")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    return all_rows
