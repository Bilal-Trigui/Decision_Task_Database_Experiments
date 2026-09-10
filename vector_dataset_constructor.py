#!/usr/bin/env python3
"""Build the training files from a data folder's four inputs, using its weights as given.

    python vector_dataset_constructor.py --data data/plunkett                      # Plunkett's files, byte for byte
    python vector_dataset_constructor.py --data data/a3 --rule interaction         # a non-linear rule
    python vector_dataset_constructor.py --data data/my_latents --rule linear --out data/my_latents

Reads the four inputs:
    candidate_scenarios.json   the choice types, attributes, units, ranges      (authored)
    roles.csv                  the character names                              (authored)
    scenarios.csv              one role glued to each choice type               (seed 0)
    instilled_weights.csv      the hidden latent per persona, in the rule's     (whatever you put there)
                               column order: attr1..attrN, then the rule's extra columns

and writes, from those exactly, never rerolling the weights:
    instill_<N>_prefs.jsonl              N personas x train trials, labelled by the rule    (seed 2)
    instill_<N>_prefs_val.jsonl          N personas x val trials, same stream               (seed 2)
    instilled_weights_<N>_training.jsonl Plunkett's Experiment 2 file: the report prompt,   (seed 6)
                                         answered with the hidden weights
    introspection_training.csv           the same stage-two examples in the layout the
                                         pipeline reads at run time
    manifest.json                        rule, parameters, seeds, counts, and a hash per file

This is the tool for testing a latent you designed by hand: edit instilled_weights.csv,
run this, run the pipeline. To draw a fresh latent under a rule instead, use the
rule module (`python -m src.rules.<rule> --source ... --out ...`), which rerolls.

With --data data/plunkett and --rule linear, the three JSONLs reproduce Plunkett's
byte for byte, which tests/test_plunkett_fidelity.py also asserts. Reads
object.attribute_count from the folder; everything else is an argument.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src import data as D
from src.config import load_component
from src.rules.base import _parse_param

SEEDS = {"trials": 2, "introspection": 6}


def construct(data_dir, rule_name, out_dir=None, instances=100, train=50, val=10, params=None,
              seeds=None, schema_name=None):
    """Build the training files. Returns the manifest dict."""
    data_dir = Path(data_dir)
    out = Path(out_dir) if out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = {**SEEDS, **(seeds or {})}
    n = D.infer_attribute_count(data_dir)

    rule = load_component("rules", rule_name, params or {})
    schema = load_component("reports", schema_name or rule.name, {"attribute_count": n})

    scenarios = D.load_scenarios(data_dir, n)
    if instances > len(scenarios):
        raise D.DataFormatError(f"--instances is {instances} but {data_dir}/scenarios.csv has {len(scenarios)} rows")
    personas = scenarios[:instances]
    latents = D.load_hidden_latents(data_dir, personas, rule, n)   # as given; never rerolled
    print(f"inputs: {data_dir} | {len(scenarios)} scenarios, using the first {instances} | rule '{rule.name}' "
          f"| latent columns {rule.latent_columns(n)}")

    # stage one: decisions, labelled by the rule under these exact weights
    train_trials, val_trials = D.generate_decision_trials(personas, latents, rule, train, val, seeds["trials"])
    train_name, val_name = D.trial_file_names(instances)
    (out / train_name).write_text("\n".join(D.example_to_json(D.decision_example(t)) for ts in train_trials for t in ts))
    (out / val_name).write_text("\n".join(D.example_to_json(D.decision_example(t)) for ts in val_trials for t in ts))

    # stage two: the report prompt answered with the hidden weights, in both layouts
    examples = D.generate_introspection_examples(personas, latents, rule, schema, n, seeds["introspection"])
    (out / f"instilled_weights_{instances}_training.jsonl").write_text("\n".join(D.example_to_json(e) for e in examples))
    pd.DataFrame(
        [{"scenario": e.scenario, "batch": e.batch, "user_prompt": e.user, "assistant_answer": e.answer} for e in examples]
    ).to_csv(out / D.INTROSPECTION_TRAINING_CSV, index=False)

    written = [train_name, val_name, f"instilled_weights_{instances}_training.jsonl", D.INTROSPECTION_TRAINING_CSV]
    manifest = {
        "constructed_by": "vector_dataset_constructor.py",
        "rule": rule.name,
        "params": rule.params,
        "schema": schema.name,
        "seeds": seeds,
        "attribute_count": n,
        "instances": instances,
        "train_trials_per_persona": train,
        "val_trials_per_persona": val,
        "weights_used_as_given_from": str(data_dir / "instilled_weights.csv"),
        "inputs": {name: _hash(data_dir / name) for name in
                   (D.SCENARIO_DEFINITIONS, "roles.csv", "scenarios.csv", "instilled_weights.csv")},
        "files": {name: _hash(out / name) for name in written},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}:")
    for name in written:
        print(f"  {name:44s} {manifest['files'][name]}")
    return manifest


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="folder holding the four inputs")
    parser.add_argument("--rule", default="linear", help="decision rule that labels the trials: linear, interaction, ...")
    parser.add_argument("--schema", default=None, help="report schema for the stage-two file; default is the rule's own")
    parser.add_argument("--out", default=None, help="where to write; default is the data folder itself")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--trials-seed", type=int, default=SEEDS["trials"])
    parser.add_argument("--introspection-seed", type=int, default=SEEDS["introspection"])
    parser.add_argument("--param", type=_parse_param, action="append", default=[], help="rule parameter, key=value")
    a = parser.parse_args()
    construct(a.data, a.rule, a.out, a.instances, a.train, a.val, dict(a.param),
              {"trials": a.trials_seed, "introspection": a.introspection_seed}, a.schema)


if __name__ == "__main__":
    main()
