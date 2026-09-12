#!/usr/bin/env python3
"""Build the training files from a data folder's four inputs, using its weights as given.

    python data/plunkett/vector_dataset_constructor.py --data data/plunkett                 # Plunkett's files, byte for byte
    python data/plunkett/vector_dataset_constructor.py --data data/a3 --rule interaction    # a non-linear rule
    python data/plunkett/vector_dataset_constructor.py --data data/mine --rule linear       # your own weights

Run from the repo root. --data also accepts a bare folder name inside data/.

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
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))   # this file lives in data/plunkett/; src/ is two levels up

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src.config import load_component  # noqa: E402
from src.rules.base import _parse_param  # noqa: E402

SEEDS = {"trials": 2, "introspection": 6}


def construct(data_dir, rule_name, out_dir=None, instances=100, train=50, val=10, params=None,
              seeds=None, schema_name=None):
    """Build the training files. Returns the manifest dict."""
    data_dir = resolve_data_dir(data_dir)
    out = Path(out_dir) if out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = {**SEEDS, **(seeds or {})}
    n = D.infer_attribute_count(data_dir)

    rule = load_component("rules", rule_name or rule_of(data_dir), params or {})
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

    # whatever this rule wants written down about the shape of its own latent
    views = rule.views(personas, latents, n)
    for name, rows in views.items():
        pd.DataFrame(rows).to_csv(out / name, index=False)
    stats = rule.dataset_stats(train_trials, val_trials, latents, n)

    written = [train_name, val_name, f"instilled_weights_{instances}_training.jsonl",
               D.INTROSPECTION_TRAINING_CSV] + sorted(views)
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
        # relative to the repo, so a manifest built on one machine does not carry that
        # machine's home directory into git
        "weights_used_as_given_from": str(Path(os.path.relpath(data_dir / "instilled_weights.csv", REPO)).as_posix()),
        "views": sorted(views),
        "stats": stats,
        "inputs": {name: _hash(data_dir / name) for name in
                   (D.SCENARIO_DEFINITIONS, "roles.csv", "scenarios.csv", "instilled_weights.csv")},
        "files": {name: _hash(out / name) for name in written},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}:")
    for name in written:
        print(f"  {name:44s} {manifest['files'][name]}")
    if stats:
        print("stats:", json.dumps(stats))
    for note in rule.audit(latents, n, stats):
        print(f"WARNING: {note}")
    return manifest


def resolve_data_dir(data_dir):
    """`--data a3` means data/a3, so a folder can be named without its path."""
    data_dir = Path(data_dir)
    if not data_dir.exists() and (REPO / "data" / data_dir).exists():
        return REPO / "data" / data_dir
    return data_dir


def rule_of(data_dir):
    """The rule a built folder says it holds, from its manifest."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise D.DataFormatError(f"no rule given and {path} does not exist to name one")
    return json.loads(path.read_text())["rule"]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main(default_data=None, default_rule=None):
    """The command line. A folder's own wrapper calls this with itself as the default."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=default_data, required=default_data is None,
                        help="folder holding the four inputs")
    parser.add_argument("--rule", default=default_rule, help="decision rule that labels the trials; default is what the folder's manifest says")
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
