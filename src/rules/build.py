"""Building a data folder for a rule, and the command line that does it. Never called by a run.

Turns a rule into a data folder the pipeline can train on, starting from Plunkett's folder.
The personas and the questions stay his. Only the hidden latents change, and with them the
A/B answers. So everything except the rule is held fixed, which is the point of the design.

  roll_weights  Call the rule's sample_latent once per persona and collect the results into the
                instilled_weights.csv table: one row per persona, one column per latent column.
                data/general/vector_weight_generator.py also uses it.
  make_dataset  The whole build. Copy his persona and question files, roll new latents, make
                and label 50 train and 10 val trials per persona, write the two JSONL training
                files, and write manifest.json recording how the folder was built.
  cli           The terminal front end every rule file ends with, for example
                    python -m src.rules.minimum_threshold --source data/plunkett --out data/my_minimum_threshold

Seeds: one RNG, seeded once, used in persona order. That is how Plunkett did it, so the linear
rule with seed 1 (weights) and seed 2 (trials) rebuilds his files exactly.
tests/test_plunkett_fidelity.py checks this byte for byte.

The pipeline itself never builds a folder. It only reads one, after base.check_manifest has
confirmed the folder was built under the rule and parameters the settings name.

Reads: object.attribute_count (through `n`), decision_rule (the rule's own parameters),
model_training seeds.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D


def roll_weights(rule, scenarios, n, seed):
    """Draw one latent per scenario from the rule's target distribution, in scenario order.

    Returns the latents and the instilled_weights.csv frame built from them. One RNG is
    seeded once and consumed in scenario order, which is Plunkett's construction, so the
    linear rule at his seed reproduces his file. Columns that came out whole are written as
    integers, as his are.
    """
    rng = random.Random(seed)
    latents = np.vstack([rule.sample_latent(n, rng) for _ in scenarios])
    columns = rule.latent_columns(n)
    weights = pd.DataFrame(latents, columns=columns)
    for c in columns:
        if np.all(np.equal(np.mod(weights[c], 1), 0)):
            weights[c] = weights[c].astype(int)
    weights.insert(0, "scenario", [sc.short_name for sc in scenarios])
    return latents, weights


def make_dataset(rule, source_dir, out_dir, attribute_count=None, instances=100, seed=1, train=50, val=10, trials_seed=2):
    """Build a data folder for `rule` from Plunkett's scenarios.

    Keeps scenario_definitions.json, roles.csv and scenarios.csv (truncated to
    `attribute_count` attributes), rerolls instilled_weights.csv under the rule
    with `seed`, regenerates Plunkett's trials with `trials_seed` and relabels
    them, writes instill_<instances>_prefs.jsonl and its _val file, and a
    manifest.json. With the linear rule and Plunkett's seeds this reproduces
    his files exactly.
    """
    source, out = Path(source_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    native = D.infer_attribute_count(source)
    n = attribute_count or native
    if n > native:
        raise D.DataFormatError(f"source folder has {native} attributes per scenario; cannot build {n}")
    scenarios = D.load_scenarios(source, native)
    definitions = D.read_definitions(source)

    # scenario_definitions.json, roles.csv, scenarios.csv
    if n == native:
        shutil.copyfile(source / D.SCENARIO_DEFINITIONS, out / D.SCENARIO_DEFINITIONS)
        shutil.copyfile(source / "scenarios.csv", out / "scenarios.csv")
    else:
        for d in definitions:
            d["attributes"] = d["attributes"][:n]
        (out / D.SCENARIO_DEFINITIONS).write_text(json.dumps(definitions, indent=2))
        df = pd.read_csv(source / "scenarios.csv")
        df[D.scenarios_columns(n)].to_csv(out / "scenarios.csv", index=False)
    shutil.copyfile(source / "roles.csv", out / "roles.csv")
    scenarios = [D.Scenario(sc.short_name, sc.question, sc.attributes[:n]) for sc in scenarios]

    # latents for every scenario row, Plunkett-style (one RNG, scenario order)
    latents, weights = roll_weights(rule, scenarios, n, seed)
    columns = rule.latent_columns(n)
    weights.to_csv(out / "instilled_weights.csv", index=False)

    # trials for the first `instances` personas
    train_trials, val_trials = D.generate_decision_trials(
        scenarios[:instances], latents[:instances], rule, train, val, trials_seed
    )
    train_name, val_name = D.trial_file_names(instances)
    (out / train_name).write_text("\n".join(D.example_to_json(D.decision_example(t, rule.option_count)) for ts in train_trials for t in ts))
    (out / val_name).write_text("\n".join(D.example_to_json(D.decision_example(t, rule.option_count)) for ts in val_trials for t in ts))

    manifest = {
        "rule": rule.name,
        "params": rule.params,
        "option_count": rule.option_count,
        "seed": seed,
        "trials_seed": trials_seed,
        "attribute_count": n,
        "instances": instances,
        "train_trials_per_persona": train,
        "val_trials_per_persona": val,
        "source": str(source),
        "latent_columns": columns,
        "stats": rule.dataset_stats(train_trials, val_trials, latents[:instances], n),
        "files": {},
    }
    for path in sorted(out.iterdir()):
        if path.name != "manifest.json" and path.is_file():
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}: {len(scenarios)} latents, {instances} personas x ({train} train + {val} val) trials")
    if manifest["stats"]:
        print("stats:", json.dumps(manifest["stats"]))
    return manifest


def _parse_param(text):
    key, _, value = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError("--param expects key=value")
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    return key, value


def cli(build, description):
    """Shared command line: python -m src.rules.<name> --source data/plunkett --out data/<name> [options]."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", required=True, help="folder holding Plunkett's four files")
    parser.add_argument("--out", required=True, help="folder to write")
    parser.add_argument("--attribute-count", type=int, default=None)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1, help="latent seed (Plunkett's weights seed is 1)")
    parser.add_argument("--trials-seed", type=int, default=2, help="trial seed (Plunkett's is 2)")
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--param", type=_parse_param, action="append", default=[], help="rule parameter, key=value")
    args = parser.parse_args()
    rule = build(dict(args.param))
    make_dataset(
        rule,
        args.source,
        args.out,
        attribute_count=args.attribute_count,
        instances=args.instances,
        seed=args.seed,
        train=args.train,
        val=args.val,
        trials_seed=args.trials_seed,
    )
