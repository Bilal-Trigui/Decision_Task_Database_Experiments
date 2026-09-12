"""Base class for decision rules, plus the shared dataset builder and its command line.

A rule declares the latent's blocks (named groups of columns), samples a latent,
scores an option, and labels a trial. `make_dataset` uses those to build a fresh
data folder from Plunkett's scenarios: the same personas and the same trials,
with the latents rerolled under the rule and every A/B label recomputed.

Reads: object.attribute_count (through `n`), decision_rule (the rule's own
parameters), model_training seeds when building a dataset.
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


class Rule:
    name = "base"
    defaults = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"rule '{self.name}' does not take parameters {sorted(unknown)}; it takes {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}

    # -- what the latent is --
    def blocks(self, n):
        """Ordered dict of block name -> latent column names. Every vector rule has a 'main' block."""
        raise NotImplementedError

    def latent_columns(self, n):
        return [c for cols in self.blocks(n).values() for c in cols]

    def block_aliases(self, n):
        """Extra names for blocks this rule already has, as alias -> real block.

        An alias is a second way of asking about one slice of the latent, not a second slice. It
        never appears in `blocks`, so it never widens instilled_weights.csv, and it exists so a
        report schema can collect the same block under a name that carries its own distance and
        its own chance level.
        """
        return {}

    def block_slices(self, n):
        slices, start = {}, 0
        for block, cols in self.blocks(n).items():
            slices[block] = slice(start, start + len(cols))
            start += len(cols)
        for alias, block in self.block_aliases(n).items():
            slices[alias] = slices[block]
        return slices

    # -- generator side --
    def sample_latent(self, n, rng):
        """Draw one hidden latent from the rule's target distribution, as a flat vector in latent_columns order."""
        raise NotImplementedError

    def score(self, latent, values, mins, maxs):
        """Utility of one option (raw attribute values) under the latent."""
        raise NotImplementedError

    def label(self, latent, trial):
        """Plunkett's selection rule: A if it scores strictly higher, otherwise B."""
        mins, maxs = trial.scenario.mins, trial.scenario.maxs
        utility_A = self.score(latent, trial.option_A.values, mins, maxs)
        utility_B = self.score(latent, trial.option_B.values, mins, maxs)
        return "A" if utility_A > utility_B else "B"

    def generate_trials(self, latent, scenario, n_trials, rng):
        """`n_trials` labelled trials for one persona from `rng`."""
        trials = [D.Trial(scenario, rng) for _ in range(n_trials)]
        for t in trials:
            t.label = self.label(latent, t)
        return trials

    def views(self, scenarios, latents, n):
        """Extra CSVs to write beside the training files: filename -> list of row dicts.

        A rule whose latent has structure worth reading returns it here and the constructor
        writes it. The weighted sum has none, so the default is nothing.
        """
        return {}

    def audit(self, latents, n, stats):
        """Lines where an authored latent disagrees with this rule's declared parameters.

        Printed rather than raised, since a latent designed by hand may depart from the draw on
        purpose. `stats` is whatever dataset_stats returned, so a rule can also warn that its own
        structure is doing nothing.
        """
        return []

    def summarise_draw(self, latents, n):
        """Lines describing what a draw produced, for the weight generator to print."""
        return []

    def dataset_stats(self, train_trials, val_trials, latents, n):
        """Rule-specific summary written to manifest.json. Empty by default."""
        return {}

    # -- dataset builder --
    def roll_weights(self, scenarios, n, seed):
        """Draw one latent per scenario from this rule's target distribution, in scenario order.

        Returns the latents and the instilled_weights.csv frame built from them. One RNG is
        seeded once and consumed in scenario order, which is Plunkett's construction, so the
        linear rule at his seed reproduces his file. Columns that came out whole are written as
        integers, as his are.
        """
        rng = random.Random(seed)
        latents = np.vstack([self.sample_latent(n, rng) for _ in scenarios])
        columns = self.latent_columns(n)
        weights = pd.DataFrame(latents, columns=columns)
        for c in columns:
            if np.all(np.equal(np.mod(weights[c], 1), 0)):
                weights[c] = weights[c].astype(int)
        weights.insert(0, "scenario", [sc.short_name for sc in scenarios])
        return latents, weights

    def make_dataset(self, source_dir, out_dir, attribute_count=None, instances=100, seed=1, train=50, val=10, trials_seed=2):
        """Build a data folder for this rule from Plunkett's scenarios.

        Keeps scenario_definitions.json, roles.csv and scenarios.csv (truncated to
        `attribute_count` attributes), rerolls instilled_weights.csv under this rule
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
        latents, weights = self.roll_weights(scenarios, n, seed)
        columns = self.latent_columns(n)
        weights.to_csv(out / "instilled_weights.csv", index=False)

        # trials for the first `instances` personas
        train_trials, val_trials = D.generate_decision_trials(
            scenarios[:instances], latents[:instances], self, train, val, trials_seed
        )
        train_name, val_name = D.trial_file_names(instances)
        (out / train_name).write_text("\n".join(D.example_to_json(D.decision_example(t)) for ts in train_trials for t in ts))
        (out / val_name).write_text("\n".join(D.example_to_json(D.decision_example(t)) for ts in val_trials for t in ts))

        manifest = {
            "rule": self.name,
            "params": self.params,
            "seed": seed,
            "trials_seed": trials_seed,
            "attribute_count": n,
            "instances": instances,
            "train_trials_per_persona": train,
            "val_trials_per_persona": val,
            "source": str(source),
            "latent_columns": columns,
            "stats": self.dataset_stats(train_trials, val_trials, latents[:instances], n),
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


def check_manifest(data_dir, rule, cfg):
    """Fail loudly if a folder's manifest says it was built under other rule parameters. Reads decision_rule."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest.get("rule") != rule.name:
        print(
            f"warning: {path} says the folder was built with rule '{manifest.get('rule')}' "
            f"but decision_rule.type is '{rule.name}'"
        )
        return
    if manifest.get("params", {}) != rule.params:
        raise D.DataFormatError(
            f"{path} was built with decision_rule parameters {manifest.get('params')} "
            f"but the settings say {rule.params}; rebuild the folder or change the settings"
        )
    # The manifest already records what each input hashed to when the training files were built.
    # Checking it is what stops the worst failure in the whole setup: edit instilled_weights.csv,
    # forget to rebuild, and the model trains on labels from the old latent while every report is
    # scored against the new one. Nothing else notices, because the labels live in the JSONL.
    for name, recorded in (manifest.get("inputs") or {}).items():
        current = Path(data_dir) / name
        if not current.exists():
            continue
        now = hashlib.sha256(current.read_bytes()).hexdigest()[: len(recorded)]
        if now != recorded:
            raise D.DataFormatError(
                f"{current} has changed since {path} was written ({recorded} -> {now}). The trial "
                f"labels in this folder still describe the previous latent. Rebuild the folder with "
                f"its dataset constructor, or restore the file."
            )


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
    rule.make_dataset(
        args.source,
        args.out,
        attribute_count=args.attribute_count,
        instances=args.instances,
        seed=args.seed,
        train=args.train,
        val=args.val,
        trials_seed=args.trials_seed,
    )
