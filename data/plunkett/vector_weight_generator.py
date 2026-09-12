#!/usr/bin/env python3
"""Reroll a data folder's instilled_weights.csv under its rule, and nothing else.

    python data/plunkett/vector_weight_generator.py --data a3 --seed 7
    python data/a3/a3_weight_generator.py --seed 7              # the same, defaulting to that folder
    python data/a4/a4_weight_generator.py --seed 7 --rebuild    # reroll, then rebuild the dataset

This writes one file. The constructor turns the latent it writes into the training files, and the
two stay separate so a latent designed by hand enters exactly the way a drawn one does: edit
instilled_weights.csv yourself and run the constructor.

A reroll leaves the trial files describing the previous latent, because labels are written into
the JSONL when it is built and nothing re-reads the weights at training time. This says so and
prints the command to run next, or runs it for you with --rebuild.

What the draw produced, and any disagreement between an authored latent and the rule's declared
parameters, both come from the rule itself, so a new rule needs no new generator.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data" / "plunkett"))

from src import data as D  # noqa: E402
from src.config import load_component  # noqa: E402
from src.rules.base import _parse_param  # noqa: E402

from vector_dataset_constructor import construct, resolve_data_dir, rule_of  # noqa: E402

WEIGHTS = "instilled_weights.csv"


def generate(data_dir, rule_name=None, out_dir=None, seed=1, params=None, rebuild=False):
    """Draw one latent per scenario and write instilled_weights.csv. Returns the latents."""
    data_dir = resolve_data_dir(data_dir)
    out = Path(out_dir) if out_dir else data_dir
    rule = load_component("rules", rule_name or rule_of(data_dir), params or {})
    n = D.infer_attribute_count(data_dir)
    scenarios = D.load_scenarios(data_dir, n)
    latents, frame = rule.roll_weights(scenarios, n, seed)
    frame.to_csv(out / WEIGHTS, index=False)

    print(f"rerolled {out / WEIGHTS} under rule '{rule.name}' at seed {seed}")
    print(f"  parameters             {rule.params}")
    print(f"  personas               {len(latents)}")
    for line in rule.summarise_draw(latents, n):
        print(f"  {line}")
    for note in rule.audit(latents, n, None):
        print(f"WARNING: {note}")

    if rebuild:
        print()
        construct(data_dir, rule.name, out_dir, params=params)
    else:
        print()
        print("the training files in this folder now describe the previous latent. rebuild them with:")
        print(f"  python data/plunkett/vector_dataset_constructor.py --data {data_dir}")
    return latents


def main(default_data=None, default_rule=None):
    """The command line. A folder's own wrapper calls this with itself as the default."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=default_data, required=default_data is None,
                        help="folder to reroll")
    parser.add_argument("--out", default=None, help="where to write; default is the data folder")
    parser.add_argument("--rule", default=default_rule, help="default is what the folder's manifest says")
    parser.add_argument("--seed", type=int, default=1, help="latent seed (Plunkett's weights seed is 1)")
    parser.add_argument("--param", type=_parse_param, action="append", default=[], help="rule parameter, key=value")
    parser.add_argument("--rebuild", action="store_true", help="run the dataset constructor afterwards")
    a = parser.parse_args()
    generate(a.data, a.rule, a.out, a.seed, dict(a.param), a.rebuild)


if __name__ == "__main__":
    main()
