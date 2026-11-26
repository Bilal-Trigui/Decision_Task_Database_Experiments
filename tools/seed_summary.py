"""Seeds of one persona side by side: each run's pooled readouts with intervals over report trials, then the spread
across seeds.

    python tools/seed_summary.py results/<run_a_seed4> results/<run_a_seed5> results/<run_a_seed6> ...
    python tools/seed_summary.py results/*nine-q-seed* --boot 2000 --pool 750 3000 --kind all

Runs are grouped by persona: their true weights and screen, read from each run's data folder (persona_rule), so the
runs of several personas can be passed at once. For each run, the pooled row of tools/persona_readouts.py (steps
--pool, replies of --kind) with a 95% interval from resampling the report trials. For each persona with two or more
runs, a `seeds` row: the mean over the runs, and `<readout>_sd`, `_min`, `_max` across them.

Two kinds of noise, read apart: an interval says how much a readout moves with which 50 trials were asked; the
spread across seeds says how much it moves with the training run. Seeds are not bootstrapped (three are too few
for a percentile); when the seeds spread wider than one run's interval, training noise dominates and a single run's
interval understates the uncertainty.

Writes seed_summary.csv (or --out) with every readout and prints the main ones.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import persona_rule  # noqa: E402
from tools.persona_readouts import run_readouts  # noqa: E402

SHOWN = ["cos4", "cos4_each", "spearman4", "spearman4_each", "neg_share"]


def persona_key(run_dir):
    rule = persona_rule.first(run_dir)
    if rule is None:
        raise SystemExit(f"{run_dir}: no persona rule found (settings.json or its data folder missing)")
    screen = None if rule.screen is None else (rule.screen.attribute, round(rule.screen.value, 4))
    return (tuple(float(w) for w in rule.weights), screen)


def summarise(runs, question="main", pool=(750, 3000), kind="all", n_boot=2000, boot_seed=0):
    groups = {}
    for run in runs:
        groups.setdefault(persona_key(run), []).append(run)
    rows = []
    for (weights, screen), members in groups.items():
        persona = f"weights {', '.join(f'{w:g}' for w in weights)}" + (f"; {screen[0]} >= {screen[1]:g}" if screen else "")
        per_run = []
        for run in members:
            out, _, _ = run_readouts(run, question, pool, n_boot, boot_seed)
            pooled = out[(out["step"].astype(str).str.startswith("pooled")) & (out["kind"] == kind)]
            if pooled.empty:
                print(f"  {Path(run).name}: no {kind} replies in steps {pool[0]} to {pool[1]}, left out")
                continue
            r = pooled.iloc[0].to_dict()
            settings = json.loads((Path(run) / "settings.json").read_text())
            r.update(persona=persona, run=Path(run).name, seed=settings["model_hyperparameters"]["seed"])
            per_run.append(r)
        rows += per_run
        if len(per_run) >= 2:
            frame = pd.DataFrame(per_run)
            names = [c for c in frame.columns if c not in ("persona", "run", "seed", "question", "step", "kind")
                     and not c.endswith(("_lo", "_hi")) and not c.startswith("boot_")]
            agg = {"persona": persona, "run": f"{len(per_run)} seeds", "seed": "seeds", "question": question,
                   "step": per_run[0]["step"], "kind": kind}
            for c in names:
                v = frame[c].astype(float)
                agg[c] = float(v.mean())
                agg[f"{c}_sd"], agg[f"{c}_min"], agg[f"{c}_max"] = float(v.std(ddof=1)), float(v.min()), float(v.max())
            rows.append(agg)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--question", default="main")
    ap.add_argument("--pool", nargs=2, type=int, default=[750, 3000])
    ap.add_argument("--kind", default="all", choices=["all", "both_pass"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default="seed_summary.csv")
    a = ap.parse_args()
    table = summarise(a.runs, a.question, tuple(a.pool), a.kind, a.boot, a.boot_seed)
    table.to_csv(a.out, index=False)
    for persona, part in table.groupby("persona", sort=False):
        print(f"\n{persona}: question {a.question}, {a.kind} replies, steps {a.pool[0]} to {a.pool[1]}")
        for _, r in part.iterrows():
            if r["seed"] == "seeds":
                cells = [f"{m} {r[m]:.2f} (sd {r[m + '_sd']:.2f}, {r[m + '_min']:.2f} to {r[m + '_max']:.2f})"
                         for m in SHOWN if m in r]
                print(f"  across {r['run']}: " + "; ".join(cells))
            else:
                cells = [f"{m} {r[m]:.2f}" + (f" [{r[m + '_lo']:.2f}, {r[m + '_hi']:.2f}]" if f"{m}_lo" in r else "")
                         for m in SHOWN if m in r]
                print(f"  seed {r['seed']} ({r['run'][:15]}): " + "; ".join(cells))
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
