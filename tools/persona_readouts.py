"""The readouts the persona C to H predictions name, per step and pooled over steps 750 to 3000.

    python tools/persona_readouts.py results/<run_id> [results/<run_id> ...]
    python tools/persona_readouts.py results/<run_id> --question main_trial_first --pool 750 3000

For each weights question reply (default `main`, the trial-shown weights question), over the unscreened weights
(the four the persona's rule uses; all five if it has no screen), against the persona's true weights (read from
the run's data folder, as step_answers does):

  sign_<dim>       share of replies giving that weight its true sign (0 counts as wrong)    C, F: which sign wins
  pos_<dim>        share of replies giving that weight a positive number
  cos4             cosine of the mean reply with the true weights                            C: a sizes-only reply scores 0
  cos4_each        mean of each reply's own cosine with the true weights
  spearman4        Spearman rank correlation of the mean reply with the true weights        E positive: the order
  spearman4_each   mean of each reply's own Spearman correlation
  neg_share        share of the reported numbers that are negative                           E negative
  mean_abs         mean absolute reported weight, beside mean_abs_true                      D: small or inflated?

Rows per step, then one `pooled` row per kind (all replies from the steps in --pool, inclusive). Kinds: `all`
replies, and `both_pass` (both options in the reply's trial clear the screen: the trials the weights decide).
Writes results/<run_id>/persona_readouts.csv and prints the table.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import persona_rule  # noqa: E402

FILE = re.compile(r"weight_reports_(?P<stage>.+?)_step-(?P<step>\d+)_paper_minimum_threshold_(?P<q>.+)\.csv$")


def _spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cos(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx and ny else np.nan


def readout(rep, true, dims):
    """One row of readouts for an (n, k) array of replies against the k true weights."""
    row = {"n": len(rep)}
    if not len(rep):
        return row
    for j, d in enumerate(dims):
        row[f"sign_{d}"] = float(np.mean(np.sign(rep[:, j]) == np.sign(true[j])))
        row[f"pos_{d}"] = float(np.mean(rep[:, j] > 0))
    mean = rep.mean(axis=0)
    row["cos4"] = _cos(mean, true)
    row["cos4_each"] = float(np.nanmean([_cos(r, true) for r in rep]))
    row["spearman4"] = _spearman(mean, true)
    each = [_spearman(r, true) for r in rep]
    row["spearman4_each"] = float(np.nanmean(each)) if not all(np.isnan(each)) else np.nan
    row["neg_share"] = float(np.mean(rep < 0))
    row["mean_abs"] = float(np.abs(rep).mean())
    row["mean_abs_true"] = float(np.abs(true).mean())
    return row


def replies(run_dir, question):
    """{step: DataFrame of that step's replies to `question`}, base model as step 0."""
    out = {}
    for f in Path(run_dir).glob("weight_reports_*.csv"):
        m = FILE.match(f.name)
        if m and m["q"] == question:
            out[int(m["step"])] = pd.read_csv(f)
    return dict(sorted(out.items()))


def run_readouts(run_dir, question="main", pool=(750, 3000)):
    rule = persona_rule.first(run_dir)
    if rule is None:
        raise SystemExit(f"{run_dir}: no persona rule found (settings.json or its data folder missing)")
    n = len(rule.dims)
    keep = [i for i in range(n) if rule.screen is None or i != rule.screen.position - 1]
    dims = [rule.dims[i] for i in keep]
    true = np.asarray(rule.weights, float)[keep]
    lows, highs = np.asarray(rule.lows, float), np.asarray(rule.highs, float)
    by_step = replies(run_dir, question)
    if not by_step:
        raise SystemExit(f"{run_dir}: no weight_reports_*_{question}.csv files")
    rows, pooled = [], {"all": [], "both_pass": []}
    for step, df in by_step.items():
        cols = [f"report_attr{i + 1}" for i in keep]
        df = df.dropna(subset=cols)
        kinds = {"all": np.ones(len(df), bool)}
        if rule.screen is not None:
            j = rule.screen.position - 1
            cut = rule.screen.value
            kinds["both_pass"] = ((df[f"A_attribute_{j + 1}"] >= cut) & (df[f"B_attribute_{j + 1}"] >= cut)).to_numpy()
        for kind, mask in kinds.items():
            rep = df.loc[mask, cols].to_numpy(float)
            rows.append({"step": step, "kind": kind, **readout(rep, true, dims)})
            if pool[0] <= step <= pool[1]:
                pooled[kind].append(rep)
    for kind, reps in pooled.items():
        if reps:
            rows.append({"step": f"pooled {pool[0]}-{pool[1]}", "kind": kind,
                         **readout(np.vstack(reps), true, dims)})
    out = pd.DataFrame(rows)
    out.insert(0, "question", question)
    return out, dims, true


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--question", default="main")
    ap.add_argument("--pool", nargs=2, type=int, default=[750, 3000])
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for run in a.runs:
        out, dims, true = run_readouts(run, a.question, tuple(a.pool))
        path = Path(run) / ("persona_readouts.csv" if a.question == "main" else f"persona_readouts_{a.question}.csv")
        out.to_csv(path, index=False)
        print(f"\n{Path(run).name}: question {a.question}, true " + ", ".join(f"{d} {w:g}" for d, w in zip(dims, true)))
        print(out.drop(columns="question").round(2).to_string(index=False))
        print(f"written {path}")


if __name__ == "__main__":
    main()
