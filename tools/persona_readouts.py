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

    python tools/persona_readouts.py results/<run_id> --boot 2000

adds a 95% percentile interval to every readout, `<readout>_lo` and `<readout>_hi`, by resampling report trials with
replacement (seed --boot-seed, 0 by default). A trial is one pair of options shown with the question (the A_ and B_
columns); every step asks the same 50, so in a pooled row a trial's replies at all the pooled steps are drawn together.
Drawing replies one by one there would treat 10 replies to one trial as 10 independent trials, and give intervals about
three times too narrow. The interval covers which trials were asked, not training: two seeds of one persona can differ
by more than it (tools/seed_summary.py sets the two side by side).
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


def _each(rep, true):
    """Every reply's own cosine and Spearman correlation with the true weights (the *_each readouts average these)."""
    return (np.array([_cos(r, true) for r in rep], float), np.array([_spearman(r, true) for r in rep], float))


def _nanmean(x):
    return float(np.nanmean(x)) if len(x) and not np.all(np.isnan(x)) else np.nan


def readout(rep, true, dims, each=None):
    """One row of readouts for an (n, k) array of replies against the k true weights.

    `each` is _each(rep, true) when the caller already has it (the bootstrap reuses it for every resample)."""
    row = {"n": len(rep)}
    if not len(rep):
        return row
    cos_each, sp_each = _each(rep, true) if each is None else each
    for j, d in enumerate(dims):
        row[f"sign_{d}"] = float(np.mean(np.sign(rep[:, j]) == np.sign(true[j])))
        row[f"pos_{d}"] = float(np.mean(rep[:, j] > 0))
    mean = rep.mean(axis=0)
    row["cos4"] = _cos(mean, true)
    row["cos4_each"] = _nanmean(cos_each)
    row["spearman4"] = _spearman(mean, true)
    row["spearman4_each"] = _nanmean(sp_each)
    row["neg_share"] = float(np.mean(rep < 0))
    row["mean_abs"] = float(np.abs(rep).mean())
    row["mean_abs_true"] = float(np.abs(true).mean())
    return row


def trial_keys(df):
    """One label per reply naming the trial it was asked about: its options' values, A then B."""
    cols = [c for c in df.columns if c.startswith(("A_attribute_", "B_attribute_"))]
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy()


def bootstrap(rep, keys, true, dims, n_boot, seed=0, level=0.95):
    """{<readout>_lo, <readout>_hi} for every readout, resampling trials (all of a trial's replies together).

    Percentile intervals over n_boot resamples of the distinct trials in `keys`, each resample as many trials as
    there are. A readout a resample leaves undefined (a zero mean reply has no cosine) is left out of that
    readout's percentiles; `boot_undefined_<readout>` counts how often, when it happens."""
    if not len(rep) or n_boot <= 0:
        return {}
    each = _each(rep, true)
    uniq, inverse = np.unique(keys, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        draws.append(readout(rep[idx], true, dims, (each[0][idx], each[1][idx])))
    out, tail = {"boot_trials": len(uniq)}, (1 - level) / 2 * 100
    for name in draws[0]:
        if name in ("n", "mean_abs_true"):
            continue
        vals = np.array([d[name] for d in draws], float)
        ok = vals[~np.isnan(vals)]
        out[f"{name}_lo"], out[f"{name}_hi"] = (np.percentile(ok, [tail, 100 - tail]) if len(ok) else (np.nan, np.nan))
        if len(ok) < len(vals):
            out[f"boot_undefined_{name}"] = int(len(vals) - len(ok))
    return out


def replies(run_dir, question):
    """{step: DataFrame of that step's replies to `question`}, base model as step 0."""
    out = {}
    for f in Path(run_dir).glob("weight_reports_*.csv"):
        m = FILE.match(f.name)
        if m and m["q"] == question:
            out[int(m["step"])] = pd.read_csv(f)
    return dict(sorted(out.items()))


def run_readouts(run_dir, question="main", pool=(750, 3000), n_boot=0, boot_seed=0):
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
        keys = trial_keys(df)
        kinds = {"all": np.ones(len(df), bool)}
        if rule.screen is not None:
            j = rule.screen.position - 1
            cut = rule.screen.value
            kinds["both_pass"] = ((df[f"A_attribute_{j + 1}"] >= cut) & (df[f"B_attribute_{j + 1}"] >= cut)).to_numpy()
        for kind, mask in kinds.items():
            rep = df.loc[mask, cols].to_numpy(float)
            rows.append({"step": step, "kind": kind, **readout(rep, true, dims),
                         **bootstrap(rep, keys[mask], true, dims, n_boot, boot_seed)})
            if pool[0] <= step <= pool[1]:
                pooled[kind].append((rep, keys[mask]))
    for kind, parts in pooled.items():
        if parts:
            rep, keys = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            rows.append({"step": f"pooled {pool[0]}-{pool[1]}", "kind": kind, **readout(rep, true, dims),
                         **bootstrap(rep, keys, true, dims, n_boot, boot_seed)})
    out = pd.DataFrame(rows)
    out.insert(0, "question", question)
    return out, dims, true


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--question", default="main")
    ap.add_argument("--pool", nargs=2, type=int, default=[750, 3000])
    ap.add_argument("--boot", type=int, default=0, help="bootstrap resamples of the report trials (0: no intervals)")
    ap.add_argument("--boot-seed", type=int, default=0)
    a = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for run in a.runs:
        out, dims, true = run_readouts(run, a.question, tuple(a.pool), a.boot, a.boot_seed)
        path = Path(run) / ("persona_readouts.csv" if a.question == "main" else f"persona_readouts_{a.question}.csv")
        out.to_csv(path, index=False)
        print(f"\n{Path(run).name}: question {a.question}, true " + ", ".join(f"{d} {w:g}" for d, w in zip(dims, true)))
        shown = out.drop(columns="question")
        if a.boot:   # the table with intervals is too wide to print: point values here, intervals in the csv and below
            shown = shown[[c for c in shown.columns if not c.endswith(("_lo", "_hi")) and not c.startswith("boot_")]]
        print(shown.round(2).to_string(index=False))
        if a.boot:
            print(f"\n95% intervals over report trials ({a.boot} resamples), pooled rows:")
            for _, r in out[out["step"].astype(str).str.startswith("pooled")].iterrows():
                names = [c[:-3] for c in out.columns if c.endswith("_lo") and f"{c[:-3]}_hi" in out.columns]
                print(f"  {r['kind']} ({int(r['boot_trials'])} trials): "
                      + "; ".join(f"{m} {r[m]:.2f} [{r[m + '_lo']:.2f}, {r[m + '_hi']:.2f}]" for m in names))
        print(f"written {path}")


if __name__ == "__main__":
    main()
