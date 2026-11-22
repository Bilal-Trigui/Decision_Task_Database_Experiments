"""Do the trial-shown reports follow the trial in the prompt, and did training cause it? The trained model, its
step-0 base model and the shuffled-label control, through the same regression.

    python tools/trial_regression.py results/<trained_run> [--shuffled results/<shuffled_run>]
    python tools/trial_regression.py results/<trained_run> --shuffled results/<shuffled_run> --question main_trial_first

For every step of each model, each reported weight (the unscreened attributes, as persona_readouts) is regressed by
least squares on the trial shown with the question, the options' values put on 0 to 1 by each attribute's range:

  own   reported weight j ~ A_j + B_j          that attribute's two values in the trial
  all   reported weight j ~ all 10 values      every value in the trial

R² is the mean over the reported weights. Next to it:

  adj_r2      adjusted R², 1 - (1 - R²)(n - 1)/(n - p - 1)
  null_mean   mean R² over --perms permutations of which reply goes with which trial (the replies kept, the link
  null_95     to the trial broken), and its 95th percentile: what the regression finds in replies that do not
  p_perm      depend on the trial at all; p_perm is the share of permutations at or above the real R²

The null matters at this size. 50 report trials and 10 predictors give an expected R² of about 10/49 = 0.20 for
replies with no relation to the trial, so an R² of 0.2 at step 0 can be fitting noise, and a trained R² is only above
the prompt's when it is above both the step-0 and the shuffled model's null_95.

Rows per model and step, then one `pooled` row per model: steps --pool (inclusive), the R² and the null averaged
over those steps (each permutation the same across steps, so the pooled null is the null of the pooled mean).
Step 0 is the trained run's base_model_step-0, or --base RUN's when the trained run skipped it.

Reading (the reviewer's test): trained pooled R² well above the step-0 and shuffled values (and their null_95):
training caused the trial readout. All three alike: the readout is something the prompt induces in any model.

Writes trial_regression.csv in the trained run's folder (or --out) and prints the table.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import persona_rule  # noqa: E402

FILE = re.compile(r"weight_reports_(?P<tag>base_model|decision)_step-(?P<step>\d+)_paper_minimum_threshold_(?P<q>.+)\.csv$")
SPECS = ("own", "all")


def _r2(y, X):
    """R² of y on X with an intercept; nan when y does not vary."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    tot = (y - y.mean()) @ (y - y.mean())
    return np.nan if tot == 0 else float(1 - res @ res / tot)


def design(spec, A, B, j):
    """The predictors for reported weight j (an index into all n attributes)."""
    if spec == "own":
        return np.column_stack([A[:, j], B[:, j]])
    if spec == "all":
        return np.column_stack([A, B])
    raise ValueError(spec)


def mean_r2(spec, R, A, B, keep):
    vals = [_r2(R[:, k], design(spec, A, B, j)) for k, j in enumerate(keep)]
    return float(np.nanmean(vals)) if not all(np.isnan(vals)) else np.nan


def step_frames(run_dir, question):
    """{step: DataFrame} of one run's replies to `question`; step 0 is the base model."""
    out = {}
    for f in Path(run_dir).glob("weight_reports_*.csv"):
        m = FILE.match(f.name)
        if m and m["q"] == question:
            out[0 if m["tag"] == "base_model" else int(m["step"])] = pd.read_csv(f)
    return dict(sorted(out.items()))


def arrays(df, n, keep, lows, highs):
    cols = [f"report_attr{j + 1}" for j in keep]
    df = df.dropna(subset=cols)
    A = (df[[f"A_attribute_{i}" for i in range(1, n + 1)]].to_numpy(float) - lows) / (highs - lows)
    B = (df[[f"B_attribute_{i}" for i in range(1, n + 1)]].to_numpy(float) - lows) / (highs - lows)
    return df[cols].to_numpy(float), A, B


def model_rows(label, frames, n, keep, lows, highs, specs, perms, pool, seed):
    rows = []
    rng = np.random.default_rng(seed)
    # one permutation sequence per reply count, reused across steps so a pooled null averages like the pooled R²
    perm_cache = {}
    pooled = {s: {"r2": [], "null": []} for s in specs}
    for step, df in frames.items():
        R, A, B = arrays(df, n, keep, lows, highs)
        m = len(R)
        if m not in perm_cache:
            perm_cache[m] = [rng.permutation(m) for _ in range(perms)]
        for spec in specs:
            p_count = len(design(spec, A, B, keep[0]).T)
            if m <= p_count + 1:
                # as many trials as coefficients: any replies fit exactly, so R² says nothing
                print(f"  {label} step {step} {spec}: {m} replies for {p_count} predictors, R² left undefined")
                rows.append({"model": label, "step": step, "spec": spec, "n": m, "predictors": p_count,
                             "r2": np.nan, "adj_r2": np.nan})
                continue
            r2 = mean_r2(spec, R, A, B, keep)
            null = np.array([mean_r2(spec, R[p], A, B, keep) for p in perm_cache[m]]) if perms else np.array([])
            row = {"model": label, "step": step, "spec": spec, "n": m, "predictors": p_count, "r2": r2,
                   "adj_r2": (1 - (1 - r2) * (m - 1) / (m - p_count - 1)) if m > p_count + 1 else np.nan}
            if perms:
                row.update(null_mean=float(np.nanmean(null)), null_95=float(np.nanpercentile(null, 95)),
                           p_perm=float(np.mean(null >= r2)))
            rows.append(row)
            if pool[0] <= step <= pool[1] and step > 0:
                pooled[spec]["r2"].append(r2)
                pooled[spec]["null"].append(null)
    for spec in specs:
        got = pooled[spec]
        if not got["r2"]:
            continue
        r2 = float(np.mean(got["r2"]))
        row = {"model": label, "step": f"pooled {pool[0]}-{pool[1]}", "spec": spec, "n": np.nan,
               "predictors": np.nan, "r2": r2, "adj_r2": np.nan}
        if perms and len({len(x) for x in got["null"]}) == 1:
            null = np.mean(np.vstack(got["null"]), axis=0)
            row.update(null_mean=float(np.nanmean(null)), null_95=float(np.nanpercentile(null, 95)),
                       p_perm=float(np.mean(null >= r2)))
        rows.append(row)
    return rows


def run(trained, shuffled=None, base=None, question="main", specs=SPECS, perms=1000, pool=(750, 3000), seed=0):
    rule = persona_rule.first(trained)
    if rule is None:
        raise SystemExit(f"{trained}: no persona rule found (settings.json or its data folder missing)")
    n = len(rule.dims)
    keep = [i for i in range(n) if rule.screen is None or i != rule.screen.position - 1]
    lows, highs = np.asarray(rule.lows, float), np.asarray(rule.highs, float)
    frames = step_frames(trained, question)
    if not frames:
        raise SystemExit(f"{trained}: no weight_reports_*_{question}.csv files")
    base_frames = {0: frames.pop(0)} if 0 in frames else {}
    if not base_frames and base:
        base_frames = {k: v for k, v in step_frames(base, question).items() if k == 0}
    rows = model_rows("trained", frames, n, keep, lows, highs, specs, perms, pool, seed)
    if base_frames:
        rows += model_rows("step 0", base_frames, n, keep, lows, highs, specs, perms, (0, 0), seed)
    else:
        print(f"  no step-0 replies to {question} (pass --base RUN with a base_model_step-0 file)")
    if shuffled:
        sh = step_frames(shuffled, question)
        sh.pop(0, None)
        if not sh:
            raise SystemExit(f"{shuffled}: no weight_reports_*_{question}.csv files")
        rows += model_rows("shuffled", sh, n, keep, lows, highs, specs, perms, pool, seed)
    out = pd.DataFrame(rows)
    out.insert(0, "question", question)
    return out, [rule.dims[j] for j in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trained")
    ap.add_argument("--shuffled")
    ap.add_argument("--base", help="a run whose base_model_step-0 replies stand for step 0")
    ap.add_argument("--question", default="main")
    ap.add_argument("--spec", nargs="+", default=list(SPECS), choices=SPECS)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--pool", nargs=2, type=int, default=[750, 3000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    out, dims = run(a.trained, a.shuffled, a.base, a.question, a.spec, a.perms, tuple(a.pool), a.seed)
    path = Path(a.out) if a.out else Path(a.trained) / (
        "trial_regression.csv" if a.question == "main" else f"trial_regression_{a.question}.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n{Path(a.trained).name}: question {a.question}, reported weights {', '.join(dims)}")
    print(out.drop(columns="question").round(3).to_string(index=False))
    summary = out[(out["step"].astype(str).str.startswith("pooled")) | (out["model"] == "step 0")]
    print("\nthe reviewer's comparison (trained and shuffled pooled over steps, step 0 the base model):")
    for spec in a.spec:
        part = summary[summary["spec"] == spec].set_index("model")
        cells = [f"{m} R² {part.loc[m, 'r2']:.2f}" + (f" (null 95% {part.loc[m, 'null_95']:.2f})" if "null_95" in part else "")
                 for m in ("trained", "step 0", "shuffled") if m in part.index]
        print(f"  {spec}: " + "; ".join(cells))
    print(f"written {path}")


if __name__ == "__main__":
    main()
