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
When a pooled step lost a reply that did not parse, the pooled row is worked out on the replies every pooled step
has, lined up by trial, so that one permutation still moves the same replies at every step; `n` says how many.
A reply that is the same at every trial scores R² 0 with p_perm 1: it does not follow the trial.
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
    """R² of y on X with an intercept. 0 when y does not vary: a reply that is the same at every trial follows
    the trial not at all. Its permutations are that same reply, so its null is 0 too and p_perm comes out 1."""
    if np.ptp(y) == 0:      # max == min; (y - mean) is not exactly 0 for every constant y in floating point
        return 0.0
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    tot = (y - y.mean()) @ (y - y.mean())
    return float(1 - res @ res / tot)


def _p_perm(null, r2):
    """The share of permutations at or above the real R²; nan when the R² or every permutation is undefined."""
    null = np.asarray(null, float)
    null = null[~np.isnan(null)]
    return float(np.mean(null >= r2)) if len(null) and not np.isnan(r2) else np.nan


def _null_columns(null, r2):
    return {"null_mean": float(np.nanmean(null)), "null_95": float(np.nanpercentile(null, 95)), "p_perm": _p_perm(null, r2)}


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


def trial_keys(df):
    """One label per reply: the trial it was asked about (its options' values, A then B) and which of that trial's
    replies it is, so the same reply is found again at another step."""
    cols = [c for c in df.columns if c.startswith(("A_attribute_", "B_attribute_"))]
    trial = df[cols].astype(str).agg("|".join, axis=1)
    return (trial + "#" + trial.groupby(trial).cumcount().astype(str)).to_numpy()


def arrays(df, n, keep, lows, highs):
    """(replies, A, B, keys) for the rows with a readable reply; keys are trial_keys, taken before rows are dropped."""
    cols = [f"report_attr{j + 1}" for j in keep]
    df = df.assign(_key=trial_keys(df)).dropna(subset=cols)
    A = (df[[f"A_attribute_{i}" for i in range(1, n + 1)]].to_numpy(float) - lows) / (highs - lows)
    B = (df[[f"B_attribute_{i}" for i in range(1, n + 1)]].to_numpy(float) - lows) / (highs - lows)
    return df[cols].to_numpy(float), A, B, df["_key"].to_numpy()


def pooled_row(spec, parts, keep, perms, perm_cache, rng):
    """The pooled R² and its null for one spec, from the pooled steps' (replies, A, B, keys, r2, null).

    The pooled null is the null of the pooled mean only when one permutation moves the same replies at every
    step. That holds when every step has the same readable replies in the same order: then the steps' own R² and
    nulls are averaged. When a step lost a reply (one that did not parse), the steps are first cut down to the
    replies every step has, lined up by trial, and the R² and the null are worked out again on those. Returns
    (columns, replies used, a note or None)."""
    others = [set(part[3]) for part in parts[1:]]
    order = [k for k in parts[0][3] if all(k in keys for keys in others)]
    if all(list(part[3]) == order for part in parts):
        r2 = float(np.mean([part[4] for part in parts]))
        row = {"n": len(order), "r2": r2}
        if perms:
            row.update(_null_columns(np.mean(np.vstack([part[5] for part in parts]), axis=0), r2))
        return row, None
    cut = []
    for R, A, B, keys, _, _ in parts:
        at = {k: i for i, k in enumerate(keys)}
        idx = [at[k] for k in order]
        cut.append((R[idx], A[idx], B[idx]))
    m = len(order)
    note = f"pooled over the {m} replies every pooled step has (steps had {sorted({len(part[0]) for part in parts})})"
    p_count = len(design(spec, parts[0][1], parts[0][2], keep[0]).T)
    if m <= p_count + 1:
        return {"n": m, "r2": np.nan}, note + f": too few for {p_count} predictors, left undefined"
    r2 = float(np.mean([mean_r2(spec, R, A, B, keep) for R, A, B in cut]))
    row = {"n": m, "r2": r2}
    if perms:
        if m not in perm_cache:
            perm_cache[m] = [rng.permutation(m) for _ in range(perms)]
        null = np.array([np.mean([mean_r2(spec, R[p], A, B, keep) for R, A, B in cut]) for p in perm_cache[m]])
        row.update(_null_columns(null, r2))
    return row, note


def model_rows(label, frames, n, keep, lows, highs, specs, perms, pool, seed):
    rows = []
    rng = np.random.default_rng(seed)
    # one permutation sequence per reply count, reused across steps so a pooled null averages like the pooled R²
    perm_cache = {}
    pooled = {s: [] for s in specs}
    for step, df in frames.items():
        R, A, B, keys = arrays(df, n, keep, lows, highs)
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
                row.update(_null_columns(null, r2))
            rows.append(row)
            if pool[0] <= step <= pool[1] and step > 0:
                pooled[spec].append((R, A, B, keys, r2, null))
    for spec in specs:
        if not pooled[spec]:
            continue
        got, note = pooled_row(spec, pooled[spec], keep, perms, perm_cache, rng)
        if note:
            print(f"  {label} {spec}: {note}")
        rows.append({"model": label, "step": f"pooled {pool[0]}-{pool[1]}", "spec": spec, "predictors": np.nan,
                     "adj_r2": np.nan, **got})
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
