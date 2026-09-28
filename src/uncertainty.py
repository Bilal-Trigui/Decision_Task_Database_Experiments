"""How much to trust a number: bootstrap confidence intervals and a collinearity check.

Three questions the point estimates in results.csv cannot answer on their own.

1. How sure are we of a faithfulness mean?  `bootstrap_mean_distance`
   Faithfulness is a mean over the personas whose reports parsed and could be scored. When that
   is a small set (27 of 100 at a parse collapse), the mean depends a lot on which personas
   happened to parse. Resampling those personas with replacement and recomputing the mean many
   times gives a percentile interval. This is cheap: the per-persona distances are computed once
   and only resampled.

2. How stable is one persona's recovered latent?  `bootstrap_recovered`
   The recovered latent is fitted from one persona's A/B choices on the verification trials.
   Resampling those trials and refitting shows how far each recovered number moves. For the
   linear and interaction estimators this is quick. For the tradeoff estimator every refit runs
   the full cut-point search, so keep n_boot small (100 to 200) or run it offline on a few
   personas.

3. Are the columns the estimator fits on too tangled to separate?  `design_correlation`
   If two columns of the design matrix move together, logistic regression cannot tell which one
   the choices respond to, and the weights it splits between them are unstable. Plunkett's option
   values are drawn independently, so the main-effect columns should be close to uncorrelated.
   The column to watch is A4's screen column: whether an option survives the screen depends on
   the screened attribute's level, so the screen column is correlated with that attribute's
   difference column by construction. With zero_cut_main_effects the true weight on that
   attribute is 0, and any weight the fit puts there may be screen signal leaking across.

Every function takes the repo's own estimator, so distances and features match evaluate.py.
"""
import numpy as np

LEVEL = 0.95


def _interval(samples, level=LEVEL):
    tail = (1.0 - level) / 2.0 * 100.0
    samples = np.asarray(samples, float)
    lo, hi = np.nanpercentile(samples, [tail, 100.0 - tail], axis=0)
    return lo, hi


# --- 1. faithfulness mean over personas ------------------------------------------------


def bootstrap_mean_distance(estimator, block, pairs, n_boot=2000, seed=0, level=LEVEL):
    """Percentile interval on the mean block distance, resampling personas.

    `pairs` is the same list of (vector, vector) per persona that evaluate._scored takes. Pairs
    with no defined distance are dropped first, exactly as _scored drops them, so the interval
    describes the same personas as the mean it sits next to.
    """
    values = np.array([estimator.block_distance(block, a, b) for a, b in pairs], float)
    values = values[~np.isnan(values)]
    out = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(values.size)}
    if values.size == 0:
        return out
    out["mean"] = float(values.mean())
    if values.size < 2:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = _interval(means, level)
    out["ci_low"], out["ci_high"] = float(lo), float(hi)
    return out


# --- 2. one persona's recovered latent --------------------------------------------------


def bootstrap_recovered(estimator, A, B, choices, mins, maxs, n_boot=200, seed=0, level=LEVEL):
    """Refit one persona's latent on resampled trials.

    Returns the full fit, the per-component interval, and for the 'main' block a direction
    stability score: the cosine between each refit's weights and the full fit's weights. A median
    near 1 with a 5th percentile near 1 means the direction barely moves. A low 5th percentile
    means some resamples point somewhere quite different.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    choices = np.asarray(choices)
    full = np.asarray(estimator.fit(A, B, list(choices), mins, maxs), float)
    rng = np.random.default_rng(seed)
    t = len(choices)
    draws = np.empty((n_boot, full.size))
    for b in range(n_boot):
        idx = rng.integers(0, t, size=t)
        draws[b] = estimator.fit(A[idx], B[idx], list(choices[idx]), mins, maxs)
    lo, hi = _interval(draws, level)
    main = estimator.block_slices()["main"]
    cosines = np.array([estimator.distance(d[main], full[main]) for d in draws])
    return {
        "full": full,
        "draws": draws,
        "ci_low": lo,
        "ci_high": hi,
        "columns": estimator.latent_columns(),
        "main_stability_median": float(np.nanmedian(cosines)),
        "main_stability_p05": float(np.nanpercentile(cosines, 5)),
    }


# --- 3. collinearity of the design matrix -----------------------------------------------


def design_matrix(estimator, A, B, mins, maxs, cuts=None):
    """The matrix the estimator actually fits on, with column names.

    For the tradeoff estimator, pass the cut vector to include the screen column the fit uses at
    those cuts. Without cuts, or for any other estimator, this is `estimator.features`.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    mins, maxs = np.asarray(mins, float), np.asarray(maxs, float)
    main_names = estimator.blocks()["main"]
    if cuts is not None and hasattr(estimator, "_design"):
        xa, xb = (A - mins) / (maxs - mins), (B - mins) / (maxs - mins)
        return estimator._design(xa, xb, np.asarray(cuts, float)), list(main_names) + ["screen"]
    X = estimator.features(A, B, mins, maxs)
    names = estimator.latent_columns()[: X.shape[1]]
    return X, list(names)


def design_correlation(X, names):
    """Pairwise correlations between design columns, the worst pair, and variance inflation.

    A constant column (a screen no trial ever splits, say) has no correlation and is reported as
    such rather than crashing the check. Rough reading: |r| under 0.3 is fine, 0.3 to 0.7 is worth
    noting, above 0.7 the two weights cannot be trusted separately. A variance inflation factor
    above 5 says the same thing about one column against all the others together.
    """
    X = np.asarray(X, float)
    live = np.std(X, axis=0) > 0
    corr = np.full((X.shape[1], X.shape[1]), np.nan)
    if live.sum() >= 2:
        corr[np.ix_(live, live)] = np.corrcoef(X[:, live], rowvar=False)
    worst = (float("nan"), None, None)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr[i, j]
            if not np.isnan(r) and (worst[1] is None or abs(r) > abs(worst[0])):
                worst = (float(r), names[i], names[j])
    vif = np.full(X.shape[1], np.nan)
    if live.sum() >= 2:
        sub = corr[np.ix_(live, live)]
        try:
            vif[live] = np.diag(np.linalg.inv(sub))
        except np.linalg.LinAlgError:
            vif[live] = np.inf
    return {
        "corr": corr,
        "names": list(names),
        "max_abs_r": abs(worst[0]) if worst[1] else float("nan"),
        "worst_pair": (worst[1], worst[2]),
        "worst_r": worst[0],
        "vif": vif,
        "dead_columns": [n for n, ok in zip(names, live) if not ok],
    }


def screen_correlation_row(estimator, A, B, mins, maxs, cuts, persona=None):
    """One summary row for one persona under A4: how tangled is the screen with what it screens on.

    `cuts` should be the persona's hidden cut vector for the question "does the task itself tangle
    these columns", or the recovered one for "did the fit work on tangled columns". The row
    records the correlation between the screen column and each screened attribute's column, the
    worst pair overall, and the largest variance inflation factor.
    """
    X, names = design_matrix(estimator, A, B, mins, maxs, cuts=cuts)
    c = design_correlation(X, names)
    n = len(names) - 1
    screened = [i for i in range(n) if float(cuts[i]) > 0]
    row = {
        "persona": persona,
        "screened": ",".join(names[i] for i in screened),
        "screen_split_share": float(np.mean(X[:, -1] != 0)),
        "max_abs_r": c["max_abs_r"],
        "worst_pair": "/".join(p for p in c["worst_pair"] if p),
        "max_vif": float(np.nanmax(c["vif"])) if np.any(~np.isnan(c["vif"])) else float("nan"),
    }
    for i in screened:
        row[f"r_screen_{names[i]}"] = float(c["corr"][n, i])
    return row
