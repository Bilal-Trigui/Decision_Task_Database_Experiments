"""A4 estimator: the screening form fitted by coordinate ascent over cut points.

The rule scores an option as a screen bonus plus a weighted sum, so with the cut vector held
fixed the utility difference between two options is linear in the parameters:

    U(a) - U(b) = g * (survives(a) - survives(b))  +  sum_k w_k * (xhat_k(a) - xhat_k(b))

Given c, fitting g and w is therefore exactly the logistic regression the rest of the repo uses,
with one extra column carrying the screen. Only c is awkward, because the indicator is a step
function of it and no gradient reaches it. This estimator searches c instead: sweep one attribute
at a time over a grid of levels, refit g and w at each candidate, and keep the level with the
best penalized log likelihood. A nonzero cut is accepted only when it buys more log likelihood than the penalty below, so an
attribute nothing is screened on keeps a cut of 0 rather than collecting a spurious one.

The penalty has to answer two things at once. A cut is one more parameter, which is the Bayesian
information criterion's half log of the trial count, and a cut is also chosen as the best of a
whole grid of levels, so the largest gain among many candidates beats a fixed threshold by luck
alone. The penalty is therefore half the log of the trial count plus the log of the number of
levels searched, with `min_gain` as a multiplier on the pair. A flat threshold fails in a way
that is easy to miss: it holds at fifty trials and then collects spurious second and third cuts
as the trial count rises, so the estimator gets worse with more data.

The fit is penalized with C = 1.0, the maximum a posteriori fit under Plunkett's N(0, 1) prior,
the same convention as the linear and interaction estimators. The paper calls this maximum
likelihood on the screening form; it is that with his prior left on.

Two blocks come back. 'main' is the weight vector, rescaled so the largest magnitude is 100,
because a weighted sum is a direction and its scale is not part of the object. 'cut' is the cut
vector in the same integer percentages of range the rule holds, 0 to 100, and it is NOT
rescaled, because a cut point is an absolute level rather than a direction. Doubling a cut
vector means a different screen, where doubling a weight vector means the same preference.

The screen gain g is a nuisance parameter. It is fitted and discarded, since the latent under
test is (w, c) and the report asks for those.

Distance follows Table 1: cosine on 'main', and on 'cut' a scaled error, one minus the mean
absolute difference over the scale. That number's chance level sits well above zero, because
most cut points are zero under a sparse draw and a report of all zeros is mostly right, so the
cut row must be read against its measured chance column and never on its own.

Reads object.attribute_count, model_estimating (grid_step, sweeps, min_gain).
"""
import numpy as np

from src.rules.tradeoff import cut_column, survives

from .base import Estimator, fit_logistic, rescale_to_100

CUT_SCALE = 100.0


def _log_likelihood(X, y, coef, intercept):
    """Bernoulli log likelihood of the fitted logistic model, computed stably."""
    z = X @ coef + intercept
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


class TradeoffEstimator(Estimator):
    name = "tradeoff"
    grid_step = 5
    sweeps = 3
    min_gain = 1.0           # multiplier on the acceptance penalty below, not a raw threshold

    def __init__(self, attribute_count, grid_step=None, sweeps=None, min_gain=None):
        super().__init__(attribute_count)
        if grid_step is not None:
            self.grid_step = int(grid_step)
        if sweeps is not None:
            self.sweeps = int(sweeps)
        if min_gain is not None:
            self.min_gain = float(min_gain)

    def blocks(self):
        return {
            "main": [f"attr{i}" for i in range(1, self.n + 1)],
            "cut": [cut_column(i) for i in range(self.n)],
        }

    def features(self, A, B, mins, maxs):
        """The main-effect half of the design matrix, Plunkett's normalized differences.

        The screen column depends on the cut vector being tested, so the full design matrix is
        assembled inside `fit`. This method exists so the estimator still answers the base
        class's question about its linear features.
        """
        return (A - mins) / (maxs - mins)

    def _design(self, xa, xb, cuts):
        """Main-effect differences with the screen difference appended as the last column."""
        screen = np.array(
            [float(survives(xa[t], cuts)) - float(survives(xb[t], cuts)) for t in range(len(xa))],
            dtype=float,
        )
        return np.column_stack([xa - xb, screen])

    def acceptance_penalty(self, n_trials, n_levels):
        """Log likelihood a nonzero cut must buy before it is kept. See the module docstring."""
        return self.min_gain * (0.5 * np.log(max(n_trials, 2)) + np.log(max(n_levels, 2)))

    def _fit_at(self, xa, xb, y, cuts, sample_weight):
        """Penalized fit of (w, g) at a fixed cut vector; returns the coefficients and its log likelihood."""
        X = self._design(xa, xb, cuts)
        coef, intercept = fit_logistic(X, y, sample_weight=sample_weight, with_intercept=True)
        return coef, _log_likelihood(X, y, coef, intercept)

    def fit(self, A, B, choices, mins, maxs, sample_weight=None):
        n = self.n
        A, B = np.asarray(A, float), np.asarray(B, float)
        mins, maxs = np.asarray(mins, float), np.asarray(maxs, float)
        xa, xb = (A - mins) / (maxs - mins), (B - mins) / (maxs - mins)
        y = np.array([1 if c == "A" else 0 for c in choices], dtype=int)
        if y.min() == y.max():
            return np.zeros(2 * n)            # no identifiable direction, as in the base estimator

        cuts = np.zeros(n)
        coef, best = self._fit_at(xa, xb, y, cuts, sample_weight)
        grid = list(range(self.grid_step, int(CUT_SCALE), self.grid_step))
        penalty = self.acceptance_penalty(len(y), len(grid))
        for _ in range(self.sweeps):
            moved = False
            for k in range(n):
                here = cuts[k]
                for level in grid:
                    if level == here:
                        continue
                    candidate = cuts.copy()
                    candidate[k] = level
                    _, ll = self._fit_at(xa, xb, y, candidate, sample_weight)
                    # A cut has to earn its parameter. Compare against this attribute being
                    # unscreened, not against the running best, so one attribute's cut cannot
                    # be held in place by another's improvement.
                    floor = best if here else best + penalty
                    if ll > floor:
                        cuts, best, moved = candidate, ll, True
            if not moved:
                break
        coef, best = self._fit_at(xa, xb, y, cuts, sample_weight)
        return np.concatenate([rescale_to_100(coef[:n]), cuts])

    def block_distance(self, block, a, b):
        """Cosine on the weights, a scaled error on the cut points. See the module docstring."""
        if block != "cut":
            return Estimator.distance(a, b)
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or a.size == 0:
            return float("nan")
        return float(1.0 - np.mean(np.abs(a - b)) / CUT_SCALE)


def build(params):
    return TradeoffEstimator(
        params["attribute_count"],
        grid_step=params.get("grid_step"),
        sweeps=params.get("sweeps"),
        min_gain=params.get("min_gain"),
    )
