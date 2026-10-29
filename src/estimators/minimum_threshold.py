"""Minimum threshold estimator: the rule read back as an if statement, at the best single attribute and cut.

The rule decides a trial in three branches, and the estimator fits exactly those:

    if both options fall below the cut     C, "neither"           (option_count 3)
    elif only one falls below it           the other one
    else                                   the weighted sum picks, A or B

With option_count 2 there is no C, and a trial both options fail goes to the weighted sum, as the
rule sends it. Given a screened attribute k and a cut level c, every trial is therefore either
decided by the screen, a fixed answer, or by the weights. The weights are fitted by Plunkett's
logistic regression on the trials the weights decide, the ones answered A or B. The screen has
nothing to fit: its answer is fixed by the branch.

The search. k and c are searched, not fitted, because the branch is a step function of c and no
gradient reaches it: every attribute at every level of the grid (every multiple of grid_step,
5 here, so 19 levels; 25, 50 and 75 for the paper's rule), one screened attribute at a time. Each
candidate is scored by the likelihood of all the choices under it, and the best is kept. The search
is exhaustive, so it cannot stop at a local best. Ties keep the first: the lowest attribute, then
the lowest level.

The likelihood. A model rarely follows a rule perfectly, and one wrong answer must not rule out the
right screen, so each choice follows the rule with probability 1 - e and is otherwise a lapse, any
of the option_count answers equally likely:

    P(choice) = (1 - e) * P_rule(choice) + e / option_count

P_rule is 1 for the screen's answer on a screened trial and 0 for any other, and the fitted
logistic probability on a weighed trial (0 for C there, which the rule never gives). e is the lapse
rate that fits this candidate best, from LAPSES. A candidate that names the right screen explains
its screened trials almost surely, at about log(1 - e) each, where a wrong one leaves them to a
weighted sum that cannot produce a C, so the right attribute and level win by a wide margin once a
few trials fall in each branch.

There is no "no cut" answer. Both rules this estimator reads give every persona exactly one screened
attribute (decision_rule.active_cuts 1, which config.resolve enforces), so "no screen" is an answer
known to be wrong. The cost is that the fit always names a screen, including for a model that learned
none. Read the attribute it names against the untrained model's, the base_model rows at step 0,
where the same search runs on choices that carry no screen.

The logistic fit is penalized with C = 1.0, the maximum a posteriori fit under Plunkett's N(0, 1)
prior, the same convention as the linear and interaction estimators.

Two blocks come back. 'main' is the weight vector, rescaled so the largest magnitude is 100,
because a weighted sum is a direction and its scale is not part of the object. 'cut' is the cut
vector in the same integer percentages of range the rule holds, 0 to 100, the level at the one
screened attribute and 0 elsewhere, and it is NOT rescaled, because a cut point is an absolute
level rather than a direction. Doubling a cut vector means a different screen, where doubling a
weight vector means the same preference. A persona whose choices are all one answer has nothing to
fit and returns all zeros, as in the base estimator.

Distance follows Table 1: cosine on 'main', and on 'cut' a scaled error, one minus the mean
absolute difference over the scale. That number's chance level sits well above zero, because
most cut points are zero under a sparse draw and a report of all zeros is mostly right, so the
cut row must be read against its measured chance column and never on its own.

Reads object.attribute_count, decision_rule.option_count (passed in by config.resolve),
model_estimating (grid_step).
"""
import numpy as np

from src.rules.minimum_threshold import cut_column, survives

from .base import Estimator, fit_logistic, rescale_to_100

CUT_SCALE = 100.0
REMOVED = ("sweeps", "min_gain")   # settings of the old penalized search, refused rather than ignored
LAPSES = (0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)   # the lapse rates a candidate is scored at, best kept

class MinimumThresholdEstimator(Estimator):
    name = "minimum_threshold"
    grid_step = 5
    single_cut = True        # fits exactly one screened attribute; config.resolve refuses a rule with more

    def __init__(self, attribute_count, grid_step=None, option_count=2):
        super().__init__(attribute_count)
        if grid_step is not None:
            self.grid_step = int(grid_step)
        if option_count not in (2, 3):
            raise ValueError(f"the minimum threshold estimator reads 2 or 3 options, not {option_count!r}")
        self.option_count = option_count

    def blocks(self):
        return {
            "main": [f"attr{i}" for i in range(1, self.n + 1)],
            "cut": [cut_column(i) for i in range(self.n)],
        }

    def features(self, A, B, mins, maxs):
        """The main-effect half of the design matrix, Plunkett's normalized differences.

        The screen column depends on the cut vector being tested, so the full design matrix is
        assembled inside `fit`. This returns the half that does not, and it returns the same thing
        every other estimator's `features` returns, a difference between the two options, so a
        caller reaching it through the base class gets a design matrix rather than one option's
        levels wearing the shape of one.
        """
        return (A - mins) / (maxs - mins) - (B - mins) / (maxs - mins)

    def _design(self, xa, xb, cuts):
        """Main-effect differences with the screen difference appended as the last column.

        Not used by the fit. It is the design the screen would add to a logistic regression, kept
        for step 3b's check of how tangled the screen is with the attribute it screens on.
        """
        screen = np.array(
            [float(survives(xa[t], cuts)) - float(survives(xb[t], cuts)) for t in range(len(xa))],
            dtype=float,
        )
        return np.column_stack([xa - xb, screen])

    def levels(self):
        """The cut levels the search tries, in percent of range: every multiple of grid_step strictly inside 0 to 100."""
        return list(range(self.grid_step, int(CUT_SCALE), self.grid_step))

    def candidates(self):
        """Every cut vector the search tries: one attribute at one grid level, the rest 0."""
        for k in range(self.n):
            for level in self.levels():
                cuts = np.zeros(self.n)
                cuts[k] = level
                yield cuts

    def branches(self, xa, xb, cuts):
        """Which branch of the rule decides each trial under one cut vector: the screen's answer, or '' for the weights.

        Returns an array of 'A', 'B', 'C' or '' per trial, '' where the weighted sum decides.
        """
        # survives(), for every trial at once: a cut of 0 is no constraint
        cuts = np.asarray(cuts, float)
        pass_a = np.all((cuts <= 0) | (xa >= cuts / 100.0), axis=1)
        pass_b = np.all((cuts <= 0) | (xb >= cuts / 100.0), axis=1)
        branch = np.full(len(xa), "", dtype=object)
        branch[pass_a & ~pass_b] = "A"
        branch[~pass_a & pass_b] = "B"
        if self.option_count == 3:
            branch[~pass_a & ~pass_b] = "C"
        return branch

    def _fit_at(self, xa, xb, choices, cuts, sample_weight):
        """The weights fitted where they decide, and the log likelihood of every choice, at one cut vector."""
        branch = self.branches(xa, xb, cuts)
        weighed = branch == ""
        ab = weighed & np.isin(choices, ("A", "B"))
        w = np.ones(len(choices)) if sample_weight is None else np.asarray(sample_weight, float)
        diff = xa - xb
        if ab.any():
            coef, intercept = fit_logistic(diff[ab], (choices[ab] == "A").astype(int),
                                           sample_weight=None if sample_weight is None else w[ab], with_intercept=True)
        else:
            coef, intercept = np.zeros(self.n), 0.0     # the screen decides every trial: no weights to fit
        p_a = 1.0 / (1.0 + np.exp(-(diff @ coef + intercept)))
        p_rule = np.where(weighed, np.where(choices == "A", p_a, np.where(choices == "B", 1.0 - p_a, 0.0)),
                          (branch == choices).astype(float))
        ll = max(float(np.sum(w * np.log((1.0 - e) * p_rule + e / self.option_count))) for e in LAPSES)
        return coef, ll

    def fit(self, A, B, choices, mins, maxs, sample_weight=None):
        n = self.n
        A, B = np.asarray(A, float), np.asarray(B, float)
        mins, maxs = np.asarray(mins, float), np.asarray(maxs, float)
        xa, xb = (A - mins) / (maxs - mins), (B - mins) / (maxs - mins)
        choices = np.asarray(choices, dtype=object)
        if len(set(choices)) < 2:
            return np.zeros(2 * n)            # no identifiable direction, as in the base estimator

        # Always the best attribute and cut: no "no cut" candidate, no penalty. Ties keep the first.
        best, coef, cuts = -np.inf, None, None
        for candidate in self.candidates():
            fitted, ll = self._fit_at(xa, xb, choices, candidate, sample_weight)
            if ll > best:
                best, coef, cuts = ll, fitted, candidate
        return np.concatenate([rescale_to_100(coef[:n]), cuts])

    def block_is_absolute(self, block):
        return block == "cut"

    def block_distance_name(self, block):
        return "scaled_error" if block == "cut" else "cosine"

    def block_distance(self, block, a, b):
        """Cosine on the weights, a scaled error on the cut points. See the module docstring."""
        if block != "cut":
            return Estimator.distance(a, b)
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or a.size == 0:
            return float("nan")
        return float(1.0 - np.mean(np.abs(a - b)) / CUT_SCALE)


def check_removed(params):
    """Refuse the old search's settings, so a file still carrying one does not silently run without it."""
    for key in REMOVED:
        if key in params:
            raise ValueError(
                f"model_estimating.{key} no longer exists: the cut search always keeps the best "
                "attribute and cut, with no penalty to tune. Remove it from the settings file."
            )


def build(params):
    check_removed(params)
    return MinimumThresholdEstimator(params["attribute_count"], grid_step=params.get("grid_step"),
                                     option_count=params.get("option_count", 2))
