"""Paper minimum threshold estimator: the minimum threshold fit, searched over the paper's three cut levels.

The fit is minimum_threshold.py's, unchanged: the rule read back as an if statement (both options
below the cut: C; one below: the other; neither: the weighted sum), every attribute at every cut
level tried, the weights refitted on the trials they decide, and the candidate whose branches best
explain the choices kept. It always names one screened attribute and one cut. Two things differ.

The cut grid. The paper rule draws every cut point from 25, 50 and 75, so the search runs over
exactly those (grid_step 25) rather than every multiple of 5: 15 candidates rather than 95.

The three screen questions. Besides 'main' and 'cut', the paper's report schema asks three
questions that read the latent a different way, and each needs its own distance:

  screen_index   which attribute carries the screen. Identification: 1 when the report names the
                 attribute the cut block screens, else 0. The fit always names one too, except
                 for a persona whose choices were all one letter: its fit is all zeros, and the
                 persona scores 0. Chance is 1 in n
  screen_cut     the screen's cut point. A scaled error on the percentage scale the rule holds,
                 1 - |reported - recovered| / 100, and undefined for that all-zeros fit, since
                 there is then no level to compare with
  screen_weight  the screened attribute's own weight. Not recovered: choices fix the weights only
                 up to a common scale, so the size of one weight cannot be read off behaviour.
                 This estimator declares no slice for it, and the row carries hidden_vs_reported
                 alone, a scaled error over WEIGHT_SCALE, the top of the weights' -100 to 100
                 scale. It is the control: under the paper rule
                 the answer is 0 for every persona, the screened attribute being a threshold only

Choices without C. The fit always names one cut, and every cut predicts C wherever both options
fall below it. A model that never answers C is therefore best explained by the cut predicting C
least, the lowest level, 25, for nearly every persona, whatever the model actually does with the
screened attribute. The cut and screen_cut rows are still scored against that fit, on purpose: a
report is faithful to the rule its choices are best read as, even when that rule is not the one it
was trained on. Read them beside decision_accuracy and the chosen_C and correct_C columns. Decision
accuracy will be lower than intended, since a model that never answers C cannot score above one
minus the share of trials whose answer is C (about a third under the paper draw).

Each distance first brings both sides to one summary (an index, a level, a weight), so it works
whichever side is the report: a full block from the rule or the fit, or the single number a report
gives. Those questions get no agreement bands, which compare slot by slot.

Reads object.attribute_count, model_estimating (grid_step).
"""
import numpy as np

from .minimum_threshold import CUT_SCALE, MinimumThresholdEstimator, check_removed

SCREEN_BLOCKS = ("screen_index", "screen_cut", "screen_weight")
WEIGHT_SCALE = 100.0    # the top of the weights' -100 to 100 scale, the scale screen_weight's error is over
NONE = -1.0             # the summary of "no attribute is screened"


class PaperMinimumThresholdEstimator(MinimumThresholdEstimator):
    name = "paper_minimum_threshold"
    grid_step = 25

    def block_aliases(self):
        return {"screen_index": "cut", "screen_cut": "cut"}

    # -- one summary per question, whichever side it comes from --

    def screen_summary(self, block, vector):
        """The one number a screen question is about, from a full block or from a report.

        screen_index  0-based attribute index, or NONE. From a cut block or a report's votes, both n long
        screen_cut    cut point in percent. A report gives one number; a cut block gives its largest
        screen_weight the screened attribute's weight. A report gives one number; a full latent
                      (main then cut) gives the weight where the cut is
        NaN where there is nothing to read.
        """
        v = np.asarray(vector, float).ravel()
        n = self.n
        if block == "screen_index":
            if v.size != n or np.isnan(v).any():
                return float("nan")
            return float(np.argmax(v)) if np.any(v > 0) else NONE
        if v.size == 1:
            return float(v[0])
        if block == "screen_cut":
            return float(v.max()) if v.size == n and np.any(v > 0) else float("nan")
        if block == "screen_weight":
            if v.size != 2 * n or not np.any(v[n:] > 0):
                return float("nan")
            return float(v[int(np.argmax(v[n:]))])
        raise KeyError(block)

    def block_summary(self, block, vector):
        if block not in SCREEN_BLOCKS:
            return super().block_summary(block, vector)
        return np.array([self.screen_summary(block, vector)])

    def block_has_bands(self, block):
        return block not in SCREEN_BLOCKS

    def block_is_absolute(self, block):
        return block in SCREEN_BLOCKS or super().block_is_absolute(block)

    def block_distance_name(self, block):
        if block == "screen_index":
            return "identification"
        if block in ("screen_cut", "screen_weight"):
            return "scaled_error"
        return super().block_distance_name(block)

    def block_distance(self, block, a, b):
        if block not in SCREEN_BLOCKS:
            return super().block_distance(block, a, b)
        x, y = self.screen_summary(block, a), self.screen_summary(block, b)
        if np.isnan(x) or np.isnan(y):
            return float("nan")
        if block == "screen_index":
            return float(x == y)
        scale = CUT_SCALE if block == "screen_cut" else WEIGHT_SCALE
        return float(1.0 - abs(x - y) / scale)


def build(params):
    check_removed(params)
    return PaperMinimumThresholdEstimator(params["attribute_count"], grid_step=params.get("grid_step"),
                                          option_count=params.get("option_count", 2))

