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
                 there is then no level to compare with. A report that states the cut on the
                 scale of an attribute it names (in units, or in percent of its range) is a cut
                 on that attribute: it scores 0 against a rule that screens another attribute,
                 and the scaled error against one that screens the same. See A named cut below
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

A named cut. A screen_cut report comes in two forms. A bare level (one number, the fraction the
screen_cut question and main_report ask for) names no attribute and is compared on the level alone.
A named cut (n + 1 numbers: the level in percent of the named attribute's range, then one vote per
attribute; the questions that ask for the cut in units or in percent) says which attribute the cut
is on, and the level means nothing on any other: "35 mg per cup" is not a statement about steeping
time. Compared on the level alone, a reply naming the wrong attribute with a value at the middle of
its range scored 0.75 against a cut at 25 or 75 and 1.0 against a cut at 50. So a named cut is first
compared with the attribute the other side screens, and scores 0 where they differ. Its rows carry
scaled_error_named_attribute in the distance column, and chance for them is a random rule stated the
same way (as_report). Two cut blocks (recovered against hidden, the row's recovery score) are still
compared on the level alone, with the attribute match in estimator_attribute_match_rate beside it;
the pooled correlation is over levels in both forms.

Reads object.attribute_count, model_estimating (grid_step).
"""
import numpy as np

from .minimum_threshold import CUT_SCALE, MinimumThresholdEstimator, check_removed

SCREEN_BLOCKS = ("screen_index", "screen_cut", "screen_weight")
WEIGHT_SCALE = 100.0    # the top of the weights' -100 to 100 scale, the scale screen_weight's error is over
NONE = -1.0             # the summary of "no attribute is screened"
NAMED = "named_attribute"   # the form of a screen_cut report that states the cut together with its attribute


class PaperMinimumThresholdEstimator(MinimumThresholdEstimator):
    name = "paper_minimum_threshold"
    grid_step = 25

    def block_aliases(self):
        return {"screen_index": "cut", "screen_cut": "cut"}

    # -- one summary per question, whichever side it comes from --

    def screen_summary(self, block, vector):
        """The one number a screen question is about, from a full block or from a report.

        screen_index  0-based attribute index, or NONE. From a cut block or a report's votes, both n long
        screen_cut    cut point in percent. A report gives one number, or n + 1 when it names the cut's
                      attribute (the level first); a cut block gives its largest
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
            if v.size == n + 1:
                return float(v[0])      # a named cut: the level, then the votes
            return float(v.max()) if v.size == n and np.any(v > 0) else float("nan")
        if block == "screen_weight":
            if v.size != 2 * n or not np.any(v[n:] > 0):
                return float("nan")
            return float(v[int(np.argmax(v[n:]))])
        raise KeyError(block)

    def cut_attribute(self, vector):
        """The 0-based attribute a cut is on, or None where the vector names none: a named cut's most voted
        attribute, a cut block's screened one, nothing for a bare level or an all-zero block."""
        v = np.asarray(vector, float).ravel()
        votes = v[1:] if v.size == self.n + 1 else v if v.size == self.n else np.zeros(0)
        return int(np.argmax(votes)) if np.any(votes > 0) else None

    def is_named_cut(self, vector):
        return np.size(vector) == self.n + 1

    def report_form(self, block, reports):
        if block == "screen_cut" and any(self.is_named_cut(v) for v in reports.values()):
            return NAMED
        return None

    def as_report(self, block, vector, form):
        """A cut block as a named cut would state it: its level, then a vote on the attribute it screens."""
        v = np.asarray(vector, float).ravel()
        if form != NAMED or block != "screen_cut" or v.size != self.n or not np.any(v > 0):
            return vector
        votes = np.zeros(self.n)
        votes[int(np.argmax(v))] = 1.0
        return np.concatenate([[v.max()], votes])

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
        if block == "screen_cut" and (self.is_named_cut(a) or self.is_named_cut(b)):
            # a cut stated with its attribute is a cut on that attribute: nothing agrees where the other side's is elsewhere
            i, j = self.cut_attribute(a), self.cut_attribute(b)
            if i is not None and j is not None and i != j:
                return 0.0
        scale = CUT_SCALE if block == "screen_cut" else WEIGHT_SCALE
        return float(1.0 - abs(x - y) / scale)


def build(params):
    check_removed(params)
    return PaperMinimumThresholdEstimator(params["attribute_count"], grid_step=params.get("grid_step"),
                                          option_count=params.get("option_count", 2))

