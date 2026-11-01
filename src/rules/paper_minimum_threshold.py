"""The paper's minimum threshold rule: the general threshold rule with Plunkett's weights and a pure screen.

The rule is threshold.py's, unchanged: an option below the cut point on the screened attribute
loses outright, and options that pass are ranked by the weighted sum. What this file sets is the
draw, so the choices it labels are easy to decide and the latent is easy to name:

  weight_range           null         Plunkett's draw, the default: every weight uniform on -100 to 100,
                                      rescaled so the largest magnitude is 100, so weights vary on both
                                      sides of 0
  min_weight_magnitude   20           every unscreened weight at least 20 from 0, so none is near
                                      irrelevant: from -100 to -20 or from 20 to 100
  zero_cut_main_effects  true         the screened attribute's own weight is exactly 0, so it acts as a
                                      threshold and nothing else: what the model can learn about that
                                      attribute is the presence of the screen alone
  screened_weight        null         (the template's other way of setting that weight, unused here)
  cut_levels             [25, 50, 75] the cut point is one of three levels, a quarter, a half or three
                                      quarters of the attribute's range: the discrete draw
  cut_range              [25, 75]     the span the levels sit in
  active_cuts            1            one screened attribute per persona
  option_count           3            A, B, or C, "neither", when both options fall below the cut
  neither_share          0.2          a fifth of each persona's training and validation trials are
                                      answered C, the rest A or B, whatever the persona's cut
  weight_decided_share   null         the share of training and validation trials both options pass,
                                      decided by the weights alone; null takes them as they come
                                      (about 29%). See threshold.py

Any of these can be changed in the settings file. weight_range [low, high] draws every weight
uniform on that span instead, to two decimals. To give the screened attribute a weight of its own,
set zero_cut_main_effects to false and screened_weight to that weight.

The extra report blocks. Besides 'main' and 'cut', a latent here is read three more ways, each
the answer to one of the paper's introspection questions (see src/reports/paper_minimum_threshold.py):
  screen_index   which attribute carries the screen    read off the cut block
  screen_cut     the screen's cut point                read off the cut block
  screen_weight  the screened attribute's own weight   read off main and cut together, so its
                                                       slice is the whole latent
They never widen instilled_weights.csv, which keeps attr1..attrN and cut1..cutN.
Reads decision_rule, object.attribute_count.
"""
from .build import cli
from .threshold import ThresholdRule


class PaperMinimumThresholdRule(ThresholdRule):
    name = "paper_minimum_threshold"
    defaults = {
        "active_cuts": 1,
        "cut_range": [25, 75],
        "cut_levels": [25, 50, 75],
        "weight_range": None,
        "min_weight_magnitude": 20,
        "screened_weight": None,
        "zero_cut_main_effects": True,
        "option_count": 3,
        "neither_share": 0.2,
        "weight_decided_share": None,
    }

    def block_aliases(self, n):
        return {"screen_index": "cut", "screen_cut": "cut"}

    def block_slices(self, n):
        slices = super().block_slices(n)
        slices["screen_weight"] = slice(0, 2 * n)
        return slices


def build(params):
    return PaperMinimumThresholdRule(**params)


if __name__ == "__main__":
    cli(build, "Build a paper minimum threshold data folder: Plunkett's personas and trials, latents drawn under the paper's design.")
