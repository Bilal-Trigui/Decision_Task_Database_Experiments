"""Minimum threshold: Plunkett's weighted sum behind a screen of per-attribute cut points (the A4 runs).

    survives(x) = all_k  xhat_k >= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

The rule itself, its draw and every table it writes live in threshold.py, the general threshold
rule. This file is that rule at its default values, unchanged, so a folder built here before the
template existed is still the folder this builds: Plunkett's weights, one active cut drawn from
every integer in 30 to 70, and the screened attribute's own weight set to 0. One change from the
A4 folder: no unscreened weight is nearer 0 than 20, so every attribute that counts matters.
min_weight_magnitude null redraws the A4 latents exactly.

Three options. A trial where both options fail the screen is answered C, "neither", so the screen
is a true negative condition the model has to report in its answers, and a fifth of every
persona's training trials are such trials. option_count 2 and neither_share null give the A4
task back: A or B only, both-fail trials decided by the weighted sum.

Target distribution, all parameters in the decision_rule block:
  active_cuts            how many of the n attributes carry a nonzero cut point (default 1)
  cut_range              [low, high] integer percentage range for an active cut (default [30, 70])
  zero_cut_main_effects  whether a screened attribute's own weight is set to 0 (default true: it
                         acts as a threshold only, never as a weight)
  min_weight_magnitude   the smallest magnitude an unscreened weight may have, on either side of 0
                         (default 20); null allows any
  option_count           3: A, B, or C when both options fail the screen (default 3)
  neither_share          the share of training and validation trials answered C (default 0.2)
The template's other dials (cut_levels, weight_range, screened_weight) are not taken here; they
belong to paper_minimum_threshold.py. Reads decision_rule, object.attribute_count.
"""
from .build import cli
from .threshold import ThresholdRule, cut_column, survives  # noqa: F401  (the estimator imports these from here)


class MinimumThresholdRule(ThresholdRule):
    name = "minimum_threshold"
    defaults = {
        "active_cuts": 1,
        "cut_range": [30, 70],
        "zero_cut_main_effects": True,
        "min_weight_magnitude": 20,
        "option_count": 3,
        "neither_share": 0.2,
        "weight_decided_share": None,
    }


def build(params):
    return MinimumThresholdRule(**params)


if __name__ == "__main__":
    cli(build, "Build a minimum threshold data folder: Plunkett's personas and trials, latents rerolled under the minimum threshold rule.")
