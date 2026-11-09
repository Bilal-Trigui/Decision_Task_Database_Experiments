"""Paper maximum threshold estimator: the paper minimum threshold fit, with the screen reversed.

The rule read back as an if statement (both options above the cut: C; one above: the other; neither:
the weighted sum), every attribute tried at 25, 50 and 75, the weights refitted on the trials they
decide, and the candidate whose branches best explain the choices kept. The search, the likelihood,
the lapse rates and the screen questions' distances are paper_minimum_threshold.py's, unchanged. The
only change is which side of the cut passes.

Choices without C. Every cut predicts C wherever both options are above it, so a model that never
answers C is best explained by the cut predicting C least. Here that is the HIGHEST level, 75, where
under the minimum rule it is the lowest, 25. Read the cut and screen_cut rows beside chosen_C and
correct_C, as there.

Reads object.attribute_count, model_estimating (grid_step).
"""
import numpy as np

from .minimum_threshold import check_removed
from .paper_minimum_threshold import PaperMinimumThresholdEstimator


class PaperMaximumThresholdEstimator(PaperMinimumThresholdEstimator):
    name = "paper_maximum_threshold"

    def _passes(self, x, cuts):
        """The rule's survives(), for every trial at once: above the cut fails, and a cut of 0 is no constraint."""
        cuts = np.asarray(cuts, float)
        return np.all((cuts <= 0) | (x <= cuts / 100.0), axis=1)


def build(params):
    check_removed(params)
    return PaperMaximumThresholdEstimator(params["attribute_count"], grid_step=params.get("grid_step"),
                                          option_count=params.get("option_count", 2))
