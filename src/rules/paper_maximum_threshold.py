"""The paper's maximum threshold rule: the paper minimum threshold design with the screen reversed.

    survives(x) = all_k  xhat_k <= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

An option ABOVE the cut point on the screened attribute loses outright, and options that pass are
ranked by the weighted sum. Read as an if statement:

    if A and B are both above the cut     C
    elif only one is above it             the other
    else                                  the weighted sum picks

A value exactly at the cut passes, as it does under the minimum rule.

Everything else is paper_minimum_threshold.py's: the draw (Plunkett's weights, none nearer 0 than 20,
the screened attribute's weight exactly 0, one cut at 25, 50 or 75), the answers A, B and C, the
quotas and the latent's shape (2n numbers, blocks 'main' and 'cut'). A cut of 0 still means "no
constraint", though a real maximum of 0 would reject nearly everything: active cuts are never 0, so
the marker stays unambiguous, and every "cut > 0" reading in the pipeline holds under both rules.

Reads decision_rule, object.attribute_count.
"""
from .build import cli
from .paper_minimum_threshold import PaperMinimumThresholdRule


def survives(xhat, cuts):
    """True when no constrained attribute goes above its cut point. A cut of 0 is no constraint."""
    return all(float(cuts[i]) <= 0 or xhat[i] <= float(cuts[i]) / 100.0 for i in range(len(xhat)))


class PaperMaximumThresholdRule(PaperMinimumThresholdRule):
    name = "paper_maximum_threshold"
    survives = staticmethod(survives)
    DECISIVE_HINT = "cut points set very low reject both options and hand every trial back to the weighted sum"
    DRAW_NOTE = "a cut near 50% bites hardest; near 0% both options fail and the screen decides nothing"


def build(params):
    return PaperMaximumThresholdRule(**params)


if __name__ == "__main__":
    cli(build, "Build a paper maximum threshold data folder: Plunkett's personas and trials, latents drawn under the paper's design.")
