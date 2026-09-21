"""Base class for estimators: the decision rule run backwards.

An estimator turns a persona's A/B choices into a recovered latent, the
predicted learned latent of the paper. It declares which blocks it recovers,
builds a design matrix from the two options of each trial, fits it, and
measures distance between two latents of one block.

Reads: object.attribute_count (through `attribute_count`); model_estimating
(type only; the behaviour-collection settings are used by evaluate.py).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_logistic(X, y, sample_weight=None, C=1.0, with_intercept=False):
    """L2 logistic regression with an intercept.

    C = 1.0 makes this the maximum a posteriori fit under Plunkett's N(0, 1)
    prior on every slope. A persona whose choices are all A or all B has no
    identifiable direction and returns zeros.
    """
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        zeros = np.zeros(X.shape[1])
        return (zeros, 0.0) if with_intercept else zeros
    clf = LogisticRegression(C=C, fit_intercept=True, max_iter=1000)
    clf.fit(X, y, sample_weight=sample_weight)
    coef = clf.coef_[0].astype(float)
    return (coef, float(clf.intercept_[0])) if with_intercept else coef


def rescale_to_100(coef):
    """Plunkett's normalisation: divide by the largest magnitude so it becomes 100. No rounding."""
    coef = np.asarray(coef, dtype=float)
    largest = np.max(np.abs(coef)) if coef.size else 0.0
    return coef * (100.0 / largest) if largest > 0 else coef


class Estimator:
    name = "base"

    def __init__(self, attribute_count):
        self.n = attribute_count

    def blocks(self):
        """Ordered dict of block name -> recovered column names."""
        raise NotImplementedError

    def latent_columns(self):
        return [c for cols in self.blocks().values() for c in cols]

    def block_aliases(self):
        """Extra names for blocks this estimator already recovers, as alias -> real block. See Rule.block_aliases."""
        return {}

    def block_slices(self):
        slices, start = {}, 0
        for block, cols in self.blocks().items():
            slices[block] = slice(start, start + len(cols))
            start += len(cols)
        for alias, block in self.block_aliases().items():
            slices[alias] = slices[block]
        return slices

    def features(self, A, B, mins, maxs):
        """Design matrix, one row per trial, from option values A and B of shape (trials, n)."""
        raise NotImplementedError

    def fit(self, A, B, choices, mins, maxs, sample_weight=None):
        """Recovered latent for one persona from its choices ('A'/'B' per trial), rescaled so max |.| = 100."""
        X = self.features(np.asarray(A, float), np.asarray(B, float), np.asarray(mins, float), np.asarray(maxs, float))
        y = np.array([1 if c == "A" else 0 for c in choices])
        return rescale_to_100(fit_logistic(X, y, sample_weight=sample_weight))

    def block_is_absolute(self, block):
        """True when a block holds levels rather than a direction, so its scale is part of the object.

        A weight vector is a direction: doubling it means the same preference, so it is normalised
        before the agreement bands are measured. A4's cut points are levels: doubling them means a
        different screen, so they are compared as they stand.
        """
        return False

    def block_distance_name(self, block):
        """What `block_distance` measures for this block, recorded in every results row.

        Rows carrying different measures land in one `faithfulness` column, so without this the
        column cannot be read or averaged safely: cosine on most blocks, a scaled error on A4's
        cut points, an identification score on the interaction schema's pair question.
        """
        return "cosine"

    def block_distance(self, block, recovered, other):
        """Distance for one named block, so an estimator whose blocks are not all directions can say so.

        Cosine everywhere by default, which is right for any block that is a direction. A block
        holding absolute levels rather than a direction overrides this, and then its chance level
        moves too, which is why chance is measured per block and never assumed.
        """
        return self.distance(recovered, other)

    @staticmethod
    def distance(recovered, other):
        """Cosine similarity between two latents of one block (higher is closer). NaN if either has no direction."""
        a, b = np.asarray(recovered, float), np.asarray(other, float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0 or a.shape != b.shape:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))


TOLERANCES = (10, 30)
AGREEMENT_COLUMNS = ("exact_match", "within_10", "within_30", "sign_agreement", "top_match", "active_error")


def _blank():
    return {c: float("nan") for c in AGREEMENT_COLUMNS}


def agreement(pairs, absolute=False):
    """Six ways of asking how close a report got, where cosine only asks whether the shape is right.

    Each answers a different question, and on a sparse block they can disagree sharply:

      exact_match     the same integer after rescaling
      within_10       within ten points on the -100 to 100 scale, a twentieth of the range
      within_30       within thirty, about a sixth of the range
      sign_agreement  merely the right side of zero, the measure that survives when a value is
                      near zero and neither the choices nor the report carry much about it
      top_match       the largest magnitude falls on the same component: did it find the right
                      attribute at all, before any question of how much
      active_error    mean gap in points over only the components the recovered latent actually
                      uses, so a mostly zero block cannot score well on zeros it got for free

    Whether to rescale first depends on what the block holds, and getting it wrong destroys the
    measurement either way. A weight vector is a direction, its scale is not part of the object,
    and reports come off the model at a median peak of 87 against a recovered vector always
    peaked at exactly 100, so without Plunkett's rescaling the bands would count that gap as
    disagreement. A cut point is an absolute level, and rescaling a sparse cut vector maps every
    single-screen report onto the same normalised vector, so a cut reported at 30 and one at 60
    would both score a perfect match. Pass `absolute` for those blocks; the estimator says which
    they are.

    Read each against its chance column. On a sparse block a wide band is high for free.
    """
    if not pairs:
        return _blank()
    norm = (lambda v: np.asarray(v, float).ravel()) if absolute else (lambda v: rescale_to_100(np.asarray(v, float).ravel()))
    rows = [(norm(x), norm(y)) for x, y in pairs]
    rows = [(x, y) for x, y in rows if x.shape == y.shape and x.size]
    if not rows:
        return _blank()
    a = np.concatenate([x for x, _ in rows])
    b = np.concatenate([y for _, y in rows])
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if not a.size:
        return _blank()
    gap = np.abs(a - b)
    out = {"exact_match": float(np.mean(gap < 0.5)),
           "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}
    for t in TOLERANCES:
        out[f"within_{t}"] = float(np.mean(gap <= t))
    # did the report put its biggest number where the recovered latent puts its biggest number
    tops = [int(np.any(x) and np.argmax(np.abs(x)) == np.argmax(np.abs(y))) for x, y in rows if np.any(y)]
    out["top_match"] = float(np.mean(tops)) if tops else float("nan")
    # and how far off is it where the latent is actually doing something
    live = [np.abs(x - y)[np.asarray(y) != 0] for x, y in rows]
    live = np.concatenate([v for v in live if v.size]) if any(v.size for v in live) else np.array([])
    out["active_error"] = float(np.mean(live)) if live.size else float("nan")
    return out


def pooled_pearson(pairs):
    """Plunkett's headline distance: Pearson r over all personas' values of one block pooled together.

    `pairs` is a list of (vector, vector) per persona. Standardising each side
    first, as Plunkett did, does not change r. NaN with fewer than two values
    or no variance on either side.
    """
    if not pairs:
        return float("nan")
    a = np.concatenate([np.asarray(p[0], float).ravel() for p in pairs])
    b = np.concatenate([np.asarray(p[1], float).ravel() for p in pairs])
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
