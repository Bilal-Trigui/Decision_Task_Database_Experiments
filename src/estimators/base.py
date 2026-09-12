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
