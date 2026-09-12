"""A3 estimator: logistic regression on attribute differences plus centred product differences.

The design matrix has n main-effect features, Plunkett's d_i, and n(n-1)/2
interaction features, z_ij(A) - z_ij(B) with z_ij = 4 (xhat_i - 0.5)(xhat_j - 0.5),
the same feature the interaction rule scores with. The fit recovers both blocks
in one vector, rescaled jointly so the largest magnitude is 100; distance is
scored separately on the 'main' and 'interaction' halves and never combined.
Reads object.attribute_count.
"""
import numpy as np

from src.rules.interaction import centred_product, pair_column, pairs

from .base import Estimator


class InteractionEstimator(Estimator):
    name = "interaction"

    def blocks(self):
        return {
            "main": [f"attr{i}" for i in range(1, self.n + 1)],
            "interaction": [pair_column(i, j) for i, j in pairs(self.n)],
        }

    def block_aliases(self):
        return {"pair_id": "interaction"}

    def block_distance(self, block, recovered, other):
        """'pair_id' is scored as identification: did the report name the pair that actually carries the weight.

        Cosine is wrong for that question. A report naming the right pair but guessing the sign
        would score -1, below a report that names nothing. This scores 1 when the largest
        magnitude falls on the same pair and 0 otherwise, so chance is one over the pair count
        and the measured chance row says so.
        """
        if block != "pair_id":
            return super().block_distance(block, recovered, other)
        a, b = np.asarray(recovered, float), np.asarray(other, float)
        if a.shape != b.shape or a.size == 0 or not np.any(a) or not np.any(b):
            return float("nan")
        return float(int(np.argmax(np.abs(a)) == np.argmax(np.abs(b))))

    def features(self, A, B, mins, maxs):
        xa = (A - mins) / (maxs - mins)
        xb = (B - mins) / (maxs - mins)
        main = xa - xb
        inter = np.stack(
            [centred_product(xa.T, i, j) - centred_product(xb.T, i, j) for i, j in pairs(self.n)], axis=1
        )
        return np.concatenate([main, inter], axis=1)


def build(params):
    return InteractionEstimator(params["attribute_count"])
