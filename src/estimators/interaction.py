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
