"""Plunkett's estimator: logistic regression of the choice on normalized attribute differences.

selection ~ d_1 + ... + d_n with d_i = (a_i - b_i) / (max_i - min_i), N(0, 1)
priors (C = 1.0), coefficients rescaled so the largest magnitude is 100.
Recovers one block, 'main'. Reads object.attribute_count.
"""
from .base import Estimator


class LinearEstimator(Estimator):
    name = "linear"

    def blocks(self):
        return {"main": [f"attr{i}" for i in range(1, self.n + 1)]}

    def features(self, A, B, mins, maxs):
        return (A - B) / (maxs - mins)


def build(params):
    return LinearEstimator(params["attribute_count"])
