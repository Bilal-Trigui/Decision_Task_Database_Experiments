"""Plunkett's decision rule: a weighted sum over range-normalized attributes.

U(x) = sum_k w_k * (x_k - min_k) / (max_k - min_k); the higher utility wins.
The latent is one block, 'main', of `attribute_count` weights drawn as in
Plunkett's generate_weights: Uniform(-100, 100), rescaled so the largest
magnitude is exactly 100, rounded to integers. Reads decision_rule (no
parameters), object.attribute_count.
"""
import numpy as np

from .base import Rule, cli


def generate_weights(n, rng):
    """Plunkett's generate_weights, verbatim, drawing from `rng`."""
    raw_weights = [rng.uniform(-100, 100) for _ in range(n)]
    max_abs_idx = max(range(len(raw_weights)), key=lambda i: abs(raw_weights[i]))
    max_signed = raw_weights[max_abs_idx]
    max_sign = np.sign(max_signed)
    scaling_factor = (100 * max_sign) / max_signed
    return [round(p * scaling_factor) for p in raw_weights]


def normalized(values, mins, maxs):
    """Plunkett's scaled_value, one attribute at a time."""
    return [(float(values[i]) - float(mins[i])) / (float(maxs[i]) - float(mins[i])) for i in range(len(values))]


class LinearRule(Rule):
    name = "linear"
    defaults = {}

    def blocks(self, n):
        return {"main": [f"attr{i}" for i in range(1, n + 1)]}

    def sample_latent(self, n, rng):
        return np.array(generate_weights(n, rng), dtype=float)

    def score(self, latent, values, mins, maxs):
        # Plunkett's calculate_utility: a sequential sum, kept in his order so labels match bit for bit.
        utility = 0
        for i, scaled_value in enumerate(normalized(values, mins, maxs)):
            utility += float(latent[i]) * scaled_value
        return utility


def build(params):
    return LinearRule(**params)


if __name__ == "__main__":
    cli(build, "Rebuild a data folder under Plunkett's linear rule (his seeds reproduce his files).")
