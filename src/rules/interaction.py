"""A3: Plunkett's weighted sum plus pairwise interaction terms over attribute pairs (strict upper triangle).

    U(x) = sum_k w_k * xhat_k  +  sum_{i<j} v_ij * z_ij,    z_ij = 4 * (xhat_i - 0.5) * (xhat_j - 0.5)

xhat is Plunkett's range-normalized value on [0, 1], so the main-effect half is
his rule verbatim. z_ij is the product of the centered attributes scaled to
[-1, 1]: centering keeps the product nearly uncorrelated with the main effects,
which is what lets the estimator separate the two blocks, and the scaling puts
v_ij on the same -100..100 scale as w. The diagonal (squared terms) is excluded:
the latent has n + n(n-1)/2 numbers in two blocks, 'main' and 'interaction'.

Target distribution, all parameters in the decision_rule block:
  active_pairs           how many of the n(n-1)/2 pairs carry a nonzero v (default 1; n(n-1)/2 gives Doc-26's dense draw)
  zero_pair_main_effects whether an active pair's own main weights are set to 0 (default true: they matter only jointly)
  main_scale             multiplier on Plunkett's main weights before rounding (default 0.5)
  interaction_magnitude  [low, high] integer range for |v|, sign random (default [50, 100])
Reads decision_rule, object.attribute_count.
"""
import itertools

import numpy as np

from .base import Rule, cli
from .linear import generate_weights, normalized

PAIR_KEY_SEPARATOR = " × "


def pairs(n):
    """Attribute index pairs (i, j) with i < j, in itertools.combinations order."""
    return list(itertools.combinations(range(n), 2))


def pair_column(i, j):
    """Latent column name for pair (i, j), 0-based in, 1-based out: v_1_2."""
    return f"v_{i + 1}_{j + 1}"


def pair_key(names, i, j):
    """Report JSON key for pair (i, j): 'first_name × second_name'."""
    return f"{names[i]}{PAIR_KEY_SEPARATOR}{names[j]}"


def centred_product(xhat, i, j):
    return 4.0 * (xhat[i] - 0.5) * (xhat[j] - 0.5)


class InteractionRule(Rule):
    name = "interaction"
    defaults = {
        "active_pairs": 1,
        "zero_pair_main_effects": True,
        "main_scale": 0.5,
        "interaction_magnitude": [50, 100],
    }

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        lo, hi = p["interaction_magnitude"]
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= 100):
            raise ValueError("decision_rule.interaction_magnitude must be [low, high] integers with 1 <= low <= high <= 100")
        if not 0 < p["main_scale"] <= 1:
            raise ValueError("decision_rule.main_scale must be in (0, 1]")
        if isinstance(p["active_pairs"], bool) or not isinstance(p["active_pairs"], int) or p["active_pairs"] < 1:
            raise ValueError("decision_rule.active_pairs must be a positive integer")

    def blocks(self, n):
        return {
            "main": [f"attr{i}" for i in range(1, n + 1)],
            "interaction": [pair_column(i, j) for i, j in pairs(n)],
        }

    def sample_latent(self, n, rng):
        p = self.params
        all_pairs = pairs(n)
        if p["active_pairs"] > len(all_pairs):
            raise ValueError(f"decision_rule.active_pairs is {p['active_pairs']} but {n} attributes have only {len(all_pairs)} pairs")
        w = [round(x * p["main_scale"]) for x in generate_weights(n, rng)]
        active = rng.sample(range(len(all_pairs)), p["active_pairs"])
        v = [0] * len(all_pairs)
        lo, hi = p["interaction_magnitude"]
        for idx in active:
            i, j = all_pairs[idx]
            if p["zero_pair_main_effects"]:
                w[i], w[j] = 0, 0
            v[idx] = rng.choice([-1, 1]) * rng.randint(lo, hi)
        return np.array(w + v, dtype=float)

    def score(self, latent, values, mins, maxs):
        n = len(values)
        xhat = normalized(values, mins, maxs)
        utility = 0
        for i in range(n):
            utility += float(latent[i]) * xhat[i]
        for p, (i, j) in enumerate(pairs(n)):
            v = float(latent[n + p])
            if v != 0:
                utility += v * centred_product(xhat, i, j)
        return utility

    def dataset_stats(self, train_trials, val_trials, latents, n):
        """Share of trials whose label flips when the interaction block is zeroed, overall and per persona."""
        shares = []
        flips = total = 0
        for k, (train, val) in enumerate(zip(train_trials, val_trials)):
            latent = np.asarray(latents[k], float)
            main_only = latent.copy()
            main_only[n:] = 0
            persona_flips = 0
            trials = list(train) + list(val)
            for t in trials:
                if self.label(latent, t) != self.label(main_only, t):
                    persona_flips += 1
            shares.append(persona_flips / len(trials))
            flips += persona_flips
            total += len(trials)
        return {
            "interaction_decisive_share": flips / total if total else None,
            "interaction_decisive_share_median_persona": float(np.median(shares)) if shares else None,
            "interaction_decisive_share_min_persona": float(np.min(shares)) if shares else None,
        }


def build(params):
    return InteractionRule(**params)


if __name__ == "__main__":
    cli(build, "Build an A3 data folder: Plunkett's personas and trials, latents rerolled under the interaction rule.")
