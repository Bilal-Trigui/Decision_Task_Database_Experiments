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

    def block_aliases(self, n):
        # 'pair_id' asks which pair interacts rather than for all n(n-1)/2 values, so it reports
        # on the interaction block through a different question. See src/reports/interaction.py.
        return {"pair_id": "interaction"}

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

    def views(self, scenarios, latents, n):
        """The interaction table twice: an n by n block per persona, and one pair per row.

        The matrix carries main effects on the diagonal and pair weights off it, mirrored across
        it because it reads better that way. The rule scores each pair once over the strict upper
        triangle, so a value at (i, j) and (j, i) is one coefficient shown twice and not two. The
        long form is the same numbers in a shape you can join against results.
        """
        all_pairs = pairs(n)
        column_of = {p: n + k for k, p in enumerate(all_pairs)}
        matrix, longform = [], []
        for k, sc in enumerate(scenarios):
            names, latent = sc.names[:n], latents[k]
            for i in range(n):
                row = {"scenario": sc.short_name, "position": i + 1, "attribute": names[i]}
                for j in range(n):
                    cell = latent[i] if i == j else latent[column_of[(min(i, j), max(i, j))]]
                    row[f"attr{j + 1}"] = float(cell)
                matrix.append(row)
            for i, j in all_pairs:
                v = float(latent[column_of[(i, j)]])
                longform.append({
                    "scenario": sc.short_name,
                    "i": i + 1,
                    "j": j + 1,
                    "attribute_i": names[i],
                    "attribute_j": names[j],
                    "pair_column": pair_column(i, j),
                    "v": v,
                    "main_i": float(latent[i]),
                    "main_j": float(latent[j]),
                    "active": bool(v != 0),
                })
        return {"interaction_matrix.csv": matrix, "interaction_pairs.csv": longform}

    def audit(self, latents, n, stats):
        notes = []
        all_pairs = pairs(n)
        active_counts, zeroed, magnitudes = [], True, []
        for latent in latents:
            v = latent[n:]
            active = [k for k in range(len(all_pairs)) if v[k] != 0]
            active_counts.append(len(active))
            for k in active:
                magnitudes.append(abs(float(v[k])))
                i, j = all_pairs[k]
                if latent[i] != 0 or latent[j] != 0:
                    zeroed = False
        declared = self.params["active_pairs"]
        if set(active_counts) != {declared}:
            notes.append(f"decision_rule.active_pairs is {declared} but personas carry {sorted(set(active_counts))} active pairs")
        if self.params["zero_pair_main_effects"] and not zeroed:
            notes.append("decision_rule.zero_pair_main_effects is true but some active pair has a nonzero main weight")
        low, high = self.params["interaction_magnitude"]
        if magnitudes and not all(low <= m <= high for m in magnitudes):
            notes.append(
                f"decision_rule.interaction_magnitude is [{low}, {high}] but |v| runs "
                f"{min(magnitudes):g} to {max(magnitudes):g}"
            )
        if not magnitudes:
            notes.append("every persona's interaction block is all zero, so this folder tests nothing the linear rule does not")
        return notes

    def summarise_draw(self, latents, n):
        all_pairs = pairs(n)
        v = latents[:, n:]
        active = (v != 0).sum(axis=1)
        magnitudes = np.abs(v[v != 0])
        counts = {}
        for row in v:
            for k in np.nonzero(row)[0]:
                key = f"attr{all_pairs[k][0] + 1} x attr{all_pairs[k][1] + 1}"
                counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        mains = latents[:, :n]
        return [
            f"active pairs each      {sorted(set(active.tolist()))}",
            f"|v| of active pairs    {magnitudes.min():.0f} to {magnitudes.max():.0f}, mean {magnitudes.mean():.1f}"
            if magnitudes.size else "|v| of active pairs    none, every interaction is zero",
            f"|w| of nonzero mains   {np.abs(mains[mains != 0]).mean():.1f} mean",
            f"most drawn pairs       {', '.join(f'{k} x{c}' for k, c in top)}",
        ]

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
