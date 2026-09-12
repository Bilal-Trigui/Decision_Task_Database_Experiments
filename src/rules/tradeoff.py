"""A4: Plunkett's weighted sum behind a screen of per-attribute cut points, the constrained tradeoff.

    survives(x) = all_k  xhat_k >= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

xhat is Plunkett's range-normalized value on [0, 1], so the tradeoff half is his rule verbatim.
S exceeds the whole spread of the weighted sum, which makes the score lexicographic and gives the
draft paper's form exactly: an option fails when any attribute falls below its cut point, options
that survive are ranked among themselves by the weighted sum, and a trial where both options fail
falls back to the weighted sum alone. The selection rule stays argmax.

Cut points are held as integer percentages of each attribute's own range, 0 to 100, so the latent
is one comparable scale across attributes measured in air watts and decibels, and so 0 means the
attribute carries no constraint. A cut of 40 on an attribute ranging 5 to 15 bars means the option
must reach 9 bars. The latent has 2n numbers in two blocks, 'main' and 'cut'.

Target distribution, all parameters in the decision_rule block:
  active_cuts            how many of the n attributes carry a nonzero cut point (default 1)
  cut_range              [low, high] integer percentage range for an active cut (default [30, 70])
  zero_cut_main_effects  whether a screened attribute's own weight is set to 0 (default true: it
                         acts as a constraint only, never as a tradeoff)
Reads decision_rule, object.attribute_count.
"""
import numpy as np

from src import data as D

from .base import Rule, cli
from .linear import generate_weights, normalized


def cut_column(i):
    """Latent column name for attribute i, 0-based in, 1-based out: cut1."""
    return f"cut{i + 1}"


def survives(xhat, cuts):
    """True when every constrained attribute reaches its cut point.

    A cut of 0 is tested as no constraint at all rather than as the bound xhat >= 0. Plunkett's
    option values are drawn inside the range and then rounded to his precision, which can put a
    value a little under the range minimum and make xhat negative, so the bound would reject
    options on an attribute nothing was meant to constrain.
    """
    return all(float(cuts[i]) <= 0 or xhat[i] >= float(cuts[i]) / 100.0 for i in range(len(xhat)))


class TradeoffRule(Rule):
    name = "tradeoff"
    DECISIVE_FLOOR = 0.05       # a screen deciding less than this share of trials is warned about
    defaults = {
        "active_cuts": 1,
        "cut_range": [30, 70],
        "zero_cut_main_effects": True,
    }

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        low, high = p["cut_range"]
        if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high <= 100):
            raise ValueError("decision_rule.cut_range must be [low, high] integers with 0 <= low <= high <= 100")
        if isinstance(p["active_cuts"], bool) or not isinstance(p["active_cuts"], int) or p["active_cuts"] < 1:
            raise ValueError("decision_rule.active_cuts must be a positive integer")
        if not isinstance(p["zero_cut_main_effects"], bool):
            raise ValueError("decision_rule.zero_cut_main_effects must be true or false")

    def blocks(self, n):
        return {
            "main": [f"attr{i}" for i in range(1, n + 1)],
            "cut": [cut_column(i) for i in range(n)],
        }

    def sample_latent(self, n, rng):
        p = self.params
        if p["active_cuts"] > n:
            raise ValueError(f"decision_rule.active_cuts is {p['active_cuts']} but there are only {n} attributes")
        w = generate_weights(n, rng)
        active = rng.sample(range(n), p["active_cuts"])
        cuts = [0] * n
        low, high = p["cut_range"]
        for i in active:
            cuts[i] = rng.randint(low, high)
            if p["zero_cut_main_effects"]:
                w[i] = 0
        return np.array(w + cuts, dtype=float)

    def score(self, latent, values, mins, maxs):
        n = len(values)
        xhat = normalized(values, mins, maxs)
        weights, cuts = latent[:n], latent[n:]
        tradeoff = 0
        for i in range(n):
            tradeoff += float(weights[i]) * xhat[i]
        screen = sum(abs(float(weights[i])) for i in range(n)) + 1.0
        return screen * float(survives(xhat, cuts)) + tradeoff

    def views(self, scenarios, latents, n):
        """The screen in the two readings it has: a percentage of range, and a value in the units.

        A cut point is stored as a percentage because that is the one scale comparable across
        attributes, but what it means on a trial is a level in the attribute's own units, so both
        are written and neither has to be worked out by hand.
        """
        rows = []
        for k, sc in enumerate(scenarios):
            latent, names = latents[k], sc.names[:n]
            for i in range(n):
                cut = float(latent[n + i])
                attribute = sc.attributes[i]
                low, high = float(attribute["range"][0]), float(attribute["range"][1])
                rows.append({
                    "scenario": sc.short_name,
                    "position": i + 1,
                    "attribute": names[i],
                    "units": attribute["units"],
                    "weight": float(latent[i]),
                    "cut_column": cut_column(i),
                    "cut_percent": cut,
                    "cut_value": round(low + (cut / 100.0) * (high - low), D.rounding_precision(attribute)),
                    "range_min": low,
                    "range_max": high,
                    "active": bool(cut > 0),
                })
        return {"constraint_table.csv": rows}

    def audit(self, latents, n, stats):
        notes = []
        active_counts, zeroed, levels = [], True, []
        for latent in latents:
            cuts = latent[n:]
            active = [i for i in range(n) if cuts[i] > 0]
            active_counts.append(len(active))
            for i in active:
                levels.append(float(cuts[i]))
                if latent[i] != 0:
                    zeroed = False
        declared = self.params["active_cuts"]
        if set(active_counts) != {declared}:
            notes.append(f"decision_rule.active_cuts is {declared} but personas carry {sorted(set(active_counts))} active cuts")
        if self.params["zero_cut_main_effects"] and not zeroed:
            notes.append("decision_rule.zero_cut_main_effects is true but some screened attribute has a nonzero weight")
        low, high = self.params["cut_range"]
        if levels and not all(low <= c <= high for c in levels):
            notes.append(f"decision_rule.cut_range is [{low}, {high}] but active cuts run {min(levels):g} to {max(levels):g}")
        if not levels:
            notes.append("every persona's cut block is all zero, so this folder is the linear rule under another name")
        share = (stats or {}).get("screen_decisive_share")
        if share is not None and share < self.DECISIVE_FLOOR:
            notes.append(
                f"the screen decides only {share:.1%} of trials, under the {self.DECISIVE_FLOOR:.0%} floor; "
                "cut points set very high reject both options and hand every trial back to the weighted sum"
            )
        return notes

    def summarise_draw(self, latents, n):
        cuts = latents[:, n:]
        active = (cuts > 0).sum(axis=1)
        levels = cuts[cuts > 0]
        where = {}
        for row in cuts:
            for k in np.nonzero(row > 0)[0]:
                where[f"attr{k + 1}"] = where.get(f"attr{k + 1}", 0) + 1
        mains = latents[:, :n]
        return [
            f"active cuts each       {sorted(set(active.tolist()))}",
            f"cut levels             {levels.min():.0f}% to {levels.max():.0f}%, mean {levels.mean():.1f}%"
            if levels.size else "cut levels             none, every cut is zero",
            f"|w| of nonzero mains   {np.abs(mains[mains != 0]).mean():.1f} mean",
            f"screened attributes    {', '.join(f'{k} x{c}' for k, c in sorted(where.items(), key=lambda kv: -kv[1]))}",
            "a cut near 50% bites hardest; near 100% both options fail and the screen decides nothing",
        ]

    def dataset_stats(self, train_trials, val_trials, latents, n):
        """Share of trials whose label flips when the screen is removed, and how often an option survives it."""
        shares = []
        flips = total = 0
        survived = options = 0
        for k, (train, val) in enumerate(zip(train_trials, val_trials)):
            latent = np.asarray(latents[k], float)
            tradeoff_only = latent.copy()
            tradeoff_only[n:] = 0
            persona_flips = 0
            trials = list(train) + list(val)
            for t in trials:
                if self.label(latent, t) != self.label(tradeoff_only, t):
                    persona_flips += 1
                mins, maxs = t.scenario.mins, t.scenario.maxs
                for option in (t.option_A, t.option_B):
                    survived += int(survives(normalized(option.values, mins, maxs), latent[n:]))
                    options += 1
            shares.append(persona_flips / len(trials))
            flips += persona_flips
            total += len(trials)
        return {
            "screen_decisive_share": flips / total if total else None,
            "screen_decisive_share_median_persona": float(np.median(shares)) if shares else None,
            "screen_decisive_share_min_persona": float(np.min(shares)) if shares else None,
            "option_survival_rate": survived / options if options else None,
        }


def build(params):
    return TradeoffRule(**params)


if __name__ == "__main__":
    cli(build, "Build an A4 data folder: Plunkett's personas and trials, latents rerolled under the constrained tradeoff rule.")
