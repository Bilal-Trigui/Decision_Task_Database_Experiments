"""The general threshold rule: Plunkett's weighted sum behind a screen of per-attribute cut points.

    survives(x) = all_k  xhat_k >= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

xhat is Plunkett's range-normalized value on [0, 1], so the weighted-sum half is his rule verbatim.
S exceeds the whole spread of the weighted sum, which makes the score lexicographic and gives the
draft paper's form exactly: an option fails when any attribute falls below its cut point, and
options that survive are ranked among themselves by the weighted sum. The selection rule stays argmax.

A trial where both options fail is where option_count matters:
  2   A or B, Plunkett's task: the trial falls back to the weighted sum alone
  3   A, B or C: both options are rejected, so the answer is C, "neither". The screen is then a
      true negative condition, visible in the answers themselves, and not only in which option
      wins. Read like an if statement:
          if A and B both fail the screen    C
          elif only one fails                the other
          else                               the weighted sum picks

Cut points are held as integer percentages of each attribute's own range, 0 to 100, so the latent
is one comparable scale across attributes measured in air watts and decibels, and so 0 means the
attribute carries no constraint. A cut of 40 on an attribute ranging 5 to 15 bars means the option
must reach 9 bars. The latent has 2n numbers in two blocks, 'main' and 'cut'.

THIS FILE IS THE TEMPLATE, NOT AN EXPERIMENT
It holds every piece of the rule and every dial, at the minimum threshold rule's values, and it
cannot be run: build() refuses, so no settings file can name it and no data folder is built from
it. An experiment is a subclass that sets its own values:
  minimum_threshold.py         the values here, unchanged (the A4 runs)
  paper_minimum_threshold.py   the paper's design: Plunkett's weights, screened weight 0, cuts 25, 50 or 75

The dials, all in the decision_rule block:
  active_cuts            how many of the n attributes carry a nonzero cut point (default 1)
  cut_range              [low, high] integer percentages an active cut is drawn from, every integer
                         in between equally likely: the continuous draw (default [30, 70])
  cut_levels             a list of percentages to draw an active cut from instead, each equally
                         likely: the discrete draw. null keeps the continuous one (default null).
                         Every level must lie inside cut_range
  weight_range           [low, high]: each weight drawn uniformly from it, to two decimals. null
                         keeps Plunkett's draw, Uniform(-100, 100) rescaled so the largest
                         magnitude is 100 (default null)
  min_weight_magnitude   the smallest magnitude an unscreened weight may have: the weights are
                         drawn again until every one that still counts is at least this far from
                         0, on either side. null allows any magnitude in the range (default null)
  screened_weight        the weight a screened attribute is given, whatever was drawn for it. null
                         keeps the draw (default null)
  zero_cut_main_effects  set a screened attribute's weight to 0, so it acts as a threshold only and
                         never as a weight (default true). Cannot be combined with screened_weight
  option_count           2 (A or B) or 3 (A, B, or C when both options fail the screen). Default 2
  neither_share          with option_count 3, the share of each persona's training and validation
                         trials whose answer is C, drawn to that count exactly, so every persona
                         sees the same number of "neither" trials whatever its cut. null takes the
                         trials as they come, where the share runs from a few percent (a low cut)
                         to over half (a high one). Default null. The estimation trials are always
                         taken as they come. With option_count 2 a rule's default share is dropped to
                         null; setting both 2 options and a share in the settings is refused
  weight_decided_share   the share of each persona's training and validation trials in which both
                         options pass the screen, so the weighted sum alone decides A or B, drawn
                         to that count exactly. These are the only trials that teach the weights:
                         taken as they come they are about 29% under the paper draw, and as few as
                         6% for a persona screened at 75. The rest, beyond neither_share's C trials,
                         are trials one option fails, decided by the screen. null takes them as they
                         come (default). neither_share plus this may not exceed 1. The rule itself is
                         unchanged; only which trials are drawn for training is. The estimation
                         trials are always taken as they come
Reads decision_rule, object.attribute_count.
"""
import numpy as np

from src import data as D

from .base import MAX_REDRAWS, Rule, check_min_weight_magnitude, clears_floor, floor_unreachable
from .linear import generate_weights, normalized

NUMBER = (int, float)


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


def _is_number(value):
    return isinstance(value, NUMBER) and not isinstance(value, bool)


class ThresholdRule(Rule):
    name = "threshold"
    DECISIVE_FLOOR = 0.05       # a screen deciding less than this share of trials is warned about
    option_counts = (2, 3)      # a trial both options fail can be answered C
    defaults = {
        "active_cuts": 1,
        "cut_range": [30, 70],
        "cut_levels": None,
        "weight_range": None,
        "min_weight_magnitude": None,
        "screened_weight": None,
        "zero_cut_main_effects": True,
        "option_count": 2,
        "neither_share": None,
        "weight_decided_share": None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        if p["option_count"] == 2 and "neither_share" not in params:
            p["neither_share"] = None    # a rule's default share of C answers means nothing without a C
        low, high = p["cut_range"]
        if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high <= 100):
            raise ValueError("decision_rule.cut_range must be [low, high] integers with 0 <= low <= high <= 100")
        if isinstance(p["active_cuts"], bool) or not isinstance(p["active_cuts"], int) or p["active_cuts"] < 1:
            raise ValueError("decision_rule.active_cuts must be a positive integer")
        if not isinstance(p["zero_cut_main_effects"], bool):
            raise ValueError("decision_rule.zero_cut_main_effects must be true or false")
        levels = p.get("cut_levels")
        if levels is not None:
            if (not isinstance(levels, list) or not levels
                    or any(isinstance(c, bool) or not isinstance(c, int) for c in levels)):
                raise ValueError("decision_rule.cut_levels must be a non-empty list of integer percentages, or null")
            if len(set(levels)) != len(levels):
                raise ValueError(f"decision_rule.cut_levels lists a level twice: {levels}")
            outside = [c for c in levels if not (max(low, 1) <= c <= high)]
            if outside:
                raise ValueError(f"decision_rule.cut_levels {outside} lie outside cut_range [{low}, {high}] or at 0, which is no cut")
        weights = p.get("weight_range")
        if weights is not None:
            if not (isinstance(weights, list) and len(weights) == 2 and all(_is_number(v) for v in weights)
                    and weights[0] <= weights[1]):
                raise ValueError("decision_rule.weight_range must be [low, high] numbers with low <= high, or null")
        floor = p.get("min_weight_magnitude")
        check_min_weight_magnitude(floor)
        if floor is not None and weights is not None and max(abs(weights[0]), abs(weights[1])) < floor:
            raise ValueError(f"decision_rule.min_weight_magnitude {floor} lies outside weight_range {weights}")
        share = p.get("neither_share")
        if share is not None:
            if isinstance(share, bool) or not _is_number(share) or not 0 < share < 1:
                raise ValueError("decision_rule.neither_share must be a number between 0 and 1, or null")
            if p["option_count"] != 3:
                raise ValueError(
                    f"decision_rule.neither_share is {share} but option_count is {p['option_count']}; "
                    "a share of C answers needs option_count 3"
                )
        weights_share = p.get("weight_decided_share")
        if weights_share is not None:
            if isinstance(weights_share, bool) or not _is_number(weights_share) or not 0 < weights_share < 1:
                raise ValueError("decision_rule.weight_decided_share must be a number between 0 and 1, or null")
            if weights_share + (share or 0) > 1:
                raise ValueError(f"decision_rule.weight_decided_share {weights_share} and neither_share {share} "
                                 "add up to more than every trial")
        screened = p.get("screened_weight")
        if screened is not None:
            if not _is_number(screened):
                raise ValueError("decision_rule.screened_weight must be a number, or null")
            if p["zero_cut_main_effects"]:
                raise ValueError(
                    "decision_rule.screened_weight is set but zero_cut_main_effects is true, which would set "
                    "that same weight to 0; set zero_cut_main_effects to false"
                )

    def blocks(self, n):
        return {
            "main": [f"attr{i}" for i in range(1, n + 1)],
            "cut": [cut_column(i) for i in range(n)],
        }

    # -- the draw --

    def weight_range(self, n, rng):
        """The weight block as drawn, before any screen touches it.

        With weight_range set, each weight is uniform on [low, high], to two decimals. Without it,
        Plunkett's draw: Uniform(-100, 100), rescaled so the largest magnitude is exactly 100.
        """
        bounds = self.params.get("weight_range")
        if bounds is None:
            return generate_weights(n, rng)
        low, high = bounds
        return [round(rng.uniform(low, high), 2) for _ in range(n)]

    def cut_level(self, rng):
        """One active cut point: any integer in cut_range (continuous), or one of cut_levels (discrete)."""
        levels = self.params.get("cut_levels")
        if levels is None:
            low, high = self.params["cut_range"]
            return rng.randint(low, high)
        return rng.choice(levels)

    def sample_latent(self, n, rng):
        p = self.params
        if p["active_cuts"] > n:
            raise ValueError(f"decision_rule.active_cuts is {p['active_cuts']} but there are only {n} attributes")
        floor = p.get("min_weight_magnitude")
        for _ in range(MAX_REDRAWS):   # with no minimum, one draw, in the order it always had
            w = self.weight_range(n, rng)
            active = rng.sample(range(n), p["active_cuts"])
            # the minimum is on the weights that still count: a screened one is replaced below
            if clears_floor(w, floor, skip=active):
                break
        else:
            raise floor_unreachable(floor)
        cuts = [0] * n
        for i in active:
            cuts[i] = self.cut_level(rng)
            if p["zero_cut_main_effects"]:
                w[i] = 0
            elif p.get("screened_weight") is not None:
                w[i] = p["screened_weight"]
        return np.array(w + cuts, dtype=float)

    # -- the rule --

    def passes(self, latent, values, mins, maxs):
        """True when an option clears every cut point of the latent: the screen half of the rule."""
        n = len(values)
        return survives(normalized(values, mins, maxs), latent[n:])

    def label(self, latent, trial):
        """C when both options fail the screen and the trial offers C; otherwise the shared A or B choice."""
        if self.option_count == 3:
            mins, maxs = trial.scenario.mins, trial.scenario.maxs
            if not self.passes(latent, trial.option_A.values, mins, maxs) and not self.passes(latent, trial.option_B.values, mins, maxs):
                return "C"
        return super().label(latent, trial)

    def label_quota(self, n_trials):
        quota = {}
        if self.params.get("neither_share") is not None:
            quota["C"] = int(round(self.params["neither_share"] * n_trials))
        if self.params.get("weight_decided_share") is not None:
            quota["weights"] = int(round(self.params["weight_decided_share"] * n_trials))
        return quota or None

    def quota_key(self, latent, trial):
        """C for a "neither" trial; "weights" when both options pass the screen and weight_decided_share is set."""
        if trial.label == "C" or self.params.get("weight_decided_share") is None:
            return trial.label
        mins, maxs = trial.scenario.mins, trial.scenario.maxs
        if self.passes(latent, trial.option_A.values, mins, maxs) and self.passes(latent, trial.option_B.values, mins, maxs):
            return "weights"
        return trial.label

    def score(self, latent, values, mins, maxs):
        n = len(values)
        xhat = normalized(values, mins, maxs)
        weights, cuts = latent[:n], latent[n:]
        weighted_sum = 0
        for i in range(n):
            weighted_sum += float(weights[i]) * xhat[i]
        screen = sum(abs(float(weights[i])) for i in range(n)) + 1.0
        return screen * float(survives(xhat, cuts)) + weighted_sum

    # -- describing a folder --

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
        p = self.params
        notes = []
        active_counts, levels, screened, unscreened = [], [], [], []
        for latent in latents:
            cuts = latent[n:]
            active = [i for i in range(n) if cuts[i] > 0]
            active_counts.append(len(active))
            for i in range(n):
                (screened if i in active else unscreened).append(float(latent[i]))
            levels += [float(cuts[i]) for i in active]
        declared = p["active_cuts"]
        if set(active_counts) != {declared}:
            notes.append(f"decision_rule.active_cuts is {declared} but personas carry {sorted(set(active_counts))} active cuts")
        if p["zero_cut_main_effects"] and any(w != 0 for w in screened):
            notes.append("decision_rule.zero_cut_main_effects is true but some screened attribute has a nonzero weight")
        if p.get("screened_weight") is not None and any(w != p["screened_weight"] for w in screened):
            notes.append(f"decision_rule.screened_weight is {p['screened_weight']} but some screened attribute has another weight")
        bounds = p.get("weight_range")
        if bounds is not None and unscreened and not all(bounds[0] <= w <= bounds[1] for w in unscreened):
            notes.append(
                f"decision_rule.weight_range is {bounds} but unscreened weights run "
                f"{min(unscreened):g} to {max(unscreened):g}"
            )
        low, high = p["cut_range"]
        if levels and not all(low <= c <= high for c in levels):
            notes.append(f"decision_rule.cut_range is [{low}, {high}] but active cuts run {min(levels):g} to {max(levels):g}")
        if p.get("cut_levels") is not None and levels and not set(levels) <= set(map(float, p["cut_levels"])):
            notes.append(f"decision_rule.cut_levels is {p['cut_levels']} but active cuts take {sorted(set(levels))}")
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
        counts = {float(c): int((levels == c).sum()) for c in np.unique(levels)}
        return [
            f"active cuts each       {sorted(set(active.tolist()))}",
            f"cut levels             {levels.min():.0f}% to {levels.max():.0f}%, mean {levels.mean():.1f}%"
            if levels.size else "cut levels             none, every cut is zero",
            f"cut level counts       {', '.join(f'{c:g}% x{k}' for c, k in counts.items())}"
            if levels.size and len(counts) <= 10 else "cut level counts       more than ten distinct levels",
            f"weights                {mains.min():g} to {mains.max():g}",
            f"|w| of nonzero mains   {np.abs(mains[mains != 0]).mean():.2f} mean",
            f"screened attributes    {', '.join(f'{k} x{c}' for k, c in sorted(where.items(), key=lambda kv: -kv[1]))}",
            "a cut near 50% bites hardest; near 100% both options fail and the screen decides nothing",
        ]

    def dataset_stats(self, train_trials, val_trials, latents, n):
        """Share of trials whose label flips when the screen is removed, how often an option survives it, and the answers.

        A trial answered C is one the screen decides: with the screen removed it would be A or B.
        """
        shares = []
        answers = {letter: 0 for letter in self.letters}
        flips = total = 0
        survived = options = both_pass = 0
        for k, (train, val) in enumerate(zip(train_trials, val_trials)):
            latent = np.asarray(latents[k], float)
            weights_only = latent.copy()
            weights_only[n:] = 0
            persona_flips = 0
            trials = list(train) + list(val)
            for t in trials:
                answers[t.label or self.label(latent, t)] += 1
                if self.label(latent, t) != self.label(weights_only, t):
                    persona_flips += 1
                mins, maxs = t.scenario.mins, t.scenario.maxs
                passing = [survives(normalized(option.values, mins, maxs), latent[n:]) for option in (t.option_A, t.option_B)]
                survived += sum(passing)
                options += 2
                both_pass += int(all(passing))
            shares.append(persona_flips / len(trials))
            flips += persona_flips
            total += len(trials)
        return {
            "screen_decisive_share": flips / total if total else None,
            "screen_decisive_share_median_persona": float(np.median(shares)) if shares else None,
            "screen_decisive_share_min_persona": float(np.min(shares)) if shares else None,
            "option_survival_rate": survived / options if options else None,
            "weight_decided_share": both_pass / total if total else None,
            "answer_shares": {letter: count / total for letter, count in answers.items()} if total else None,
        }


TEMPLATE_ONLY = (
    "src/rules/threshold.py is the general threshold rule, a template holding the default values, and is "
    "not run directly. Name an experiment built on it instead: decision_rule.type 'paper_minimum_threshold' "
    "(the paper's design) or 'minimum_threshold' (the A4 values)."
)


def build(params):
    from src.config import SettingsError

    raise SettingsError(TEMPLATE_ONLY)


if __name__ == "__main__":
    raise SystemExit(TEMPLATE_ONLY)
