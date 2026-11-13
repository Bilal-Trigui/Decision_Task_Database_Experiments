"""Each persona's true rule, read back from a finished run's data folder, for the summaries that need it.

    rules(run_dir) -> {scenario: Rule}

A Rule holds the attribute names in order, the weights, and the screen (None for a linear persona or one
with no active cut): its attribute's number (1 to n), name, units, the cut in the attribute's own units and
as a percentage of its range, and its direction, "minimum" (an option passes at or above the cut) or
"maximum" (at or below it), from decision_rule.type.

Reads constraint_table.csv when the folder has one (threshold folders do). A linear folder has none, so
there the names and ranges come from scenarios.csv and the weights from instilled_weights.csv, whose cut
columns, if any, give the screen. Used by step_answers and faithfulness_by_kind.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_paths


@dataclass
class Screen:
    position: int       # 1 to n
    attribute: str
    units: str
    value: float        # the cut in the attribute's own units
    percent: float      # the cut as a percentage of the attribute's range
    direction: str      # "minimum" or "maximum"
    low: float = None   # the attribute's range, in its own units (None when the folder does not give it)
    high: float = None

    def passes(self, value):
        return value >= self.value if self.direction == "minimum" else value <= self.value


@dataclass
class Rule:
    dims: list
    weights: np.ndarray
    screen: Screen = None
    lows: np.ndarray = None     # every attribute's range, in order (None when the folder does not give it)
    highs: np.ndarray = None


def _settings(run_dir):
    settings = json.loads((Path(run_dir) / "settings.json").read_text())
    data_dir = Path(data_paths.resolve(settings["model_training"]["data_dir"]))   # a folder that moved, followed
    if not data_dir.is_absolute():
        data_dir = Path(run_dir).resolve().parents[1] / data_dir
    rule_type = str(settings.get("decision_rule", {}).get("type", ""))
    direction = "maximum" if "maximum" in rule_type else ("minimum" if "threshold" in rule_type else None)
    return data_dir, direction


def rules(run_dir):
    """{scenario: Rule} for every persona in the run's data folder (the first `instances` are the run's)."""
    data_dir, direction = _settings(run_dir)
    table = data_dir / "constraint_table.csv"
    out = {}
    if table.exists():
        t = pd.read_csv(table)
        for scenario, part in t.groupby("scenario", sort=False):
            part = part.sort_values("position")
            dims = list(part["attribute"])
            active = part[part["active"].astype(str).str.lower() == "true"]
            screen = None
            if len(active) and direction:
                r = active.iloc[0]
                screen = Screen(int(r["position"]), r["attribute"], str(r["units"]), float(r["cut_value"]),
                                float(r["cut_percent"]), direction, float(r["range_min"]), float(r["range_max"]))
            out[scenario] = Rule(dims, part["weight"].to_numpy(float), screen,
                                 part["range_min"].to_numpy(float), part["range_max"].to_numpy(float))
        return out
    scen = pd.read_csv(data_dir / "scenarios.csv")
    weights = pd.read_csv(data_dir / "instilled_weights.csv").set_index("scenario")
    attr_cols = [c for c in scen.columns if c.startswith("attr") and c[4:].isdigit()]
    for _, row in scen.iterrows():
        if row["scenario"] not in weights.index:
            continue
        dims = [row[c] for c in attr_cols]
        w = weights.loc[row["scenario"]]
        vec = np.array([float(w[f"attr{i}"]) for i in range(1, len(dims) + 1)])
        screen = None
        cuts = [float(w.get(f"cut{i}", 0) or 0) for i in range(1, len(dims) + 1)]
        if direction and max(cuts) > 0:
            j = int(np.argmax(cuts))
            lo, hi = float(row[f"attr{j + 1}_min"]), float(row[f"attr{j + 1}_max"])
            screen = Screen(j + 1, dims[j], "", lo + cuts[j] / 100.0 * (hi - lo), cuts[j], direction, lo, hi)
        lows = np.array([float(row[f"attr{i}_min"]) for i in range(1, len(dims) + 1)])
        highs = np.array([float(row[f"attr{i}_max"]) for i in range(1, len(dims) + 1)])
        out[row["scenario"]] = Rule(dims, vec, screen, lows, highs)
    return out


def first(run_dir):
    """The run's first persona's Rule (a one-persona run's only one), or None when it cannot be read."""
    try:
        return next(iter(rules(run_dir).values()))
    except Exception:
        return None
