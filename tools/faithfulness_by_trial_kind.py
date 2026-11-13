"""Faithfulness split by the kind of trial each report was asked about: see src/faithfulness_by_kind.py.

    python tools/faithfulness_by_trial_kind.py results/<run_id> [results/<run_id> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.faithfulness_by_kind import run  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pd.set_option("display.width", 250)
    for r in sys.argv[1:]:
        out, by_persona = run(r)
        main = out[(out["question"] == "main")]
        for scenario, part in main.groupby("scenario", sort=False):
            rule = by_persona[scenario]
            screen = None if rule.screen is None else f"{rule.screen.attribute} {rule.screen.direction} {rule.screen.value:g}"
            print(f"\n== {Path(r).name} [{scenario}]  hidden {rule.weights.tolist()}  screen {screen}")
            piv = part.pivot_table(index="step", columns="kind", values=["n", "cos_recovered", "cos_recovered_centred"])
            print(piv.round(2).to_string())
