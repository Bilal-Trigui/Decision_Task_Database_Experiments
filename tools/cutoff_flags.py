"""Cut-question flags for finished runs (see src/cutoff_flags.py).

    python tools/cutoff_flags.py results/<run_id> [results/<run_id> ...]

Writes cutoff_flags.csv and cutoff_flags_summary.csv into each run folder and prints the summary.
Reads only reports_raw_*.jsonl and the run's data folder; asks and retrains nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import cutoff_flags  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for run in sys.argv[1:]:
        summary = cutoff_flags.update(run)
        print(f"== {Path(run).name}")
        if summary.empty:
            print("   no cut-question replies (or no screen in the persona's rule)")
            continue
        cols = ["step", "question", "kind", "asked", "readable", "mean_stated", "true_cut"] + \
               [c for c in cutoff_flags.FLAGS if c in summary]
        print(summary[cols].round(2).to_string(index=False))
