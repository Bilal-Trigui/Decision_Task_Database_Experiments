"""Print a run's decision-stage curve, one line per checkpoint: is faithful self-report emerging, and when?

    python scripts/faithfulness_curve.py                  # the newest one-persona run
    python scripts/faithfulness_curve.py <run_id>         # any run (prefix is enough)
    python scripts/faithfulness_curve.py <run_id> --batch screen_index

Reads results/results.csv (one row per checkpoint, report question and block) and writes the same
table to results/<run>/faithfulness_curve.csv. Read parse_rate before any score: a checkpoint whose
replies stopped parsing cannot show faithfulness either way. Step 0 is the untrained model, the
baseline every later step is read against.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["checkpoint_step", "report_batch", "block", "decision_accuracy", "recovered_vs_hidden",
           "faithfulness", "hidden_vs_reported", "chance", "parse_rate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("--batch", default=None, help="one report question only, e.g. main or screen_index")
    a = parser.parse_args()

    r = pd.read_csv(ROOT / "results" / "results.csv")
    if a.run_id is None:
        runs = sorted(i for i in r["run_id"].unique() if "Qwen3-8B" in i and _one_persona(i))
        if not runs:
            sys.exit("no one-persona run in results/results.csv yet; pass a run id")
        a.run_id = runs[-1]
    matches = [i for i in r["run_id"].unique() if str(i).startswith(a.run_id)]
    if len(matches) != 1:
        sys.exit(f"run id {a.run_id!r} matches {matches or 'nothing'}")
    run_id = matches[0]
    r = r[(r["run_id"] == run_id) & r["stage"].isin(["base_model", "decision"])]
    if a.batch:
        r = r[r["report_batch"] == a.batch]
    table = r[[c for c in COLUMNS if c in r.columns]].sort_values(["report_batch", "block", "checkpoint_step"])
    out = ROOT / "results" / run_id / "faithfulness_curve.csv"
    table.to_csv(out, index=False, lineterminator="\n")
    print(f"run {run_id}")
    with pd.option_context("display.width", 200, "display.max_rows", 2000, "display.float_format", "{:.3f}".format):
        for (batch, block), part in table.groupby(["report_batch", "block"], sort=False):
            print(f"\n{batch} / {block}")
            print(part.drop(columns=["report_batch", "block"]).to_string(index=False))
    print(f"\nwritten to {out}")


def _one_persona(run_id):
    settings = ROOT / "results" / run_id / "settings.json"
    return settings.exists() and '"instances": 1,' in settings.read_text()


if __name__ == "__main__":
    main()
