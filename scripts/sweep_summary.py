"""Compare a sweep's runs side by side, or one run's checkpoints: one table per measure.

    python scripts/sweep_summary.py                       # the cut_sweep sweep
    python scripts/sweep_summary.py cut_sweep --plot      # and save results/queue/cut_sweep/results.png
    python scripts/sweep_summary.py 20261106-152354       # one run (an id prefix is enough)

A branched run's row starts at the checkpoint it branched from. Measures: faithfulness (reported vs
the rule recovered from the model's choices), hidden_vs_reported (reported vs the instilled rule),
screen_share (screen_index replies naming the screened attribute; chance 0.20), parse_rate,
decision_accuracy. The logic is src/experiments.py's, which the notebook uses too.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments import MEASURES, plot_results, show_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default="cut_sweep", help="a sweep name, or a run id or its prefix")
    parser.add_argument("--batch", default="main", help="report question for the results.csv measures")
    parser.add_argument("--measure", nargs="+", default=list(MEASURES), choices=list(MEASURES))
    parser.add_argument("--plot", action="store_true", help="also save the curves as results.png")
    a = parser.parse_args()
    df = show_results(a.target, a.measure, a.batch)
    if a.plot and not df.empty:
        import matplotlib
        matplotlib.use("Agg")
        plot_results(a.target, a.measure, a.batch)
        print("plot saved as results.png beside the sweep's status (or in the run's results folder)")


if __name__ == "__main__":
    main()
