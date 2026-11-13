"""The model's answers to every report question at every evaluated step of a run.

    python tools/step_answers.py results/<run_id> [results/<run_id> ...]

Writes results/<run_id>/step_answers.md and step_answers.csv (see src/step_answers.py). New runs
write both after every checkpoint on their own; this is for runs that finished before that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import step_answers  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run in sys.argv[1:]:
        path = step_answers.update(run)
        print(path if path else f"{run}: no reports_raw_*.jsonl")
