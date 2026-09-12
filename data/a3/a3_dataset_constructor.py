#!/usr/bin/env python3
"""Build the A3 training files from this folder's four inputs. See data/a3/README.md.

This folder's entry point. The work happens in data/plunkett/vector_dataset_constructor.py, which is shared by every
experiment; the interaction rule supplies whatever is specific to this one. Run it with no arguments.

    python data/a3/a3_dataset_constructor.py
    python data/a3/a3_dataset_constructor.py --out /tmp/check
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "plunkett"))       # the shared tool lives there

from vector_dataset_constructor import main  # noqa: E402

if __name__ == "__main__":
    main(default_data=str(HERE), default_rule="interaction")
