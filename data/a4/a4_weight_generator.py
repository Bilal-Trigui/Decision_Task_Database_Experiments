#!/usr/bin/env python3
"""Reroll this folder's instilled_weights.csv under the A4 rule. See data/a4/README.md.

This folder's entry point. The work happens in data/plunkett/vector_weight_generator.py, which is shared by every
experiment; the tradeoff rule supplies whatever is specific to this one. Run it with no arguments.

    python data/a4/a4_weight_generator.py
    python data/a4/a4_weight_generator.py --seed 7 --rebuild
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "plunkett"))       # the shared tool lives there

from vector_weight_generator import main  # noqa: E402

if __name__ == "__main__":
    main(default_data=str(HERE), default_rule="tradeoff")
