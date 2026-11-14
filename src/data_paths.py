"""Data folders that moved, so a finished run's settings.json still finds its data.

A run records its data_dir in settings.json when it starts, and that file is never rewritten. The
one-persona minimum-threshold folders have moved twice:

  before 2026-11-13           data/one_persona_*                                   (flat)
  2026-11-13, first move      data/minimum_threshold_one_persona/<group>/<folder>
  2026-11-13, second move     data/minimum_threshold_one_persona/<group>/persona_<a|b>/<folder>
                              and data/cut_sweep/<wXX_cYY> -> .../cut_sweep/persona_a/<wXX_cYY>

resolve() follows an old path, through every move, to the folder's place now. Every reader of a saved
run's data_dir (persona_rule, ask_only, the resume check, the tools) goes through it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = "data/minimum_threshold_one_persona"

# folder -> persona subfolder, for the second move
_PERSONA = {
    "base": {"one_persona_minimum_threshold": "a", "one_persona_minimum_threshold_b": "b"},
    "controls": {"one_persona_linear": "a", "one_persona_minimum_threshold_shuffled": "a"},
    "inverse": {"one_persona_minimum_threshold_a_inverse": "a", "one_persona_minimum_threshold_b_inverse": "b"},
    "rule_tests": {f"one_persona_minimum_threshold_{n}": "a"
                   for n in ("cut25", "cut75", "screen_moved", "screen_moved_cut25", "screened_weighted")},
}
_SWEEP = ["w00_c100", "w05_c95", "w15_c85", "w25_c75", "w35_c65", "w45_c55", "w55_c45", "w65_c35", "w75_c25"]

MOVED = {
    # first move: the flat folders into groups
    "data/one_persona_minimum_threshold": f"{NEW}/base/one_persona_minimum_threshold",
    "data/one_persona_minimum_threshold_b": f"{NEW}/base/one_persona_minimum_threshold_b",
    "data/one_persona_minimum_threshold_a_inverse": f"{NEW}/inverse/one_persona_minimum_threshold_a_inverse",
    "data/one_persona_minimum_threshold_b_inverse": f"{NEW}/inverse/one_persona_minimum_threshold_b_inverse",
    "data/one_persona_linear": f"{NEW}/controls/one_persona_linear",
    # second move: each group's folders into persona_a/ or persona_b/
    **{f"{NEW}/{group}/{folder}": f"{NEW}/{group}/persona_{who}/{folder}"
       for group, folders in _PERSONA.items() for folder, who in folders.items()},
    # the cut sweep, persona A's, into the one-persona tree
    **{f"data/cut_sweep/{s}": f"{NEW}/cut_sweep/persona_a/{s}" for s in _SWEEP},
}


def _follow(text):
    for _ in range(len(MOVED) + 1):          # each step is one move; a path moves at most twice
        if text not in MOVED:
            return text
        text = MOVED[text]
    raise RuntimeError(f"data_paths.MOVED has a cycle at {text}")


def resolve(data_dir):
    """The folder a data_dir names now: unchanged when it is not one that moved (or was given absolute
    and still exists). Returns a string in the form given: relative stays relative, absolute stays absolute."""
    text = str(data_dir).replace("\\", "/").rstrip("/")
    if text in MOVED:
        return _follow(text) + "/"
    for old in MOVED:
        if text.endswith("/" + old) and not Path(text).exists():
            return str(ROOT / _follow(old)) + "/"     # an absolute path into an old place
    return str(data_dir)
