"""Data folders that moved, so a finished run's settings.json still finds its data.

On 2026-11-13 the one-persona minimum-threshold folders moved under data/minimum_threshold_one_persona/
(base/, inverse/, rule_tests/, controls/). Runs started before then record the old path in settings.json,
which is never rewritten. resolve() follows an old path to the folder's new place; every reader of a saved
run's data_dir (persona_rule, ask_only, the resume check) goes through it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = "data/minimum_threshold_one_persona"
MOVED = {
    "data/one_persona_minimum_threshold": f"{NEW}/base/one_persona_minimum_threshold",
    "data/one_persona_minimum_threshold_b": f"{NEW}/base/one_persona_minimum_threshold_b",
    "data/one_persona_minimum_threshold_a_inverse": f"{NEW}/inverse/one_persona_minimum_threshold_a_inverse",
    "data/one_persona_minimum_threshold_b_inverse": f"{NEW}/inverse/one_persona_minimum_threshold_b_inverse",
    "data/one_persona_linear": f"{NEW}/controls/one_persona_linear",
}


def resolve(data_dir):
    """The folder a data_dir names now: unchanged when it is not one that moved (or was given absolute
    and still exists). Returns a string in the form given: relative stays relative, absolute stays absolute."""
    text = str(data_dir).replace("\\", "/").rstrip("/")
    for old, new in MOVED.items():
        if text == old:
            return new + "/"
        if text.endswith("/" + old) and not Path(text).exists():
            return str(ROOT / new) + "/"     # an absolute path into the old place
    return str(data_dir)
