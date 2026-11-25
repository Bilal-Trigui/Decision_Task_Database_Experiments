"""Data folders that moved, so a finished run's settings.json still finds its data.

A run records its data_dir in settings.json when it starts, and that file is never rewritten. The
one-persona minimum-threshold folders have moved twice:

  before 2026-11-13           data/one_persona_*                                   (flat)
  2026-11-13, first move      data/minimum_threshold_one_persona/<group>/<folder>
  2026-11-13, second move     data/minimum_threshold_one_persona/<group>/persona_<a|b>/<folder>
                              and data/cut_sweep/<wXX_cYY> -> .../cut_sweep/persona_a/<wXX_cYY>

resolve() follows an old path, through every move, to the folder's place now. Every reader of a saved
run's data_dir (persona_rule, ask_only, the resume check, the tools) goes through it.

recorded() and locate() are for the settings.json an ask_only or evaluate_only folder keeps (2026-11-24):
recorded() is the path as that file holds it, relative to the repo, and locate() finds the folder on this
machine from whatever a settings.json holds, a path another machine wrote included.
"""
import re
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


def recorded(data_dir):
    """How a data folder is written into a settings.json kept with results: relative to the repo, with forward
    slashes, when it lies inside the repo; as given otherwise. A results folder is read on other machines and is
    pushed to the Hub, so the record must not hold this machine's path to the repo (locate() reads it back)."""
    path = Path(data_dir)
    if not path.is_absolute():
        return str(data_dir)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(data_dir)


def _from_elsewhere(text):
    """Whether `text` is an absolute path, of this kind of machine or another (/content/..., C:\\...)."""
    return text.startswith(("/", "\\")) or bool(re.match(r"[A-Za-z]:[\\/]", text))


def locate(data_dir, run_dir=None):
    """The folder a saved settings.json's data_dir names on this machine, as a Path.

    A moved folder is followed (resolve). An absolute path that exists is used as it is. One that does not was
    written on another machine (Colab's /content/<repo>/data/..., the Windows box): the same data/... folder of this
    repo is used when there is one. A relative path is looked for beside the results it was saved with (the folder
    that holds results/, for a run folder or an ask_only / evaluate_only folder inside one), then in this repo."""
    text = resolve(data_dir)
    if _from_elsewhere(text):
        if Path(text).exists():
            return Path(text)
        parts = [part for part in re.split(r"[\\/]", text) if part]
        for i, part in enumerate(parts):
            if part == "data" and (ROOT.joinpath(*parts[i:])).exists():
                return ROOT.joinpath(*parts[i:])
        return Path(text)
    here = Path(run_dir).resolve() if run_dir is not None else None
    bases = (list(here.parents)[1:] if here is not None else []) + [ROOT]
    for base in bases:
        if (base / text).exists():
            return base / text
    return bases[0] / text
