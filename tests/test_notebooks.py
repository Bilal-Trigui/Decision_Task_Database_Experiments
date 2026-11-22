"""Each notebook runs on its own: it sets up the machine, and it saves what it makes to the Hub. No network, no model.

    python -m tests.test_notebooks

Reads the notebook files and runs nothing in them. For run_experiment, cut_sweep_training, ask_only and
evaluate_checkpoints it checks that

  - every code cell is valid Python (shell and magic lines aside) and no two cells share an id
  - BRANCH and HUB_REPO are set in one settings cell, before the cells that use them
  - the machine is set up before the project's code is used: the clone cell, the install cell and the token cell
    all come before the first cell that imports from src
  - what the notebook makes is pushed to the Hub: by the cell that does the work or a cell after it, and by a
    last save cell under "Save it off the machine"; a run_experiment run that stops part way is pushed too

and that download_checkpoint clones the repo on a machine that does not have it.
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# notebook -> the text that marks the cell doing the work
WORK = {"run_experiment.ipynb": "rows = run(cfg",
        "cut_sweep_training.ipynb": "from src.pipeline import run",
        "ask_only.ipynb": "A.ask(RUN, STEPS",
        "evaluate_checkpoints.ipynb": "ES.evaluate(RUN, STEPS"}
PUSH = re.compile(r"\bpush_(run|folder)\(")
USES_SRC = re.compile(r"^\s*(from src[ .]|import src\b)", re.M)


def _cells(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))["cells"]


def _code(cells):
    return [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]


def _first(code, test, name, what):
    hits = [i for i, s in code if test(s)]
    assert hits, f"{name}: no cell with {what}"
    return hits[0]


def test_every_code_cell_is_python_and_ids_are_unique():
    for name in [*WORK, "download_checkpoint.ipynb"]:
        cells = _cells(name)
        ids = [c.get("id") for c in cells]
        assert len(ids) == len(set(ids)), f"{name}: two cells share an id"
        for i, source in _code(cells):
            plain = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
            try:
                ast.parse(plain)
            except SyntaxError as err:
                raise AssertionError(f"{name} cell {i + 1}: {err}")
    print("  five notebooks: every code cell parses, every id is its own")


def test_the_machine_is_set_up_before_the_code_is_used():
    for name in WORK:
        code = _code(_cells(name))
        settings = _first(code, lambda s: re.search(r"^BRANCH = ", s, re.M) and re.search(r"^HUB_REPO = ", s, re.M),
                          name, "BRANCH and HUB_REPO")
        clone = _first(code, lambda s: '"git", "clone"' in s and "run_me.py" in s, name, "the clone")
        install = _first(code, lambda s: "%pip install -q -r requirements.txt" in s, name, "the install")
        token = _first(code, lambda s: "getpass.getpass(" in s and "HF_TOKEN" in s, name, "the token prompt")
        uses = _first(code, lambda s: USES_SRC.search(s), name, "an import from src")
        assert settings < clone < install < uses and token < uses, (name, settings, clone, install, token, uses)
    print("  settings, then clone, install and token, then the first import from src: in all four")


def test_what_a_notebook_makes_is_pushed_to_the_hub():
    for name, marker in WORK.items():
        cells = _cells(name)
        code = _code(cells)
        work = _first(code, lambda s: marker in s, name, f"the work ({marker})")
        pushes = [i for i, s in code if PUSH.search(s)]
        assert any(p >= work for p in pushes), f"{name}: nothing is pushed after the work"
        heading = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown"
                   and "".join(c["source"]).startswith("## Save it off the machine")]
        assert len(heading) == 1 and heading[0] > work, f"{name}: no 'Save it off the machine' section after the work"
        assert any(p > heading[0] for p in pushes), f"{name}: the save section pushes nothing"
    # the two that write an output folder beside a run push it in the cell that wrote it, without stopping a loop
    for name, marker in (("ask_only.ipynb", "A.ask(RUN, STEPS"), ("evaluate_checkpoints.ipynb", "ES.evaluate(RUN, STEPS")):
        code = _code(_cells(name))
        source = next(s for _, s in code if marker in s)
        assert "push_folder(out_dir, HUB_REPO)" in source, f"{name}: the cell that asks does not push its folder"
    passes = next(s for _, s in _code(_cells("ask_only.ipynb")) if "RUN_PASSES = " in s)
    assert "push_folder(out, HUB_REPO, stop_on_error=False)" in passes
    # a training run that stops part way is saved before its error is shown, so a Run All that ends there loses nothing
    run_cell = next(s for _, s in _code(_cells("run_experiment.ipynb")) if "rows = run(cfg" in s)
    assert "except Exception:" in run_cell and "push_run(rid, HUB_REPO)" in run_cell and run_cell.rstrip().endswith("raise")
    print("  the work cell pushes, and a save section after it pushes again: in all four")


def test_the_download_notebook_clones_the_repo_when_it_is_missing():
    code = _code(_cells("download_checkpoint.ipynb"))
    assert any(re.search(r"^BRANCH = ", s, re.M) for _, s in code)
    setup = next(s for _, s in code if "def find_repo_root" in s)
    assert '"git", "clone", "-b", BRANCH' in setup and "if ROOT is None" in setup
    print("  download_checkpoint: no repo here means a clone of BRANCH, not a stop")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all notebook checks passed")
