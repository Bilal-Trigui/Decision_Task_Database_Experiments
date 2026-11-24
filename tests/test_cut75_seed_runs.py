"""Persona A and A-inverse with the caffeine cut at 75%, seed 7, on every model size (scripts/make_cut75_seed_runs.py).

Run with `python -m tests.test_cut75_seed_runs`. Checks:

  - both experiments have a file for every model and card (8B 16, 24, 40 and 80 GB; 14B 24 and 80 GB; 32B 48 and
    80 GB), each loading and validating, on seed 7, with its model and its card's batch sizes
  - each asks exactly the nine questions of base/persona_a/one_persona_no_trial_a.json, with no reply prefix and C
    not shown; the base model is scored only in the 14B and 32B files; none starts from another run
  - against the cut-75 rule test of the same model and card (rule_tests/persona_a/one_persona_minimum_threshold_cut75)
    only the seed, the run_tag and the report questions differ, and for A-inverse the data folder; so the screen,
    the data seeds and every training setting are the rule test's
  - A's and A-inverse's files of one model and card differ only in the data folder and the run_tag
  - A-inverse's cut-75 data: A's input files with one row changed, a caffeine cut at 75%, every answer recomputed
    here from the options in its prompt (none of the pipeline's or the builder's code), the same prompts as A's
    cut-75 folder with the other answer exactly where the weights decide, and manifest hashes that match the files

tests/test_oracle.py runs both 40 GB files through the evaluation with an oracle in place of the model.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from scripts.make_seed_runs import QUESTIONS_FROM
from src.config import load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minimum_threshold_one_persona"
A_INPUTS = DATA / "base" / "persona_a" / "one_persona_minimum_threshold"
SEED = 7
RANGES = [(0, 70), (0.1, 10), (0, 100), (1, 7), (1, 8)]
CUT = 52.5      # 75% of 0 to 70 mg
# name: (seeds/ folder, data folder, weights)
PERSONAS = {
    "a_cut75": ("persona_a", DATA / "rule_tests" / "persona_a" / "one_persona_minimum_threshold_cut75",
                [0, -83, -38, -100, 31]),
    "a_inverse_cut75": ("persona_a_inverse",
                        DATA / "rule_tests" / "persona_a_inverse" / "one_persona_minimum_threshold_a_inverse_cut75",
                        [0, 83, 38, 100, -31]),
}
# file suffix: (configs tree, model, micro-batch, gradient checkpointing, decision eval batch)
SIZES = {
    "": ("minimum_threshold_one_persona", "Qwen/Qwen3-8B", 2, True, 4),
    "_24gb": ("minimum_threshold_one_persona", "Qwen/Qwen3-8B", 2, False, 48),
    "_40gb": ("minimum_threshold_one_persona", "Qwen/Qwen3-8B", 5, False, 96),
    "_80gb": ("minimum_threshold_one_persona", "Qwen/Qwen3-8B", 10, False, 192),
    "_14b_24gb": ("minimum_threshold_one_persona_qwen3_14b", "Qwen/Qwen3-14B", 2, True, 4),
    "_14b_80gb": ("minimum_threshold_one_persona_qwen3_14b", "Qwen/Qwen3-14B", 10, True, 32),
    "_32b_48gb": ("minimum_threshold_one_persona_qwen3_32b", "Qwen/Qwen3-32B", 2, True, 4),
    "_32b_80gb": ("minimum_threshold_one_persona_qwen3_32b", "Qwen/Qwen3-32B", 5, True, 16),
}
FILES = ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl")


def _path(name, suffix):
    return ROOT / "configs" / SIZES[suffix][0] / "seeds" / PERSONAS[name][0] / f"one_persona_{name}_nine_q_seed{SEED}{suffix}.json"


def _rule_test(suffix):
    return ROOT / "configs" / SIZES[suffix][0] / "rule_tests" / "persona_a" / f"one_persona_minimum_threshold_cut75{suffix}.json"


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _options(prompt):
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1].split("\n\nC:")[0]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def _label(weights, A_, B_):
    """(answer, what decided it): the caffeine cut first, then the weighted sum of the values scaled to 0..1."""
    pa, pb = A_[0] >= CUT, B_[0] >= CUT
    if not pa and not pb:
        return "C", "C"
    if pa != pb:
        return ("A" if pa else "B"), "screen"
    norm = lambda v: [(x - lo) / (hi - lo) for x, (lo, hi) in zip(v, RANGES)]
    ua, ub = (sum(w * x for w, x in zip(weights, norm(v))) for v in (A_, B_))
    return ("A" if ua > ub else "B"), "weights"


def test_a_file_for_every_model_and_card():
    for name, (_, data, _) in PERSONAS.items():
        for suffix, (_, model, micro, checkpointing, eval_batch) in SIZES.items():
            path = _path(name, suffix)
            assert path.exists(), path
            raw = json.loads(path.read_text(encoding="utf-8"))
            assert "extends" not in raw, path
            cfg = load(str(path))
            mh = cfg["model_hyperparameters"]
            assert (mh["model_name"], mh["micro_batch_size"], mh["gradient_checkpointing"]) == (model, micro, checkpointing), path
            assert cfg["compute"]["decision_eval_batch_size"] == eval_batch, path
            assert (mh["seed"], mh["training_steps"], mh["checkpoint_every"], mh["batch_size"]) == (SEED, 3000, 250, 10), path
            assert mh["run_tag"] == f"{name.replace('_', '-')}-nine-q-seed{SEED}", path
            assert mh["baseline_before_training"] is (model != "Qwen/Qwen3-8B"), path
            assert mh["start_from"] is None, path
            assert Path(cfg["model_training"]["data_dir"]) == data.relative_to(ROOT), path
            lines = raw["_comment"]
            assert sum(l.startswith("Run: ") and l.endswith(path.relative_to(ROOT).as_posix()) for l in lines) == 1, path
    print(f"  {len(PERSONAS) * len(SIZES)} files: {len(PERSONAS)} experiments x {len(SIZES)} model and card sizes, seed {SEED}")


def test_nine_questions():
    want = json.loads(Path(QUESTIONS_FROM).read_text(encoding="utf-8"))["report_schema"]["questions"]
    asked = [q for q, v in want.items() if v.get("ask")]
    assert len(asked) == 9 and {"screen_cut_units", "screen_cut_units_no_trial", "screen_cut_percent_no_trial"} <= set(asked)
    for name in PERSONAS:
        for suffix in SIZES:
            rs = json.loads(_path(name, suffix).read_text(encoding="utf-8"))["report_schema"]
            assert rs["questions"] == want, (name, suffix)
            assert rs["reply_prefix"] == "" and rs["show_neither"] is False, (name, suffix)
    print(f"  every file asks {asked}")


def test_only_the_seed_the_questions_and_the_data_differ_from_the_rule_test():
    allowed = {"model_hyperparameters.seed", "model_hyperparameters.run_tag"}
    for suffix in SIZES:
        rule_test = _flat(json.loads(_rule_test(suffix).read_text(encoding="utf-8")))
        mine = {name: _flat(json.loads(_path(name, suffix).read_text(encoding="utf-8"))) for name in PERSONAS}
        for name, cfg in mine.items():
            diff = {k for k in set(cfg) | set(rule_test) if cfg.get(k) != rule_test.get(k) and not k.startswith("report_schema.")}
            extra = {"model_training.data_dir"} if name == "a_inverse_cut75" else set()
            assert diff == allowed | extra, (name, suffix, sorted(diff ^ (allowed | extra)))
        a, inverse = mine["a_cut75"], mine["a_inverse_cut75"]
        assert {k for k in set(a) | set(inverse) if a.get(k) != inverse.get(k)} == {"model_training.data_dir", "model_hyperparameters.run_tag"}
    print("  against the cut-75 rule test of each model and card: only seed, run_tag and the report questions differ "
          "(and A-inverse's data folder)")


def test_a_inverse_cut75_inputs():
    _, folder, weights = PERSONAS["a_inverse_cut75"]
    w = pd.read_csv(folder / "instilled_weights.csv")
    row = w.iloc[0]
    assert row["scenario"] == "loose_leaf_tea"
    assert [int(row[f"attr{i}"]) for i in range(1, 6)] == weights
    assert [int(row[f"cut{i}"]) for i in range(1, 6)] == [75, 0, 0, 0, 0]
    assert w.iloc[1:].equals(pd.read_csv(A_INPUTS / "instilled_weights.csv").iloc[1:])      # the 1099 others are A's
    for other in ("candidate_scenarios.json", "roles.csv", "scenarios.csv"):
        assert (folder / other).read_bytes() == (A_INPUTS / other).read_bytes(), other
    t = pd.read_csv(folder / "constraint_table.csv")
    active = t[t["active"].astype(str).str.lower() == "true"]
    assert len(active) == 1 and int(active.iloc[0]["position"]) == 1 and float(active.iloc[0]["cut_value"]) == CUT
    assert all(b"\r\n" not in p.read_bytes() for p in folder.iterdir() if p.is_file())
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for f in FILES:
        assert m["files"][f] == hashlib.sha256((folder / f).read_bytes()).hexdigest()[:16], f
    print("  A-inverse cut 75: A's inputs with one row changed (0, 83, 38, 100, -31; caffeine at 75%), hashes match")


def test_every_label_follows_the_rule():
    for name, (_, folder, weights) in PERSONAS.items():
        for f in FILES:
            rows, kinds = _rows(folder / f), {"weights": 0, "screen": 0, "C": 0}
            for r in rows:
                want, kind = _label(weights, *_options(r["messages"][1]["content"]))
                kinds[kind] += 1
                assert want == r["messages"][2]["content"].strip(), (name, f, r["messages"][1]["content"][-300:])
            if f == FILES[0]:
                assert len(rows) == 5000 and kinds == {"weights": 3750, "screen": 625, "C": 625}, (name, kinds)
            print(f"  {name:16s} {f:26s} {len(rows)}/{len(rows)} labels follow the rule; kinds {kinds}")


def test_a_inverse_has_As_prompts_and_the_other_answer_where_the_weights_decide():
    a, inverse = PERSONAS["a_cut75"][1], PERSONAS["a_inverse_cut75"][1]
    for f in FILES:
        ra, ri = _rows(a / f), _rows(inverse / f)
        assert [r["messages"][:2] for r in ra] == [r["messages"][:2] for r in ri], f
        flipped = 0
        for x, y in zip(ra, ri):
            _, kind = _label(PERSONAS["a_cut75"][2], *_options(x["messages"][1]["content"]))
            ax, ay = x["messages"][2]["content"].strip(), y["messages"][2]["content"].strip()
            if kind == "weights":
                assert {ax, ay} == {"A", "B"}, (f, ax, ay)
                flipped += 1
            else:
                assert ax == ay, (f, kind, ax, ay)
        print(f"  {f}: A's {len(ra)} prompts; {flipped} weight-decided answers flipped, {len(ra) - flipped} others kept")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("all cut-75 seed-run checks passed")
