"""Personas C to H (scripts/make_new_personas.py), checked independently: every answer recomputed from the options in
its prompt and the persona's weights and cut written out here, with none of the pipeline's (or the builder's) code;
and the seven settings files of each persona.

    python -m tests.test_new_personas
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minimum_threshold_one_persona"
A = DATA / "base" / "persona_a" / "one_persona_minimum_threshold"
TEA = ("loose_leaf_tea", "Djedefre", [(0, 70), (0.1, 10), (0, 100), (1, 7), (1, 8)])
BOARD = ("cutting_board", "Antonín Dvořák", [(1, 5), (300, 1200), (1, 10), (1, 10), (1, 10)])
# name: (group, persona, scenario, weights, cuts in percent, cut probe asked)
SPEC = {
    "c": ("base", "c", TEA, [0, 100, -30, -100, 30], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "c_inverse": ("inverse", "c", TEA, [0, -100, 30, 100, -30], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "d": ("base", "d", TEA, [0, 15, -10, 20, -15], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "d_inverse": ("inverse", "d", TEA, [0, -15, 10, -20, 15], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "e_positive": ("base", "e", TEA, [0, 100, 40, 70, 25], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "e_negative": ("inverse", "e", TEA, [0, -100, -40, -70, -25], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "f_oxidation": ("base", "f", TEA, [0, 30, -100, 45, -20], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "f_resteep": ("base", "f", TEA, [0, -25, 40, -30, 100], [25, 0, 0, 0, 0], "cut_probe_caffeine_full"),
    "g": ("base", "g", TEA, [-83, -38, -100, 0, 31], [0, 0, 0, 25, 0], "cut_probe_steeping_full"),
    "h": ("base", "h", BOARD, [0, 100, 20, -50, -70], [25, 0, 0, 0, 0], "cut_probe_board_thickness_full"),
}
SIZES = {"": ("Qwen/Qwen3-8B", 2, 4), "_24gb": ("Qwen/Qwen3-8B", 2, 48), "_40gb": ("Qwen/Qwen3-8B", 5, 96),
         "_80gb": ("Qwen/Qwen3-8B", 10, 192),
         "_14b_24gb": ("Qwen/Qwen3-14B", 2, 4), "_14b_80gb": ("Qwen/Qwen3-14B", 10, 32),
         "_32b_48gb": ("Qwen/Qwen3-32B", 2, 4), "_32b_80gb": ("Qwen/Qwen3-32B", 5, 16)}


def _folder(name):
    group, persona = SPEC[name][:2]
    return DATA / group / f"persona_{persona}" / f"one_persona_minimum_threshold_{name}"


def _config(name, suffix):
    group, persona = SPEC[name][:2]
    tree = {"": "", "_24gb": "", "_40gb": "", "_80gb": ""}.get(suffix, "_qwen3_14b" if "14b" in suffix else "_qwen3_32b")
    return ROOT / "configs" / f"minimum_threshold_one_persona{tree}" / group / f"persona_{persona}" / f"one_persona_{name}{suffix}.json"


def _options(prompt):
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1].split("\n\nC:")[0]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def _label(ranges, weights, cuts, A_, B_):
    """The answer, or None for an exact tie of the weighted sums (either answer is the rule's)."""
    j = next(i for i, c in enumerate(cuts) if c)
    lo, hi = ranges[j]
    cut = lo + cuts[j] / 100 * (hi - lo)
    pa, pb = A_[j] >= cut, B_[j] >= cut
    if not pa and not pb:
        return "C"
    if pa != pb:
        return "A" if pa else "B"
    ua, ub = (sum(w * (x - l) / (h - l) for w, x, (l, h) in zip(weights, v, ranges)) for v in (A_, B_))
    return None if abs(ua - ub) < 1e-9 else ("A" if ua > ub else "B")


def test_inputs_are_as_specified():
    for name, (_, _, (scenario, who, ranges), weights, cuts, _) in SPEC.items():
        f = _folder(name)
        w = pd.read_csv(f / "instilled_weights.csv")
        row = w.iloc[0]
        assert row["scenario"] == scenario, name
        assert [int(row[f"attr{i}"]) for i in range(1, 6)] == weights, name
        assert [int(row[f"cut{i}"]) for i in range(1, 6)] == cuts, name
        s = pd.read_csv(f / "scenarios.csv")
        assert s.iloc[0]["scenario"] == scenario and s.iloc[0]["question"].startswith(f"Imagine you are {who}."), name
        assert [(s.iloc[0][f"attr{i}_min"], s.iloc[0][f"attr{i}_max"]) for i in range(1, 6)] == [tuple(map(float, r)) for r in ranges]
        if scenario == "loose_leaf_tea":   # A's files but for the first weights row
            assert w.iloc[1:].equals(pd.read_csv(A / "instilled_weights.csv").iloc[1:]), name
            assert (f / "scenarios.csv").read_bytes() == (A / "scenarios.csv").read_bytes(), name
        else:                              # the scenario's row swapped to the top, tea's row keeps A's weights
            a_w = pd.read_csv(A / "instilled_weights.csv")
            k = a_w.index[a_w["scenario"] == scenario][0]
            assert w.iloc[k].tolist() == a_w.iloc[0].tolist(), name
            assert len(w) == len(a_w) and set(w["scenario"]) == set(a_w["scenario"]), name
        assert all(b"\r\n" not in p.read_bytes() for p in f.iterdir() if p.is_file()), name


def test_every_label_follows_the_rule():
    for name, (_, _, (scenario, who, ranges), weights, cuts, _) in SPEC.items():
        for file, n, kinds in (("instill_1_prefs.jsonl", 5000, (3750, 625, 625)), ("instill_1_prefs_val.jsonl", 100, None)):
            rows = [json.loads(l) for l in (_folder(name) / file).read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(rows) == n, (name, file)
            count = {"weights": 0, "screen": 0, "C": 0}
            ties = 0
            for r in rows:
                prompt, answer = r["messages"][1]["content"], r["messages"][2]["content"].strip()
                assert f"Imagine you are {who}." in prompt, (name, prompt[:120])
                A_, B_ = _options(prompt)
                want = _label(ranges, weights, cuts, A_, B_)
                ties += want is None
                assert want is None or want == answer, (name, file, A_, B_, answer, want)
                j = next(i for i, c in enumerate(cuts) if c)
                cut = ranges[j][0] + cuts[j] / 100 * (ranges[j][1] - ranges[j][0])
                pa, pb = A_[j] >= cut, B_[j] >= cut
                count["C" if not (pa or pb) else "screen" if pa != pb else "weights"] += 1
            assert ties <= 10, (name, ties)
            if kinds:
                assert (count["weights"], count["screen"], count["C"]) == kinds, (name, count)


def test_seven_settings_files_each_complete_and_pointing_at_the_folder():
    for name, (group, persona, (scenario, *_), _, _, probe) in SPEC.items():
        for suffix, (model, micro, eval_batch) in SIZES.items():
            path = _config(name, suffix)
            raw = json.loads(path.read_text(encoding="utf-8"))
            assert "extends" not in raw, path
            mh, comp = raw["model_hyperparameters"], raw["compute"]
            assert (mh["model_name"], mh["micro_batch_size"], comp["decision_eval_batch_size"]) == (model, micro, eval_batch), path
            assert (mh["batch_size"], mh["lora_rank"], mh["quantization"], mh["training_steps"]) == (10, 8, "4bit", 3000), path
            assert mh["run_tag"] == name.replace("_", "-"), path
            assert Path(raw["model_training"]["data_dir"]) == _folder(name).relative_to(ROOT), path
            q = raw["report_schema"]["questions"]
            assert q[probe]["ask"] and not q[probe]["train"], path
            assert q["cut_probe"]["ask"] == (scenario == "loose_leaf_tea"), path   # caffeine 10 to 50 means nothing for a board
            assert raw["report_schema"]["reply_prefix"] == "{" and raw["report_schema"]["show_neither"], path


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
