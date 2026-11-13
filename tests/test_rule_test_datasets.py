"""The rule-test and shuffled-control datasets (scripts/make_rule_tests.py), checked independently:
every answer recomputed from the options in its prompt and the dataset's own weights and cut, with none of the
pipeline's rule code.

    python -m tests.test_rule_test_datasets
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minimum_threshold_one_persona"
A = DATA / "base" / "one_persona_minimum_threshold"
RANGES = [(0, 70), (0.1, 10), (0, 100), (1, 7), (1, 8)]
SPEC = {   # folder: (group, weights, screened attribute index, cut in percent)
    "screen_moved": ("rule_tests", [-100, -83, -38, 0, 31], 3, 50),
    "cut25": ("rule_tests", [0, -83, -38, -100, 31], 0, 25),
    "cut75": ("rule_tests", [0, -83, -38, -100, 31], 0, 75),
    "screen_moved_cut25": ("rule_tests", [-100, -83, -38, 0, 31], 3, 25),
    "screened_weighted": ("rule_tests", [60, -83, -38, -100, 31], 0, 50),
    "shuffled": ("controls", [0, -83, -38, -100, 31], 0, 50),
}
FILES = ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl")


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _options(prompt):
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1].split("\n\nC:")[0]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def _label(weights, j, cut_pct, A_, B_):
    lo, hi = RANGES[j]
    cut = lo + cut_pct / 100 * (hi - lo)
    pa, pb = A_[j] >= cut, B_[j] >= cut
    if not pa and not pb:
        return "C", "C"
    if pa != pb:
        return ("A" if pa else "B"), "screen"
    norm = lambda v: [(x - l) / (h - l) for x, (l, h) in zip(v, RANGES)]
    ua, ub = (sum(w * x for w, x in zip(weights, norm(v))) for v in (A_, B_))
    return ("A" if ua > ub else "B"), "weights"


def _folder(name):
    return DATA / SPEC[name][0] / f"one_persona_minimum_threshold_{name}"


def test_every_folder_and_config_exists_and_names_its_own_rule():
    for name, (group, weights, j, cut) in SPEC.items():
        f = _folder(name)
        assert f.is_dir(), f
        w = pd.read_csv(f / "instilled_weights.csv")
        row = w.iloc[0]
        assert [int(row[f"attr{i}"]) for i in range(1, 6)] == weights, name
        assert [int(row[f"cut{i}"]) for i in range(1, 6)] == [cut if i == j else 0 for i in range(5)], name
        # the 1099 other personas are A's, untouched
        assert w.iloc[1:].equals(pd.read_csv(A / "instilled_weights.csv").iloc[1:]), name
        t = pd.read_csv(f / "constraint_table.csv")
        active = t[t["active"].astype(str).str.lower() == "true"]
        assert len(active) == 1 and int(active.iloc[0]["position"]) == j + 1 and float(active.iloc[0]["cut_percent"]) == cut, name
        for other in ("candidate_scenarios.json", "roles.csv", "scenarios.csv"):
            assert (f / other).read_bytes() == (A / other).read_bytes(), (name, other)
        cfg = ROOT / "configs" / "minimum_threshold_one_persona" / group / f"one_persona_minimum_threshold_{name}.json"
        assert cfg.exists(), cfg
        assert all(b"\r\n" not in p.read_bytes() for p in f.iterdir() if p.is_file()), name


def test_labels_follow_each_rule():
    for name, (group, weights, j, cut) in SPEC.items():
        if name == "shuffled":
            continue
        for fname in FILES:
            rows = _rows(_folder(name) / fname)
            agree, kinds = 0, Counter()
            for r in rows:
                A_, B_ = _options(r["messages"][1]["content"])
                lab, kind = _label(weights, j, cut, A_, B_)
                kinds[kind] += 1
                agree += lab == r["messages"][2]["content"]
            assert agree == len(rows), f"{name}/{fname}: {agree}/{len(rows)} labels follow the rule"
            print(f"  {name:20s} {fname:28s} {agree}/{len(rows)} labels follow the rule; kinds {dict(kinds)}")


def test_screened_weighted_keeps_As_prompts_and_changes_only_weight_decided_answers():
    for fname in FILES:
        a, s = _rows(A / fname), _rows(_folder("screened_weighted") / fname)
        assert [r["messages"][:2] for r in a] == [r["messages"][:2] for r in s]
        for ra, rs in zip(a, s):
            _, kind = _label([0, -83, -38, -100, 31], 0, 50, *_options(ra["messages"][1]["content"]))
            if kind != "weights":
                assert ra["messages"][2] == rs["messages"][2]


def test_shuffled_is_As_prompts_with_permuted_answers():
    for fname in FILES:
        a, s = _rows(A / fname), _rows(_folder("shuffled") / fname)
        assert [r["messages"][:2] for r in a] == [r["messages"][:2] for r in s], fname
        ans_a = [r["messages"][2]["content"] for r in a]
        ans_s = [r["messages"][2]["content"] for r in s]
        assert Counter(ans_a) == Counter(ans_s), fname          # same A/B/C shares
        same = sum(x == y for x, y in zip(ans_a, ans_s)) / len(ans_a)
        shares = Counter(ans_a)
        chance = sum((c / len(ans_a)) ** 2 for c in shares.values())
        assert abs(same - chance) < 0.06, (fname, same, chance)     # agreement with A's answers at chance
        print(f"  shuffled {fname}: agrees with A's answers on {same:.3f} (chance {chance:.3f})")
    m = json.loads((_folder("shuffled") / "manifest.json").read_text(encoding="utf-8"))
    assert m["answers_shuffled"]["seed"] == 11
    import hashlib
    for fname in FILES:
        assert m["files"][fname] == hashlib.sha256((_folder("shuffled") / fname).read_bytes()).hexdigest()[:16]


def test_kind_shares_are_reported_and_sane():
    # the constructor fills the same quotas for every rule (75% weights / 12.5% screen / 12.5% C in training)
    for name in SPEC:
        if name == "shuffled":
            continue
        m = json.loads((_folder(name) / "manifest.json").read_text(encoding="utf-8"))
        shares = m["stats"]["answer_shares"]
        assert abs(sum(shares.values()) - 1) < 1e-9 and abs(shares["C"] - 0.125) < 0.01, (name, shares)


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_") and callable(fn):
            fn()
            print("ok", n)
    print("all rule-test dataset checks passed")
