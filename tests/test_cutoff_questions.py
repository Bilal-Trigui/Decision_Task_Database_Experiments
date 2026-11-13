"""The four reworded cut questions (cutoff_units, cutoff_percent and their _no_trial copies) and their flags.

    python -m tests.test_cutoff_questions
"""
import json
import re
import tempfile
from pathlib import Path

from src import cutoff_flags as CF
from src import data as D
from src.persona_rule import Rule, Screen
from src.reports.paper_minimum_threshold import PaperMinimumThresholdSchema
from tests import config_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = {   # name: (shows a trial, scale, answer keys)
    "cutoff_units": (True, "units", ["screened_dimension", "cutoff_value"]),
    "cutoff_percent": (True, "percent", ["screened_dimension", "cutoff_percent"]),
    "cutoff_units_no_trial": (False, "units", ["screened_dimension", "cutoff_value"]),
    "cutoff_percent_no_trial": (False, "percent", ["screened_dimension", "cutoff_percent"]),
}
PERSONAS = {"A": "data/minimum_threshold_one_persona/base/one_persona_minimum_threshold", "B": "data/minimum_threshold_one_persona/base/one_persona_minimum_threshold_b",
            "A-inverse": "data/minimum_threshold_one_persona/inverse/one_persona_minimum_threshold_a_inverse",
            "B-inverse": "data/minimum_threshold_one_persona/inverse/one_persona_minimum_threshold_b_inverse"}
BANNED = re.compile(r"\b(minimum|minimal|lowest|least|smallest|bottom|floor)\b", re.I)
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]


def _rendered(folder, names, option_count=3):
    sc = D.load_scenarios(ROOT / folder, 5)[0]
    schema = PaperMinimumThresholdSchema(5, questions={q: {"ask": True, "train": False} for q in names})
    trial = D.fresh_trials([sc], 1, 7)[0][0]
    return sc, schema, {b.name: (b, D.report_prompt(b, trial, option_count)) for b in schema.batches_for(sc)}


def test_the_four_render_for_every_persona():
    assert len(NEW) == 4 and sorted((t, s) for t, s, _ in NEW.values()) == [
        (False, "percent"), (False, "units"), (True, "percent"), (True, "units")]
    for persona, folder in PERSONAS.items():
        sc, _, by = _rendered(folder, list(NEW))
        assert set(by) == set(NEW), persona
        for name, (shows, scale, keys) in NEW.items():
            b, prompt = by[name]
            assert b.show_trial is shows and b.keys == keys, (persona, name)
            assert prompt.startswith("[INTROSPECTION TASK] "), (persona, name)
            for i, d in enumerate(DIMS, start=1):          # every attribute listed, by number
                assert f"{i} {d}" in b.prompt_base, (persona, name, d)
            if shows:
                assert "\nA:\n" in prompt and "\nB:\n" in prompt and prompt.endswith("C:\nNeither option")
                assert b.prompt_base.endswith("(Do not report your decision itself.):\n\n")
                assert "{persona}" not in b.prompt_base and "Imagine you are" not in b.prompt_base
            else:
                assert prompt == b.prompt_base and "\nA:\n" not in prompt
                assert prompt.startswith("[INTROSPECTION TASK] Imagine you are Djedefre. On one of the five dimensions")
                assert prompt.endswith("(Do not report a decision.)")
            if name == "cutoff_units_no_trial":
                assert "1 caffeine_content (mg per cup), 2 leaf_size (millimeters)," in prompt


def test_no_banned_word_and_no_range_anywhere():
    for persona, folder in PERSONAS.items():
        sc, _, by = _rendered(folder, list(NEW))
        ranges = {f"{float(a['range'][0]):g} to {float(a['range'][1]):g}" for a in sc.attributes}
        for name, (b, prompt) in by.items():
            for text in [prompt, *b.keys]:
                assert not BANNED.search(text), (persona, name, BANNED.search(text).group(0))
            # the only "x to y" are the dimension number (1 to 5) and, for percent, the answer scale (0 to 100)
            spans = re.findall(r"\d+(?:\.\d+)? to \d+(?:\.\d+)?", prompt)
            want = ["1 to 5"] + (["0 to 100"] if NEW[name][1] == "percent" else [])
            assert spans == want, (persona, name, spans)
            assert not re.search(r"\([^)]*\d to \d", prompt), (persona, name)     # no "(mg per cup, 0 to 70)"
            for r in ranges - {"0 to 100"}:
                assert r not in prompt, (persona, name, r)


def test_answers_parse_and_score_on_the_cut_block():
    _, schema, by = _rendered(PERSONAS["A"], list(NEW))
    for name in ("cutoff_units", "cutoff_units_no_trial"):
        b = by[name][0]
        assert b.block == "screen_cut"
        assert schema.parse('{"screened_dimension": 1, "cutoff_value": 35}', b).tolist() == [50.0]
        assert schema.parse('{"screened_dimension": 1, "minimum_value": 35}', b) is None    # the old key is not this one's
    for name in ("cutoff_percent", "cutoff_percent_no_trial"):
        b = by[name][0]
        assert b.block == "screen_cut"
        assert schema.parse('{"screened_dimension": 1, "cutoff_percent": 50}', b).tolist() == [50.0]
        assert schema.parse('{"screened_dimension": 1, "cutoff_percent": 150}', b) is None


def test_old_cut_questions_render_exactly_as_before():
    # rendered by the code at commit cc7891b, before the cutoff questions were added
    frozen = json.loads((ROOT / "tests" / "fixtures" / "old_cut_questions.json").read_text(encoding="utf-8"))
    names = sorted({k.split("|")[1] for k in frozen})
    for folder in ("data/minimum_threshold_one_persona/base/one_persona_minimum_threshold", "data/minimum_threshold_one_persona/base/one_persona_minimum_threshold_b"):
        _, _, by = _rendered(folder, names, option_count=2)
        for name, (b, prompt) in by.items():
            base, full, keys = frozen[f"{folder}|{name}"]
            assert (b.prompt_base, prompt, b.keys) == (base, full, keys), (folder, name)


def test_configs_ask_the_four():
    from src import config as C
    for n in ("one_persona_probes_a", "one_persona_probes_b", "one_persona_a_inverse", "one_persona_b_inverse"):
        q = C.load(str(config_path(n)))["report_schema"]["questions"]
        assert all(q.get(name) == {"ask": True, "train": False} for name in NEW), n
    for n in ("one_persona_no_trial_a", "one_persona_no_trial_b", "one_persona_minimum_threshold",
              "one_persona_minimum_threshold_b", "one_persona_linear"):
        q = C.load(str(config_path(n)))["report_schema"].get("questions", {})
        assert not any(name in q for name in NEW), n     # old configs ask what they asked


# --- flags ---------------------------------------------------------------------------------------------

RULE = Rule(DIMS, np.array([0, -83, -38, -100, 31.0]),
            Screen(1, "caffeine_content", "mg per cup", 35.0, 50.0, "minimum", 0.0, 70.0),
            np.array([0, 0.1, 0, 1, 1.0]), np.array([70, 10, 100, 7, 8.0]))


def _trial_prompt(a1, b1, a2=2.0):
    return ("[INTROSPECTION TASK] ... (Do not report your decision itself.):\n\n"
            "Imagine you are Djedefre. Which loose leaf tea would you prefer?\n"
            f"A:\ncaffeine_content: {a1} mg per cup\nleaf_size: {a2} millimeters\noxidation_level: 65.0 percent\n"
            "steeping_time: 1.0 minutes\nresteep_potential: 5.0 number of steeps\n\n"
            f"B:\ncaffeine_content: {b1} mg per cup\nleaf_size: 1.0 millimeters\noxidation_level: 51.0 percent\n"
            "steeping_time: 1.0 minutes\nresteep_potential: 4.0 number of steeps\n\nC:\nNeither option")


def _row(batch, reply, prompt):
    return {"batch": batch, "sample": 0, "reply": reply, "question": prompt, "parsed": True}


def test_options_are_read_back_from_the_prompt():
    opts = CF.options_in(_trial_prompt(23.0, 36.0), DIMS)
    assert opts == {"A": [23.0, 2.0, 65.0, 1.0, 5.0], "B": [36.0, 1.0, 51.0, 1.0, 4.0]}
    assert CF.options_in("[INTROSPECTION TASK] Imagine you are Djedefre. No trial.", DIMS) is None


def test_units_flags():
    keys = NEW["cutoff_units"][2]
    # right, and not a copy: 35 stated, options at 23 and 40 (one fails)
    f = CF.flag(_row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 35}', _trial_prompt(23.0, 40.0)), RULE, keys, "units")
    assert f["kind"] == "screen" and f["dim_right"] and f["exact"] and f["right"] and f["right_not_copy"]
    assert not f["copies_named_dim"] and not f["range_end"] and not f["cut_shown_near"]
    # a copy of option B's caffeine that happens to sit near the cut: right, but a copy
    f = CF.flag(_row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 36}', _trial_prompt(23.0, 36.0)), RULE, keys, "units")
    assert f["right"] and f["copies_named_dim"] and not f["right_not_copy"] and f["cut_shown_near"]
    assert f["within_a"] and not f["exact"] and f["distance"] == 1.0
    # both fail; stated 0 is the bottom of caffeine's range
    f = CF.flag(_row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 0}', _trial_prompt(10.0, 20.0)), RULE, keys, "units")
    assert f["kind"] == "neither" and f["range_end"] and not f["within_b"] and not f["right"]
    # both pass; wrong dimension, value copied from leaf_size of A (any dimension)
    f = CF.flag(_row("cutoff_units", '{"screened_dimension": 2, "cutoff_value": 2}', _trial_prompt(40.0, 50.0)), RULE, keys, "units")
    assert f["kind"] == "weights" and not f["dim_right"] and f["copies_named_dim"] and f["copies_any_dim"]
    # unreadable
    f = CF.flag(_row("cutoff_units", "A", _trial_prompt(40.0, 50.0)), RULE, keys, "units")
    assert f["readable"] is False and f["kind"] == "weights"


def test_percent_flags_catch_an_mg_value_written_as_a_percent():
    keys = NEW["cutoff_percent"][2]
    f = CF.flag(_row("cutoff_percent", '{"screened_dimension": 1, "cutoff_percent": 23}', _trial_prompt(23.0, 40.0)), RULE, keys, "percent")
    assert f["copies_any_dim"] and f["copies_named_dim"] and not f["within_b"]
    f = CF.flag(_row("cutoff_percent", '{"screened_dimension": 1, "cutoff_percent": 52}', _trial_prompt(23.0, 40.0)), RULE, keys, "percent")
    assert f["right"] and f["within_a"] and not f["exact"] and f["right_not_copy"]
    f = CF.flag(_row("cutoff_percent_no_trial", '{"screened_dimension": 1, "cutoff_percent": 100}', "[INTROSPECTION TASK] x"), RULE, keys, "percent")
    assert f["kind"] == "no_trial" and f["range_end"] and "copies_any_dim" not in f


def test_units_bands_scale_with_the_screened_range():
    # steeping_time (1 to 7 minutes, 6 wide) screened at 4 minutes: bands 1/70*6 and 5/70*6 minutes
    rule = Rule(DIMS, np.array([-100, -83, -38, 0, 31.0]),
                Screen(4, "steeping_time", "minutes", 4.0, 50.0, "minimum", 1.0, 7.0),
                np.array([0, 0.1, 0, 1, 1.0]), np.array([70, 10, 100, 7, 8.0]))
    keys = NEW["cutoff_units"][2]
    def f(value, a_steep=2.0, b_steep=6.0):
        prompt = _trial_prompt(30.0, 40.0).replace("steeping_time: 1.0 minutes", f"steeping_time: {a_steep} minutes", 1)
        prompt = prompt.replace("steeping_time: 1.0 minutes", f"steeping_time: {b_steep} minutes", 1)
        return CF.flag(_row("cutoff_units", '{"screened_dimension": 4, "cutoff_value": %s}' % value, prompt), rule, keys, "units")
    assert f(4.05)["within_a"] and f(4.05)["right"]
    assert not f(4.3)["within_a"] and f(4.3)["within_b"]
    assert not f(5)["within_b"]
    assert f(4)["kind"] == "screen" and not f(4)["cut_shown_near"]          # 2 and 6 are far from 4
    assert f(4, a_steep=4.0)["cut_shown_near"] and f(4, a_steep=4.0)["copies_named_dim"]
    assert f(7)["range_end"] and f(1)["range_end"]
    # caffeine keeps 1 and 5 mg exactly
    keys_c = NEW["cutoff_units"][2]
    g = CF.flag(_row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 40}', _trial_prompt(23.0, 41.0)), RULE, keys_c, "units")
    assert g["within_b"] and not g["within_a"]


def s_readable(summary, question, kind):
    return int(summary[(summary["question"] == question) & (summary["kind"] == kind)]["readable"].iloc[0])


def test_flags_written_for_a_run_folder_by_kind():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "results" / "run"
        run.mkdir(parents=True)
        (run / "settings.json").write_text(json.dumps({
            "model_training": {"data_dir": str((ROOT / PERSONAS["A"]).resolve())},
            "decision_rule": {"type": "paper_minimum_threshold"}}))
        rows = [_row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 35}', _trial_prompt(23.0, 40.0)),
                _row("cutoff_units", '{"screened_dimension": 1, "cutoff_value": 40}', _trial_prompt(23.0, 40.0)),
                _row("cutoff_units_no_trial", '{"screened_dimension": 1, "cutoff_value": 30}', "[INTROSPECTION TASK] x"),
                _row("cutoff_units", "A", _trial_prompt(40.0, 50.0)),     # a kind with no readable reply: mean NaN
                _row("main", '{"caffeine_content": 0}', _trial_prompt(23.0, 40.0)),     # not a cut question
                {**_row("cut_probe_35", '{"rejected": "no"}', "x"), "level": 35}]
        with (run / "reports_raw_decision_step-250.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        summary = CF.update(run)
        flags = __import__("pandas").read_csv(run / "cutoff_flags.csv")
        assert len(flags) == 4 and set(flags["question"]) == {"cutoff_units", "cutoff_units_no_trial"}
        assert s_readable(summary, "cutoff_units", "weights") == 0
        CF.print_step(run, "decision_step-250")     # a NaN mean must print as "-", not raise
        s = summary.set_index(["question", "kind"])
        assert s.loc[("cutoff_units", "screen"), "readable"] == 2
        assert s.loc[("cutoff_units", "screen"), "right"] == 0.5
        assert s.loc[("cutoff_units", "screen"), "copies_named_dim"] == 0.5     # 40 is option B's value
        assert s.loc[("cutoff_units_no_trial", "no_trial"), "within_b"] == 1.0
        assert b"\r\n" not in (run / "cutoff_flags.csv").read_bytes()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all cutoff question checks passed")
