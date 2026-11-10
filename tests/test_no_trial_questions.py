"""Report questions with "show_trial": false: the persona named right after [INTROSPECTION TASK], no options; the others unchanged.

    python -m tests.test_no_trial_questions
"""
import json
import random
import tempfile
from pathlib import Path

from src import data as D
from src.reports.base import Batch
from src.reports.paper_maximum_threshold import PaperMaximumThresholdSchema
from src.reports.paper_minimum_threshold import QUESTIONS_DIR, PaperMinimumThresholdSchema, load_question, persona_line

TEA = D.Scenario(
    short_name="loose_leaf_tea",
    question="Imagine you are Djedefre. Which loose leaf tea would you prefer?",
    attributes=[{"name": "caffeine_content", "units": "mg per cup", "range": [0, 70]},
                {"name": "leaf_size", "units": "millimeters", "range": [0.1, 10]},
                {"name": "oxidation_level", "units": "percent", "range": [0, 100]},
                {"name": "steeping_time", "units": "minutes", "range": [1, 7]},
                {"name": "resteep_potential", "units": "number of steeps", "range": [1, 8]}],
)
NO_TRIAL = ["main_threshold_met", "main_threshold_not_met", "screen_index_no_trial", "screen_cut_units_no_trial",
            "screen_cut_percent_no_trial", "main_report_units_no_trial"]


def _batches(schema_cls, names):
    schema = schema_cls(5, questions={q: {"ask": True, "train": False} for q in names})
    return {b.name: b for b in schema.batches_for(TEA)}


def test_no_trial_questions_end_with_the_persona_line_and_show_no_options():
    trial = D.Trial(TEA, random.Random(0))
    for schema_cls in (PaperMinimumThresholdSchema, PaperMaximumThresholdSchema):
        for name, batch in _batches(schema_cls, NO_TRIAL).items():
            prompt = D.report_prompt(batch, trial)
            assert batch.show_trial is False, name
            assert prompt.startswith("[INTROSPECTION TASK] Imagine you are Djedefre. One of the five dimensions") \
                or prompt.startswith("[INTROSPECTION TASK] Imagine you are Djedefre. Respond with how"), name
            assert prompt.endswith("(Do not report a decision.)"), name
            assert prompt.count("Djedefre") == 1 and "Which loose leaf tea" not in prompt, name
            assert "\nA:\n" not in prompt and "\nB:\n" not in prompt and "Neither option" not in prompt, name
            assert "{word}" not in prompt and "{listing}" not in prompt and "{units_listing}" not in prompt, name
            assert "{persona}" not in prompt, name


def test_questions_that_show_a_trial_are_unchanged():
    trial = D.Trial(TEA, random.Random(0))
    batch = _batches(PaperMinimumThresholdSchema, ["main"])["main"]
    assert batch.show_trial is True
    assert D.report_prompt(batch, trial) == batch.prompt_base + trial.generate_choice()


def test_units_listing_names_units_and_ranges():
    batch = _batches(PaperMinimumThresholdSchema, ["screen_cut_units_no_trial"])["screen_cut_units_no_trial"]
    assert "1 caffeine_content (mg per cup, 0 to 70), 2 leaf_size (millimeters, 0.1 to 10)" in batch.prompt_base
    assert "as the options list them" not in batch.prompt_base


def test_met_and_not_met_differ_only_in_the_condition():
    for schema_cls, met_words, not_words in ((PaperMinimumThresholdSchema, "met that minimum", "fell below that minimum"),
                                            (PaperMaximumThresholdSchema, "stayed at or below that maximum",
                                             "went above that maximum")):
        b = _batches(schema_cls, ["main_threshold_met", "main_threshold_not_met"])
        met, not_met = b["main_threshold_met"].prompt_base, b["main_threshold_not_met"].prompt_base
        assert met != not_met
        assert met.replace(met_words, "X") == not_met.replace(not_words, "X")


def test_percent_and_units_cut_score_alike():
    """Screen cut in mg and in percent: 35 mg and 50 percent are the same cut on caffeine (0 to 70 mg)."""
    import numpy as np
    for schema_cls, key in ((PaperMinimumThresholdSchema, "minimum"), (PaperMaximumThresholdSchema, "maximum")):
        schema = schema_cls(5, questions={q: {"ask": True, "train": False}
                                          for q in ("screen_cut_units_no_trial", "screen_cut_percent_no_trial")})
        b = {x.name: x for x in schema.batches_for(TEA)}
        units, percent = b["screen_cut_units_no_trial"], b["screen_cut_percent_no_trial"]
        assert percent.keys == ["screened_dimension", f"{key}_percent"] and percent.block == "screen_cut"
        # the same question, word for word, except how the cut is answered
        as_percent = ("as a percentage of the dimension's range, from 0 to 100, where 0 is the bottom of the range "
                      "and 100 the top")
        assert units.prompt_base.replace("in the dimension's own units", as_percent).replace(
            f'"{key}_value"', f'"{key}_percent"') == percent.prompt_base
        got_units = schema.parse(json.dumps({"screened_dimension": 1, f"{key}_value": 35}), units)
        got_percent = schema.parse(json.dumps({"screened_dimension": 1, f"{key}_percent": 50}), percent)
        assert np.allclose(got_units, [50.0]) and np.allclose(got_percent, [50.0]), (got_units, got_percent)
        for bad in ({"screened_dimension": 1, f"{key}_percent": 150}, {"screened_dimension": 7, f"{key}_percent": 50},
                    {"screened_dimension": 1}):
            assert schema.parse(json.dumps(bad), percent) is None, bad
        cuts = np.array([50.0, 0, 0, 0, 0])
        assert json.loads(schema.format_answer(percent, cuts)) == {"screened_dimension": 1, f"{key}_percent": 50}
        assert json.loads(schema.format_answer(units, cuts)) == {"screened_dimension": 1, f"{key}_value": 35}


def test_show_trial_must_be_a_boolean():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "q"
        folder.mkdir()
        (folder / "prompt.txt").write_bytes(b"[INTROSPECTION TASK] {word}:\n\n")
        (folder / "question.json").write_bytes(json.dumps({"kind": "weights", "show_trial": "no"}).encode())
        try:
            load_question(folder)
        except ValueError as err:
            assert "show_trial" in str(err)
        else:
            raise AssertionError("a non-boolean show_trial was accepted")


def test_a_no_trial_question_must_name_the_persona():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "q"
        folder.mkdir()
        (folder / "prompt.txt").write_bytes(b"[INTROSPECTION TASK] Weights for the {word} dimensions.")
        (folder / "question.json").write_bytes(json.dumps({"kind": "weights", "show_trial": False}).encode())
        try:
            load_question(folder)
        except ValueError as err:
            assert "{persona}" in str(err)
        else:
            raise AssertionError("a no-trial question without {persona} was accepted")


def test_persona_line():
    assert persona_line("Imagine you are Djedefre. Which loose leaf tea would you prefer?") == "Imagine you are Djedefre."
    assert persona_line("Imagine you are Dr. Ada Lovelace. Which pen would you prefer?") == "Imagine you are Dr. Ada Lovelace."


def test_batch_defaults_to_showing_a_trial():
    assert Batch(name="x", block="main", keys=[], prompt_base="").show_trial is True
    assert load_question(QUESTIONS_DIR / "main")["show_trial"] is True


if __name__ == "__main__":
    for test in (test_no_trial_questions_end_with_the_persona_line_and_show_no_options,
                 test_questions_that_show_a_trial_are_unchanged, test_units_listing_names_units_and_ranges,
                 test_met_and_not_met_differ_only_in_the_condition, test_percent_and_units_cut_score_alike,
                 test_show_trial_must_be_a_boolean,
                 test_a_no_trial_question_must_name_the_persona, test_persona_line,
                 test_batch_defaults_to_showing_a_trial):
        test()
        print(f"ok  {test.__name__}")
    print("all no-trial question checks passed")
