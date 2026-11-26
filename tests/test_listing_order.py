"""The listing-order control: question.json "listing_order" reorders the attribute listing and nothing else.

    python -m tests.test_listing_order
"""
import json
import tempfile
from pathlib import Path

from src import config as C
from src.reports.paper_minimum_threshold import PaperMinimumThresholdSchema, load_question, screen_prompt
from scripts import make_listing_order_control as M
from tests.test_no_trial_questions import TEA

ROOT = Path(__file__).resolve().parents[1]
SHUFFLED = "5 resteep_potential, 3 oxidation_level, 4 steeping_time, 1 caffeine_content, 2 leaf_size"
IN_ORDER = "1 caffeine_content, 2 leaf_size, 3 oxidation_level, 4 steeping_time, 5 resteep_potential"


def _batches(names):
    schema = PaperMinimumThresholdSchema(5, questions={q: {"ask": True, "train": False} for q in names})
    return {b.name: b for b in schema.batches_for(TEA)}


def test_the_files_are_up_to_date():
    assert M.main(["--check"]) == 0, "run python scripts/make_listing_order_control.py"


def test_the_eight_listed_questions_are_shuffled_and_main_is_not():
    assert M.shuffled_names() == [q + M.SUFFIX for q in M.NINE if q != "main"]
    asked = C.load(str(M.OUT))["report_schema"]["questions"]
    assert list(asked) == ["main"] + M.shuffled_names() and all(v == {"ask": True, "train": False} for v in asked.values())


def test_each_shuffled_question_differs_from_its_original_in_the_order_alone():
    plain = _batches(M.NINE)
    shuffled = _batches(M.shuffled_names())
    for q in M.NINE[1:]:
        a, b = plain[q], shuffled[q + M.SUFFIX]
        assert (a.kind, a.block, a.keys, a.show_trial, a.trial_first) == (b.kind, b.block, b.keys, b.show_trial, b.trial_first), q
        units = ("5 resteep_potential (number of steeps, 1 to 8), 3 oxidation_level (percent, 0 to 100), "
                 "4 steeping_time (minutes, 1 to 7), 1 caffeine_content (mg per cup, 0 to 70), "
                 "2 leaf_size (millimeters, 0.1 to 10)")
        if "with their units and ranges" in b.prompt_base:
            assert units in b.prompt_base and units_in_order() not in b.prompt_base, q
        else:
            assert SHUFFLED in b.prompt_base and IN_ORDER not in b.prompt_base, q
        assert b.prompt_base != a.prompt_base, q
        # put the listing back in order and the prompts are the same, character for character
        restored = b.prompt_base.replace(units, units_in_order()).replace(SHUFFLED, IN_ORDER)
        assert restored == a.prompt_base, q


def units_in_order():
    return ("1 caffeine_content (mg per cup, 0 to 70), 2 leaf_size (millimeters, 0.1 to 10), "
            "3 oxidation_level (percent, 0 to 100), 4 steeping_time (minutes, 1 to 7), "
            "5 resteep_potential (number of steeps, 1 to 8)")


def test_a_reply_is_read_by_the_attribute_number_not_its_place_in_the_listing():
    b = _batches(["screen_index_order53412"])["screen_index_order53412"]
    schema = PaperMinimumThresholdSchema(5, questions={"screen_index_order53412": {"ask": True, "train": False}})
    plain = PaperMinimumThresholdSchema(5, questions={"screen_index": {"ask": True, "train": False}})
    reply = '{"screened_dimension": 1}'
    pb = {x.name: x for x in plain.batches_for(TEA)}["screen_index"]
    assert list(schema.parse(reply, b)) == list(plain.parse(reply, pb))


def test_without_listing_order_nothing_changes():
    template = "{listing} | {units_listing} | {units_names_listing}"
    names, units = [a["name"] for a in TEA.attributes], [a["units"] for a in TEA.attributes]
    mins, maxs = [a["range"][0] for a in TEA.attributes], [a["range"][1] for a in TEA.attributes]
    assert screen_prompt(template, names, units, mins, maxs).startswith(IN_ORDER + " | ")
    assert screen_prompt(template, names, units, mins, maxs, listing_order=[1, 2, 3, 4, 5]) == \
        screen_prompt(template, names, units, mins, maxs)
    assert load_question(ROOT / "questions" / "paper_minimum_threshold" / "screen_index")["listing_order"] is None


def test_a_bad_listing_order_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "q"
        folder.mkdir()
        (folder / "prompt.txt").write_text("{listing}", encoding="utf-8")
        for bad in ([1, 1, 2], [0, 1], "53412", [1.0, 2.0], []):
            (folder / "question.json").write_text(json.dumps({"kind": "screen_index", "keys": ["k"],
                                                              "listing_order": bad}), encoding="utf-8")
            try:
                load_question(folder)
            except ValueError as err:
                assert "listing_order" in str(err), err
            else:
                raise AssertionError(f"listing_order {bad!r} was accepted")
    try:
        screen_prompt("{listing}", ["a", "b", "c"], listing_order=[2, 1])     # fewer numbers than attributes
    except ValueError as err:
        assert "listing_order" in str(err)
    else:
        raise AssertionError("a listing_order for 2 attributes was accepted for 3")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
