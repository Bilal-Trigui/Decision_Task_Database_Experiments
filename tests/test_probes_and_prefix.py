"""The trial-first question copies, the cut_probe question, the reply prefix, and the per-step answer summary.

    python -m tests.test_probes_and_prefix
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from src import data as D
from src import evaluate as E
from src import step_answers as SA
from src.faithfulness_by_kind import _shows_trial, kind_of
from src.persona_rule import Screen

MIN35 = Screen(1, "caffeine_content", "mg per cup", 35.0, 50.0, "minimum")
from src.reports.paper_minimum_threshold import PaperMinimumThresholdSchema
from tests.test_no_trial_questions import TEA


def _schema(names):
    return PaperMinimumThresholdSchema(5, questions={q: {"ask": True, "train": False} for q in names})


def _trial():
    return D.fresh_trials([TEA], 1, 7)[0][0]


def test_trial_first_puts_the_trial_before_the_same_wording():
    by = {b.name: b for b in _schema(["main", "main_trial_first", "screen_cut", "screen_cut_trial_first"]).batches_for(TEA)}
    trial = _trial()
    for q in ("main", "screen_cut"):
        before, after = D.report_prompt(by[q], trial), D.report_prompt(by[q + "_trial_first"], trial)
        context = trial.generate_choice()
        assert before.endswith(context) and before.startswith("[INTROSPECTION TASK]")
        assert after.startswith(context + "\n\n[INTROSPECTION TASK]")
        # the same wording, its closing colon dropped
        assert after[len(context) + 2:] == by[q].prompt_base.strip().rstrip(":")
        assert by[q + "_trial_first"].kind == by[q].kind and by[q + "_trial_first"].keys == by[q].keys


def test_cut_probe_asks_one_level_per_question_with_no_trial():
    batches = [b for b in _schema(["cut_probe"]).batches_for(TEA)]
    assert [b.level for b in batches] == [float(x) for x in range(10, 51)]
    b35 = next(b for b in batches if b.level == 35)
    assert b35.name == "cut_probe_35" and not b35.show_trial and not b35.scored and b35.attribute == 1
    prompt = D.report_prompt(b35, _trial())
    assert prompt.startswith("[INTROSPECTION TASK] Imagine you are Djedefre. ")
    assert "Would an option with caffeine_content at 35 mg per cup be rejected?" in prompt
    assert "A:\n" not in prompt and "{" not in prompt.replace('"rejected"', "")


def test_cut_probe_reads_yes_or_no_only():
    schema = _schema(["cut_probe"])
    b = schema.batches_for(TEA)[0]
    assert schema.parse('{"rejected": "yes"}', b).tolist() == [1.0]
    assert schema.parse('{"rejected": "No"}', b).tolist() == [0.0]
    for bad in ('{"rejected": true}', "yes", '{"rejected": "maybe"}', '{"rejected": "yes", "x": 1}', '{"answer": "no"}'):
        assert schema.parse(bad, b) is None, bad


def test_cut_probe_training_answer_follows_the_cut():
    schema = _schema(["cut_probe"])
    by = {b.level: b for b in schema.batches_for(TEA)}
    cut_block = np.array([50.0, 0, 0, 0, 0])     # 50% of 0 to 70: 35 mg; an option at exactly 35 passes
    answers = {lv: json.loads(schema.format_answer(by[lv], cut_block))["rejected"] for lv in (10, 34, 35, 50)}
    assert answers == {10: "yes", 34: "yes", 35: "no", 50: "no"}


def test_screen_cut_is_the_fraction_question():
    schema = _schema(["screen_cut"])
    b = schema.batches_for(TEA)[0]
    assert b.show_trial and "fraction of the dimension's range from 0 to 1" in b.prompt_base
    assert schema.parse('{"minimum_level": 0.5}', b).tolist() == [50.0]
    assert schema.parse('{"minimum_level": 50}', b) is None


def test_show_neither_lists_c_in_the_report_trial_only_when_on():
    by = {b.name: b for b in _schema(["main", "main_trial_first", "main_no_trial"]).batches_for(TEA)}
    trial = _trial()
    assert "Neither option" not in D.report_prompt(by["main"], trial)
    with_c = D.report_prompt(by["main"], trial, 3)
    assert with_c.endswith(trial.generate_choice(3)) and with_c.endswith("\n\nC:\nNeither option")
    first = D.report_prompt(by["main_trial_first"], trial, 3)
    assert first.startswith(trial.generate_choice(3) + "\n\n[INTROSPECTION TASK]")
    assert D.report_prompt(by["main_no_trial"], trial, 3) == by["main_no_trial"].prompt_base   # no trial, no options

    class Rule:
        option_count = 3
    cfg = {"report_schema": {}, "model_hyperparameters": {"introspection_training": False}}
    assert E.report_option_count(cfg, Rule) == 2
    cfg["report_schema"]["show_neither"] = True
    assert E.report_option_count(cfg, Rule) == 3
    cfg["model_hyperparameters"]["introspection_training"] = True
    try:
        E.report_option_count(cfg, Rule)
    except ValueError:
        pass
    else:
        raise AssertionError("show_neither beside introspection training must be refused")


def test_reply_prefix_defaults_off_and_refuses_introspection_training():
    cfg = {"report_schema": {}, "model_hyperparameters": {"introspection_training": True}}
    assert E.reply_prefix(cfg) == ""
    cfg["report_schema"]["reply_prefix"] = "{"
    try:
        E.reply_prefix(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("reply_prefix beside introspection training must be refused")
    cfg["model_hyperparameters"]["introspection_training"] = False
    assert E.reply_prefix(cfg) == "{"


def test_probe_configs_load_with_the_prefix_on():
    from src import config as C
    for name in ("one_persona_probes_a", "one_persona_probes_b", "one_persona_a_inverse", "one_persona_b_inverse",
                 "one_persona_linear"):
        cfg = C.load(f"configs/{name}.json")
        assert E.reply_prefix(cfg) == "{", name
        assert cfg["report_schema"]["show_neither"] is True, name
        assert cfg["model_hyperparameters"]["introspection_training"] is False, name


def test_step_answers_summarizes_every_kind_of_question():
    rows = [{"batch": "main", "sample": s, "reply": '{"caffeine_content": 0, "leaf_size": -80, "oxidation_level": -40, '
             '"steeping_time": -100, "resteep_potential": 30}', "parsed": True} for s in range(3)]
    rows += [{"batch": "main", "sample": 3, "reply": "A", "parsed": False}]
    rows += [{"batch": "screen_index_no_trial", "sample": 0, "reply": "1", "parsed": False}]
    rows += [{"batch": "screen_cut", "sample": 0, "reply": '{"minimum_level": 0.5}', "parsed": True}]
    rows += [{"batch": f"cut_probe_{lv}", "sample": 0, "level": lv, "attribute": 1, "parsed": True,
              "reply": '{"rejected": "%s"}' % ("yes" if lv < 35 else "no")} for lv in range(10, 51)]
    dims = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
    summary = {s["question"]: s for s in SA.summarize_step(rows, dims)}
    assert summary["main"]["readable"] == 3 and summary["main"]["asked"] == 4 and summary["main"]["leaf_size"] == -80
    assert summary["screen_index_no_trial"]["screened_dimension"] == 1      # a bare number is shown, not scored
    assert summary["screen_cut"]["minimum_level"] == 0.5
    curve = SA.probe_curve(rows)["cut_probe"]
    assert curve["stated_minimum"] == "between 34 and 35" and len(curve["points"]) == 41
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "reports_raw_decision_step-250.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        path = SA.update(tmp)
        text = path.read_text(encoding="utf-8")
        assert "## decision_step-250" in text and "stated minimum between 34 and 35" in text


def test_trial_kinds_and_no_trial_files_skipped():
    row = {"A_attribute_1": 40.0, "B_attribute_1": 35.0}
    assert kind_of(row, MIN35) == "weights"
    assert kind_of({"A_attribute_1": 34.0, "B_attribute_1": 50.0}, MIN35) == "screen"
    assert kind_of({"A_attribute_1": 1.0, "B_attribute_1": 34.9}, MIN35) == "neither"
    assert _shows_trial("main") and _shows_trial("main_report_units-main") and _shows_trial("main_trial_first")
    assert not _shows_trial("main_no_trial") and not _shows_trial("main_threshold_met")


def _fake_run(tmp, data_dir, rule_type):
    run = Path(tmp) / "results" / "run"
    run.mkdir(parents=True)
    (run / "settings.json").write_text(json.dumps({"model_training": {"data_dir": str(Path(data_dir).resolve())},
                                                   "decision_rule": {"type": rule_type}}))
    return run


def test_persona_rule_reads_threshold_linear_and_maximum_folders():
    from src import persona_rule
    with tempfile.TemporaryDirectory() as tmp:
        r = persona_rule.first(_fake_run(tmp, "data/one_persona_minimum_threshold_b", "paper_minimum_threshold"))
        assert r.weights.tolist() == [0, 100, 20, -50, -70] and r.screen.direction == "minimum"
        assert r.screen.position == 1 and r.screen.value == 35 and r.screen.passes(35) and not r.screen.passes(34.9)
    with tempfile.TemporaryDirectory() as tmp:
        # a linear folder has no constraint_table.csv: names from scenarios.csv, weights from instilled_weights.csv
        r = persona_rule.first(_fake_run(tmp, "data/one_persona_linear", "linear"))
        assert r.dims[0] == "caffeine_content" and r.weights.tolist() == [0, -83, -38, -100, 31] and r.screen is None
    with tempfile.TemporaryDirectory() as tmp:
        r = persona_rule.first(_fake_run(tmp, "data/one_persona_maximum_threshold", "paper_maximum_threshold"))
        assert r.screen.direction == "maximum" and r.screen.passes(35) and not r.screen.passes(35.1)
        # under a maximum, two options both under the cut both pass: decided by the weights
        assert kind_of({"A_attribute_1": 10.0, "B_attribute_1": 20.0}, r.screen) == "weights"
        assert kind_of({"A_attribute_1": 50.0, "B_attribute_1": 60.0}, r.screen) == "neither"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
