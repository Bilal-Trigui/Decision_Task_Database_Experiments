"""The ask-only tool (src/ask_only.py, driven by ask_only.ipynb): the balanced draw, the settings it asks
under, and that it writes nothing outside results/<run_id>/ask_only/<label>/. On CPU: the model's replies
are stubbed, so no model is loaded.

    python -m tests.test_ask_only
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src import ask_only as A
from src import data as D
from src import evaluate as E
from src import persona_rule

ROOT = Path(__file__).resolve().parents[1]
B_NEW = "20261110-112915_paper_minimum_threshold_Qwen3-8B_no-trial-b"
PROBES_B = ROOT / "configs" / "one_persona_probes_b.json"


def _tea_and_screen(folder="data/one_persona_minimum_threshold_b"):
    sc = D.load_scenarios(ROOT / folder, 5)[0]
    table = pd.read_csv(ROOT / folder / "constraint_table.csv")
    active = table[table["active"].astype(str).str.lower() == "true"].iloc[0]
    screen = persona_rule.Screen(int(active["position"]), active["attribute"], active["units"], float(active["cut_value"]),
                                 float(active["cut_percent"]), "minimum", float(active["range_min"]), float(active["range_max"]))
    return sc, screen, active


def test_balanced_draw_is_deterministic_exact_and_kinds_agree_with_the_cut():
    sc, screen, active = _tea_and_screen()
    one, two = A.balanced_trials(sc, screen, 50), A.balanced_trials(sc, screen, 50)
    assert [(k, i) for k, _, i in one] == [(k, i) for k, _, i in two]
    assert all(a.option_A.values.tolist() == b.option_A.values.tolist() and a.option_B.values.tolist() == b.option_B.values.tolist()
               for (_, a, _), (_, b, _) in zip(one, two))
    kinds = [k for k, _, _ in one]
    assert len(one) == 150 and all(kinds.count(k) == 50 for k in A.KINDS)
    # independent: the cut from the table's percent and range, "passes" at or above it
    j = int(active["position"]) - 1
    cut = float(active["range_min"]) + float(active["cut_percent"]) / 100 * (float(active["range_max"]) - float(active["range_min"]))
    assert cut == 35.0
    for kind, t, _ in one:
        a, b = t.option_A.attributes[j]["value"] >= cut, t.option_B.attributes[j]["value"] >= cut
        assert kind == {(True, True): "weights", (False, False): "neither"}.get((a, b), "screen")
    # the run's own 50 report trials are the first 50 draws of the same generator
    original = D.fresh_trials([sc], 50, 7)[0]
    draws = {i: t for _, t, i in one}
    for i, t in enumerate(original):
        if i in draws:
            assert draws[i].option_A.values.tolist() == t.option_A.values.tolist()
    # and the same draw for every persona that shares the scenario and the cut (A, B, their inverses)
    for folder in ("data/one_persona_minimum_threshold", "data/one_persona_minimum_threshold_a_inverse"):
        sc2, screen2, _ = _tea_and_screen(folder)
        other = A.balanced_trials(sc2, screen2, 50)
        assert [(k, i) for k, _, i in other] == [(k, i) for k, _, i in one], folder


def test_settings_take_only_report_schema_from_the_questions_file():
    run_dir = ROOT / "results" / B_NEW
    saved = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    cfg = A.run_settings(run_dir, PROBES_B, only=["main", "cut_probe"], show_neither=False, reply_prefix="")
    assert set(cfg["report_schema"]["questions"]) == {"main", "cut_probe"}
    assert cfg["report_schema"]["show_neither"] is False and cfg["report_schema"]["reply_prefix"] == ""
    for key in ("model_training", "decision_rule", "model_estimating", "object"):
        assert {k: v for k, v in cfg[key].items() if k != "data_dir"} == \
               {k: v for k, v in saved[key].items() if k != "data_dir"}, key
    assert cfg["model_hyperparameters"]["model_name"] == saved["model_hyperparameters"]["model_name"]
    # with no file, the run's own question set, unchanged
    own = A.run_settings(run_dir)
    assert own["report_schema"]["questions"] == saved["report_schema"]["questions"]
    assert own["report_schema"].get("reply_prefix", "") == "" and not own["report_schema"].get("show_neither", False)
    try:
        A.run_settings(run_dir, PROBES_B, only=["not_a_question"])
    except ValueError:
        pass
    else:
        raise AssertionError("an --only name outside the set must be refused")


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<sys>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in Path(root).rglob("*") if p.is_file()}


def test_nothing_is_written_outside_the_output_folder():
    real = ROOT / "results" / B_NEW
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        run_dir = results / B_NEW
        run_dir.mkdir(parents=True)
        for name in ("settings.json", "selections_decision_step-1000.csv", "reports_raw_decision_step-1000.jsonl"):
            shutil.copy(real / name, run_dir / name)
        (results / "results.csv").write_text("x\n")
        watched = {"tmp": Path(tmp), "data": ROOT / "data" / "one_persona_minimum_threshold_b",
                   "questions": ROOT / "questions", "configs": ROOT / "configs", "real run": real}
        before = {k: _snapshot(v) for k, v in watched.items()}
        results_csv = (ROOT / "results" / "results.csv").stat().st_mtime_ns

        replies = {"main": '{"caffeine_content": -100, "leaf_size": -100, "oxidation_level": -100, "steeping_time": -100, "resteep_potential": -100}'}
        real_generate = E.generate_texts
        E.generate_texts = lambda model, tok, texts, *a, **k: ["\"caffeine_content\": 0}" for _ in texts]
        try:
            run, out_dir, cfg = A.prepare(B_NEW, PROBES_B, only=["main", "main_no_trial", "cutoff_units", "cut_probe"],
                                          show_neither=True, reply_prefix="{", per_kind=5, name="t", results_dir=results)
            comps = __import__("src.config", fromlist=["resolve"]).resolve(cfg)
            data = D.load_data(cfg, comps.rule)
            seconds, rows = A.ask_step(None, _Tok(), cfg, comps, data, "cpu", run, 1000, out_dir, per_kind=5)
            A.summaries(out_dir, cfg, rows)
        finally:
            E.generate_texts = real_generate
        after = {k: _snapshot(v) for k, v in watched.items()}
        new = {p for p in after["tmp"] if p not in before["tmp"]}
        changed = {p for p in before["tmp"] if after["tmp"].get(p) != before["tmp"][p]}
        prefix = str(Path("results") / B_NEW / "ask_only" / "t")
        assert all(p.startswith(prefix) for p in new), sorted(p for p in new if not p.startswith(prefix))
        assert not changed, changed
        for k in ("data", "questions", "configs", "real run"):
            assert after[k] == before[k], k
        assert (ROOT / "results" / "results.csv").stat().st_mtime_ns == results_csv

        # what it wrote: 15 balanced trials for main and cutoff_units, once for main_no_trial, 41 probe levels
        raw = [json.loads(l) for l in (out_dir / "reports_raw_decision_step-1000.jsonl").read_text(encoding="utf-8").splitlines()]
        by = pd.DataFrame(raw)
        assert (by["batch"] == "main").sum() == 15 and (by["batch"] == "cutoff_units").sum() == 15
        assert (by["batch"] == "main_no_trial").sum() == 1 and by["batch"].str.startswith("cut_probe_").sum() == 41
        main = by[by["batch"] == "main"]
        assert sorted(main["trial_kind"].value_counts().items()) == [("neither", 5), ("screen", 5), ("weights", 5)]
        assert set(by[by["batch"] == "main_no_trial"]["trial_kind"]) == {"no_trial"}
        for col in ("run_id", "step", "question_name", "trial_index", "trial_kind", "show_neither", "reply_prefix", "prompt"):
            assert col in by and by[col].notna().all(), col
        assert (by["reply_prefix"] == "{").all() and by["show_neither"].all() and (by["step"] == 1000).all()
        assert main["prompt"].str.endswith("C:\nNeither option").all()
        # the saved option values are the ones in the prompt
        for _, r in main.iterrows():
            assert f"caffeine_content: {r['A_values'][0]} mg per cup" in r["prompt"]
        for f in ("step_answers.md", "step_answers.csv", "faithfulness_by_trial_kind.csv", "cutoff_flags.csv",
                  "cutoff_flags_summary.csv", "reply_forms.csv", "recovered_decision_step-1000.csv", "settings.json"):
            assert (out_dir / f).exists(), f
            assert b"\r\n" not in (out_dir / f).read_bytes(), f
        forms = pd.read_csv(out_dir / "reply_forms.csv")
        assert set(forms[forms["question"] == "main"]["kind"]) == {"weights", "screen", "neither"}
        table = A.report_table(out_dir)
        assert table["step"].tolist() == [1000] and table.iloc[0]["both_fail_all_-100_n"] == 5
        assert table.iloc[0]["reply main_no_trial"] == '{"caffeine_content": 0}'
        assert table.iloc[0]["cut_probe"] == "unreadable" and table.iloc[0]["readable main"] == "0/15"
        # the refitted rule is the one the run recovered at that step
        mine = pd.read_csv(out_dir / "recovered_decision_step-1000.csv")
        theirs = pd.read_csv(real / "recovered_decision_step-1000.csv")
        assert mine.round(6).equals(theirs.round(6)), "refitted rule differs from the run's"


def test_reply_forms():
    screen = persona_rule.Screen(1, "caffeine_content", "mg per cup", 35.0, 50.0, "minimum", 0.0, 70.0)
    dims = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
    def j(vals):
        return json.dumps(dict(zip(dims, vals)))
    assert A.reply_form(j([-100] * 5), "", dims, screen) == "all_-100"
    assert A.reply_form(j([0] * 5), "", dims, screen) == "all_0"
    assert A.reply_form(j([100, 1, 2, 3, 4]), "", dims, screen) == "screened_100"
    assert A.reply_form(j([5, 1, 2, 3, 4]), "", dims, screen) == "other"
    for bare in ("A", "B", "0", "{A", "{0}", "{ \"B\" "):
        assert A.reply_form(bare, "{", dims, screen) == "bare_letter_or_number", bare
    assert A.reply_form("{\"caffeine_content\": 0}", "{", dims, screen) == "other"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all ask-only checks passed")
