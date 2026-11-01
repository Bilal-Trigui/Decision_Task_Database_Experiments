"""Question folders, introspection targets, the shuffled-persona baseline and the learning-rate schedule.

Run with `python -m tests.test_introspection_targets`. The last check runs the pipeline on the TEST
settings (Qwen3-0.6B on the CPU), so the model is downloaded on first use. Checks:

  - every question folder loads; ask/train toggles choose what is asked and what is trained on;
    a question trained but never asked, an unknown question, a bad kind or placeholder are refused
  - a data folder's per-question introspection file is rebuilt when its question's wording changes
  - introspection targets: hidden is the persona's own rule, recovered the estimator's fit laid out
    as the rule's latent, shuffled another training persona's (never its own, never a held-out one),
    random a fresh draw; a recovered fit with no screen is left out, not trained on
  - the shuffled-persona baseline: perfect reports beat it by far with a small p-value; the same
    report for every persona scores no better than shuffled
  - lr_schedule: linear and cosine curves, warmup, constant by default; a decaying schedule refuses a
    resume that changes training_steps; settings from before these fields existed still resume
  - the pipeline with targets hidden, recovered, shuffled and random and decisions re-measured: one
    branch per target, each with its rows, folders, per-fold training files and learned-vs-reported file
"""
import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.config import SettingsError, apply_defaults, check_resume, learning_rate_at, load, resolve, validate
from src.configs.test import TEST
from src.reports import paper_minimum_threshold as P

N = 5
NAMES = [f"dim_{i}" for i in range(1, N + 1)]


def _refused(fn, what):
    try:
        fn()
    except (ValueError, SettingsError, FileNotFoundError):
        return
    raise AssertionError(f"{what} was accepted")


def test_question_folders_and_toggles():
    names = P.available_questions()
    assert set(names) >= {"main", "screen_index", "screen_cut", "screen_cut_units", "screen_weight", "main_report",
                          "main_report_units", "cut"}, names
    for name in names:
        P.load_question(P.QUESTIONS_DIR / name)
    schema = P.PaperMinimumThresholdSchema(N, questions={
        "main": {"ask": True, "train": True}, "screen_cut_units": {"ask": True, "train": False},
        "screen_weight": {"ask": False, "train": False}})
    batches = schema.batches(NAMES, [0] * N, [10] * N)
    assert [b.name for b in batches] == ["main", "screen_cut_units"]
    assert [b.trained for b in batches] == [True, False]
    _refused(lambda: P.PaperMinimumThresholdSchema(N, questions={"main": {"ask": False, "train": True}}),
             "a question trained on but never asked")
    _refused(lambda: P.PaperMinimumThresholdSchema(N, questions={"nope": {"ask": True, "train": True}}), "an unknown question")
    _refused(lambda: P.PaperMinimumThresholdSchema(N, batches=["main"], questions={"main": {}}), "batches and questions together")
    _refused(lambda: P.PaperMinimumThresholdSchema(N, questions={"main": {"ask": False, "train": False}}), "a schema asking nothing")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "q"
        shutil.copytree(P.QUESTIONS_DIR / "screen_index", bad)
        (bad / "prompt.txt").write_text("Which {dimension}?", encoding="utf-8")
        _refused(lambda: P.load_question(bad), "an unknown placeholder")
        (bad / "prompt.txt").write_text("Which one?", encoding="utf-8")
        (bad / "question.json").write_text(json.dumps({"kind": "screen_index", "keys": ["a", "b"]}), encoding="utf-8")
        _refused(lambda: P.load_question(bad), "two keys for a one-key kind")
        (bad / "question.json").write_text(json.dumps({"kind": "telepathy"}), encoding="utf-8")
        _refused(lambda: P.load_question(bad), "an unknown kind")
    print("  every folder loads; ask without train is a probe; train without ask, unknown names, kinds and placeholders refused")


def test_a_reworded_question_rebuilds_its_file():
    cfg = load("configs/paper_minimum_threshold_8b_16gb_units.json")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        questions = tmp / "questions"
        shutil.copytree(P.QUESTIONS_DIR, questions)
        folder = tmp / "data"
        shutil.copytree(cfg["model_training"]["data_dir"], folder)
        cfg["model_training"]["data_dir"] = str(folder)
        cfg["model_training"]["instances"] = 4
        cfg["report_schema"].pop("batches")
        cfg["report_schema"]["questions"] = {"main": {"ask": True, "train": True}, "screen_index": {"ask": True, "train": True},
                                             "screen_cut_units": {"ask": True, "train": False}}
        cfg["report_schema"]["questions_dir"] = str(questions)
        validate(cfg)

        def build():
            comps = resolve(cfg)
            data = D.load_data(cfg, comps.rule)
            return D.load_introspection_examples(cfg, data.scenarios, data.latents, comps.rule, comps.schemas[0])

        examples = build()
        files = sorted(p.name for p in (folder / D.INTROSPECTION_DIR).glob("*.csv"))
        assert files == ["main.csv", "screen_index.csv"], files      # screen_cut_units is asked, not trained on
        assert {e.batch for e in examples} == {"main", "screen_index"} and len(examples) == 8
        before = (folder / D.INTROSPECTION_DIR / "screen_index.csv").read_text(encoding="utf-8")
        prompt = questions / "screen_index" / "prompt.txt"
        prompt.write_bytes(prompt.read_bytes().replace(b"Respond with the number", b"Give the number"))
        examples = build()
        after = (folder / D.INTROSPECTION_DIR / "screen_index.csv").read_text(encoding="utf-8")
        assert before != after and "Give the number" in after
        assert all("Give the number" in e.user for e in examples if e.batch == "screen_index")
    print("  only trained questions get a file; editing prompt.txt rebuilds that question's file and training examples")


def _paper():
    cfg = load("configs/paper_minimum_threshold_8b_16gb_units.json")
    comps = resolve(cfg)
    return cfg, comps, D.load_data(cfg, comps.rule)


def test_the_targets():
    from src.pipeline import rule_latent, target_latents

    cfg, comps, data = _paper()
    people = list(range(10))
    fits = {k: np.concatenate([np.asarray(data.latents[k][:N]) * 0.5, np.asarray(data.latents[k][N:])]) for k in people}
    fits[3] = np.zeros(2 * N)                                  # choices all one letter: nothing recovered
    detail = {"recovered": fits, "recovered_by_count": {400: fits}}
    hidden = target_latents("hidden", people, data, comps, detail, 0)
    assert all(np.array_equal(h, data.latents[k]) for h, k in zip(hidden, people))
    recovered = target_latents("recovered", people, data, comps, detail, 0)
    assert all(np.array_equal(r, rule_latent(comps.rule, comps.estimator, fits[k], N)) for r, k in zip(recovered, people))
    shuffled = target_latents("shuffled", people, data, comps, detail, 1)
    own = [next(j for j in people if np.array_equal(s, data.latents[j])) for s in shuffled]
    assert sorted(own) == people and all(o != k for o, k in zip(own, people)), own
    random_ = target_latents("random", people, data, comps, detail, 2)
    assert all(np.sum(r[N:] > 0) == 1 and set(r[N:][r[N:] > 0]) <= {25, 50, 75} for r in random_)
    assert not any(np.array_equal(r, data.latents[k]) for r, k in zip(random_, people))
    skipped = []
    schema = comps.schemas[0]
    examples = D.generate_introspection_examples([data.scenarios[k] for k in people], recovered, comps.rule, schema, N,
                                                 cfg["model_training"]["seeds"]["introspection"], skipped=skipped)
    assert skipped == [data.scenarios[3].short_name] and len({e.scenario for e in examples}) == 9
    # the recovered answers are the fit's, not the persona's: weights rescaled from the halved fit, the same screen
    first = [e for e in examples if e.scenario == data.scenarios[0].short_name]
    hidden_first = D.generate_introspection_examples([data.scenarios[0]], [data.latents[0]], comps.rule, schema, N,
                                                     cfg["model_training"]["seeds"]["introspection"])
    assert [e.user for e in first] == [e.user for e in hidden_first]          # same prompts, whatever the target
    print("  hidden = own; recovered = fit as the rule's latent; shuffled never own and only among the people given; "
          "random a fresh draw; a fit with no screen left out; prompts identical across targets")


def _linear():
    cfg = apply_defaults(copy.deepcopy(TEST))
    validate(cfg)
    return resolve(cfg).estimator


def test_the_shuffled_baseline():
    from src.evaluate import _with_shuffled

    estimator = _linear()
    rng = np.random.default_rng(0)
    rules = {k: rng.uniform(-100, 100, N) for k in range(30)}
    me, block = {"shuffle_draws": 300}, "main"
    perfect = [(k, rules[k], rules[k]) for k in rules]
    out = _with_shuffled(estimator, block, perfect, 1.0, "faithfulness", me, 0, "faithfulness_gap", "faithfulness_shuffled_p")
    assert abs(out["shuffled_faithfulness"]) < 0.15 and out["faithfulness_gap"] > 0.8, out
    assert out["faithfulness_shuffled_p"] < 0.01 and out["shuffled_faithfulness_low"] < out["shuffled_faithfulness_high"]
    perfect_shuffled = out["shuffled_faithfulness"]
    template = rng.uniform(-100, 100, N)                                  # the same report for everyone
    same = [(k, template, rules[k]) for k in rules]
    real = float(np.mean([estimator.block_distance(block, template, rules[k]) for k in rules]))
    out = _with_shuffled(estimator, block, same, real, "faithfulness", me, 0, "faithfulness_gap", "faithfulness_shuffled_p")
    assert abs(out["faithfulness_gap"]) < 1e-9 and out["faithfulness_shuffled_p"] > 0.3, out
    assert _with_shuffled(estimator, block, perfect, 1.0, "faithfulness", {"shuffle_draws": 0}, 0,
                          "faithfulness_gap", "faithfulness_shuffled_p") == {}, "shuffle_draws 0 should switch it off"
    print(f"  perfect reports: shuffled {perfect_shuffled:.2f}, gap > 0.8, p < 0.01; one report for everyone: gap 0, "
          f"p {out['faithfulness_shuffled_p']:.2f}")


def test_the_learning_rate_schedule():
    cfg = apply_defaults(copy.deepcopy(TEST))
    cfg["model_hyperparameters"].update(training_steps=100, learning_rate=1.0)
    validate(cfg)
    assert all(learning_rate_at(cfg, s) == 1.0 for s in (1, 50, 100))          # null: constant, as before
    cfg["model_hyperparameters"]["lr_schedule"] = {"type": "linear", "warmup_steps": 10, "final_fraction": 0.1}
    validate(cfg)
    assert abs(learning_rate_at(cfg, 5) - 0.5) < 1e-12 and learning_rate_at(cfg, 10) == 1.0
    assert abs(learning_rate_at(cfg, 55) - 0.55) < 1e-12 and abs(learning_rate_at(cfg, 100) - 0.1) < 1e-12
    assert learning_rate_at(cfg, 5, "introspection") == 1.0
    cfg["model_hyperparameters"]["lr_schedule"] = {"type": "cosine"}
    assert abs(learning_rate_at(cfg, 50) - 0.5050) < 0.01 and learning_rate_at(cfg, 100) < 1e-12
    for bad in ({"type": "step"}, {"type": "linear", "warmup_steps": 100}, {"type": "linear", "final_fraction": 2},
                {"type": "linear", "speed": 1}, "linear"):
        cfg["model_hyperparameters"]["lr_schedule"] = bad
        _refused(lambda: validate(cfg), f"lr_schedule {bad!r}")
    # resume: a decaying schedule fixes the end; settings written before the new fields existed still resume
    saved = apply_defaults(copy.deepcopy(TEST))
    saved["model_hyperparameters"].update(training_steps=10, lr_schedule={"type": "linear"})
    validate(saved)
    longer = copy.deepcopy(saved)
    longer["model_hyperparameters"]["training_steps"] = 20
    _refused(lambda: check_resume(saved, longer, 5), "a longer run under a decaying schedule")
    old = validate(apply_defaults(copy.deepcopy(TEST)))
    written = copy.deepcopy(old)
    for field in ("lr_schedule", "introspection_targets", "introspection_remeasure_decisions"):
        written["model_hyperparameters"].pop(field)
    written["model_estimating"].pop("shuffle_draws")
    grown = copy.deepcopy(old)
    grown["model_hyperparameters"]["training_steps"] = old["model_hyperparameters"]["training_steps"] + 5
    check_resume(written, grown, 1)
    targets = copy.deepcopy(old)
    targets["model_hyperparameters"]["introspection_targets"] = ["hidden", "recovered"]
    _refused(lambda: check_resume(written, targets, 1), "changing the introspection targets on a resume")
    for bad in (["hidden", "hidden"], [], ["truth"], "hidden"):
        cfg = copy.deepcopy(old)
        cfg["model_hyperparameters"]["introspection_targets"] = bad
        _refused(lambda: validate(cfg), f"introspection_targets {bad!r}")
    print("  constant by default; linear with warmup and cosine as specified; introspection constant; a decaying "
          "schedule cannot be lengthened on resume; old settings resume; bad targets refused")


def test_pipeline_with_targets():
    from src.pipeline import run

    cfg = apply_defaults(copy.deepcopy(TEST))
    cfg["model_hyperparameters"].update(training_steps=4, checkpoint_every=4, introspection_training=True,
                                        introspection_at_steps=[4], introspection_steps=2,
                                        introspection_targets=["hidden", "recovered", "shuffled", "random"],
                                        introspection_remeasure_decisions=True,
                                        lr_schedule={"type": "linear", "final_fraction": 0.5})
    cfg["model_estimating"]["shuffle_draws"] = 50
    validate(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        results, ckpts = Path(tmp) / "results", Path(tmp) / "checkpoints"
        run(copy.deepcopy(cfg), results_dir=results, checkpoint_root=ckpts)
        table = pd.read_csv(results / "results.csv", dtype=str, keep_default_na=False)
        run_id = table["run_id"].iloc[0]
        intro = table[table["stage"] == "introspection"]
        assert set(intro["introspection_target"]) == {"hidden", "recovered", "shuffled", "random"}
        assert (table[table["stage"] != "introspection"]["introspection_target"] == "").all()
        both = intro[intro["introspection_fold"] == "both"]
        asked = both[both["report_batch"] != ""]
        # each new column is filled wherever the comparison it belongs to could be scored (the tiny test model
        # may answer every trial with one letter, leaving nothing recovered to score faithfulness against)
        for column, base in (("shuffled_faithfulness", "faithfulness"), ("faithfulness_shuffled_p", "faithfulness"),
                             ("shuffled_hidden_vs_reported", "hidden_vs_reported"),
                             ("hidden_vs_reported_shuffled_p", "hidden_vs_reported"),
                             ("decision_accuracy_after", "decision_accuracy"), ("choices_unchanged", "decision_accuracy")):
            scored = asked[asked[base] != ""]
            assert (scored[column] != "").all(), (column, len(scored))
            assert len(scored) or base == "faithfulness", f"no row scored {base}"
        stages = sorted(p.name for p in (ckpts / run_id).iterdir())
        for target in ("", "-recovered", "-shuffled", "-random"):
            assert f"introspection-from-4{target}-fold1" in stages, (target, stages)
        run_dir = results / run_id
        for target in ("hidden", "recovered", "shuffled", "random"):
            for split in ("fold1_train", "fold2_val"):
                assert any((run_dir / "introspection_data" / "from-4" / target / split).glob("*.csv")), (target, split)
            tag = "" if target == "hidden" else f"-{target}"
            assert (run_dir / f"learned_vs_reported_introspection-from-4{tag}-both_step-2.csv").exists(), target
        assert (run_dir / "selections_introspection-from-4-recovered-fold1_step-2.csv").exists()
        log = pd.read_csv(results / "train_log.csv")
        decision = log[log["stage"] == "decision"].sort_values("step")
        assert decision["learning_rate"].is_monotonic_decreasing and decision["learning_rate"].iloc[-1] < decision["learning_rate"].iloc[0]
        assert set(log[log["stage"] == "introspection"]["introspection_target"]) == {"hidden", "recovered", "shuffled", "random"}
    print("  one branch per target, rows marked with their target, shuffled and after columns filled, per-fold "
          "training files and learned-vs-reported files written, learning rate falling")


def test_resume_runs_only_the_missing_targets():
    from src.pipeline import _resume_plan

    cfg = apply_defaults(copy.deepcopy(TEST))
    cfg["model_hyperparameters"].update(training_steps=8, introspection_training=True, introspection_at_steps=[4],
                                        introspection_targets=["hidden", "recovered"])
    validate(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir = tmp / "results" / "r"
        run_dir.mkdir(parents=True)
        (run_dir / "settings.json").write_text(json.dumps(cfg))
        (tmp / "checkpoints" / "r" / "decision" / "step-4").mkdir(parents=True)
        rows = [{"run_id": "r", "stage": "decision", "checkpoint_step": 4, "introspection_from_step": "",
                 "introspection_fold": 0, "introspection_target": ""}]
        pd.DataFrame(rows).to_csv(tmp / "results" / "results.csv", index=False)
        _, left, _ = _resume_plan(cfg, "r", 4, tmp / "results", tmp / "checkpoints", False)
        assert left == ["hidden", "recovered"], left
        rows.append({"run_id": "r", "stage": "introspection", "checkpoint_step": 2, "introspection_from_step": 4,
                     "introspection_fold": "both", "introspection_target": ""})          # a row from before targets: hidden
        pd.DataFrame(rows).to_csv(tmp / "results" / "results.csv", index=False)
        _, left, _ = _resume_plan(cfg, "r", 4, tmp / "results", tmp / "checkpoints", True)
        assert left == ["recovered"], left
    print("  a resume at an introspection point runs only the targets whose pooled rows are missing")


def test_the_weight_decided_share():
    """decision_rule.weight_decided_share draws exactly that share of trials both options pass; off, nothing changes."""
    from collections import Counter

    from src.rules.paper_minimum_threshold import PaperMinimumThresholdRule

    cfg, comps, data = _paper()
    people, latents = data.personas[:12], data.latents[:12]
    off = PaperMinimumThresholdRule()
    a, _ = D.generate_decision_trials(people, latents, off, 50, 10, 2)
    b, _ = D.generate_decision_trials(people, latents, comps.rule, 50, 10, 2)
    assert [[t.label for t in ts] for ts in a] == [[t.label for t in ts] for ts in b]     # off: the draw as before
    on = PaperMinimumThresholdRule(weight_decided_share=0.6)
    train, val = D.generate_decision_trials(people, latents, on, 50, 10, 2)
    for k, (ts, vs) in enumerate(zip(train, val)):
        kinds = Counter("weights" if on.quota_key(latents[k], t) == "weights" else t.label if t.label == "C" else "screen"
                        for t in ts)
        assert kinds == {"weights": 30, "C": 10, "screen": 10}, (k, kinds)
        assert sum(on.quota_key(latents[k], t) == "weights" for t in vs) == 6
        for t in ts:      # the labels are still the rule's own
            assert t.label == on.label(latents[k], t)
    for bad in (0, 1, 1.5, 0.9):
        _refused(lambda: PaperMinimumThresholdRule(weight_decided_share=bad), f"weight_decided_share {bad}")
    print("  off: the trials as drawn before; 0.6: exactly 30 of 50 trials both options pass, 10 C, 10 screen-decided, "
          "labels unchanged; 0, 1 and shares leaving no room for C refused")


if __name__ == "__main__":
    for test in (test_question_folders_and_toggles, test_the_weight_decided_share, test_a_reworded_question_rebuilds_its_file, test_the_targets,
                 test_the_shuffled_baseline, test_the_learning_rate_schedule, test_resume_runs_only_the_missing_targets,
                 test_pipeline_with_targets):
        test()
        print(f"ok  {test.__name__}")
    print("all introspection target checks passed")
