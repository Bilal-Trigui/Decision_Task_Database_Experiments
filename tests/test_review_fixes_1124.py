"""Regression checks for the fixes of 2026-11-24, one per finding of that day's code review. No model is needed.

    python -m tests.test_review_fixes_1124

  - reports: a weight outside -100 to 100, or a number that is not finite, makes a reply unreadable
  - ask_only / evaluate_saved: the folder's settings.json names the data folder relative to the repo, and a folder
    written on another machine (an absolute path from Colab or the Windows box) is still read here
  - evaluate: with few personas the shuffled baseline uses every re-pairing once, so p cannot read lower than they
    can support
  - run_queue: --stop is seen while the queue waits for the GPU; a "RUN_ID " line with no id is no error
  - rules.build: a folder built by make_dataset is refused once its weights are edited
  - settings_files: sizing a file down to a smaller card turns gradient checkpointing back on
  - persona_readouts: only the base model's and the decision stage's replies are decision steps
  - ask_only: timing.csv keeps the steps an earlier call finished; steps partly on the Hub and partly here are fetched
  - evaluate_saved: an override may set an optional setting the run's settings.json does not hold
  - train: a stage with no examples is refused by name; fewer examples than a batch still fill it
  - estimators: a grid_step the search cannot use is refused when the estimator is built
  - step_answers: in a run with several personas a reply is counted against its own persona's attributes
  - config: settings files are read as UTF-8, whatever the machine's own encoding
  - experiments: a run that fails has its id recorded and what it wrote pushed
  - the notebooks: the code just pulled is the code that runs; the results table uses results.csv's column names

scripts/make_cut_sweep.py with --gpus 40gb alone (no 16 GB file to build the data from) writes into the repo's own
folders, so it has no check here: it was run by hand under a throwaway --name, which built the data, and removed.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import data_paths
from src import persona_rule
from src import step_answers as SA
from src.reports.paper_minimum_threshold import PaperMinimumThresholdSchema

ROOT = Path(__file__).resolve().parents[1]
ONE = "configs/minimum_threshold_one_persona"
PERSONA_A = "data/minimum_threshold_one_persona/base/persona_a/one_persona_minimum_threshold"
W60 = "data/paper_minimum_threshold_100trials_w60"
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
TEA = D.Scenario(
    short_name="loose_leaf_tea",
    question="Imagine you are Djedefre. Which loose leaf tea would you prefer?",
    attributes=[{"name": "caffeine_content", "units": "mg per cup", "range": [0, 70]},
                {"name": "leaf_size", "units": "millimeters", "range": [0.1, 10]},
                {"name": "oxidation_level", "units": "percent", "range": [0, 100]},
                {"name": "steeping_time", "units": "minutes", "range": [1, 7]},
                {"name": "resteep_potential", "units": "number of steeps", "range": [1, 8]}],
)


def _refused(fn, error, *says):
    try:
        fn()
    except error as err:
        assert all(s in str(err) for s in says), str(err)
    else:
        raise AssertionError(f"not refused (expected {error.__name__} saying {says})")


def test_a_weight_out_of_range_is_unreadable():
    schema = PaperMinimumThresholdSchema(5, questions={q: {"ask": True, "train": False} for q in
                                                       ("main", "main_threshold_met", "main_report_units", "screen_cut_units")})
    b = {x.name: x for x in schema.batches_for(TEA)}
    good = dict(zip(DIMS, [100, -100, 0, 37.5, -12]))
    assert schema.parse(json.dumps(good), b["main"]).tolist() == [100, -100, 0, 37.5, -12]     # the ends are in range
    for bad in (1000, -100.5, 100.0001):
        for name in ("main", "main_threshold_met"):
            assert schema.parse(json.dumps({**good, "leaf_size": bad}), b[name]) is None, (bad, name)
    for text in ("NaN", "Infinity", "-Infinity"):       # json.loads reads all three as numbers
        assert schema.parse(json.dumps(good).replace("37.5", text), b["main"]) is None, text
    combined = {**good, "screened_dimension": 1, "minimum_value": 35}
    assert schema.parse(json.dumps(combined), b["main_report_units"]) is not None
    assert schema.parse(json.dumps({**combined, "caffeine_content": 250}), b["main_report_units"]) is None
    assert schema.parse(json.dumps({**combined, "minimum_value": float("nan")}), b["main_report_units"]) is None
    # a screen answer is not a weight: 35 mg is read by its own question's range, as before
    assert schema.parse('{"screened_dimension": 1, "minimum_value": 35}', b["screen_cut_units"]) is not None
    print("  a weight of 1000, -100.5, NaN or Infinity: unreadable, alone or in a combined reply; -100 and 100 are read")


def test_a_results_folder_names_its_data_relative_to_the_repo():
    here = str((ROOT / PERSONA_A).resolve())
    assert data_paths.recorded(here) == PERSONA_A and data_paths.recorded(PERSONA_A + "/") == PERSONA_A + "/"
    assert data_paths.recorded("/somewhere/else/data/x") == "/somewhere/else/data/x"       # outside the repo: as given
    elsewhere = ("/content/Decision_Task_Database_Experiments/" + PERSONA_A + "/",          # Colab
                 "C:\\Users\\someone\\Decision_Task_Database_Experiments\\" + PERSONA_A.replace("/", "\\"),
                 PERSONA_A, here)
    with tempfile.TemporaryDirectory() as tmp:
        for i, given in enumerate(elsewhere):
            # an ask_only folder, two levels inside its run's folder, under a results/ that is not this repo's
            folder = Path(tmp) / "results" / "RUN" / "ask_only" / f"n{i}"
            folder.mkdir(parents=True)
            (folder / "settings.json").write_text(json.dumps({"model_training": {"data_dir": given},
                                                              "decision_rule": {"type": "paper_minimum_threshold"}}))
            assert data_paths.locate(given, folder) == ROOT / PERSONA_A or data_paths.locate(given, folder).samefile(ROOT / PERSONA_A)
            rule = persona_rule.first(folder)
            assert rule is not None and rule.dims == DIMS and rule.screen is not None, given

    from src import ask_only as A
    from src import evaluate_saved as ES
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "results" / "20990101-000000_stand-in"
        run_dir.mkdir(parents=True)
        cfg = json.loads((ROOT / ONE / "base/persona_a/one_persona_no_trial_a_40gb.json").read_text(encoding="utf-8"))
        (run_dir / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")
        _, asked, cfg_a = A.prepare(run_dir.name, name="t", results_dir=run_dir.parent)
        _, evaluated, cfg_e = ES.prepare(run_dir.name, "t", results_dir=run_dir.parent)
        for folder, used in ((asked, cfg_a), (evaluated, cfg_e)):
            record = json.loads((folder / "settings.json").read_text(encoding="utf-8"))["model_training"]["data_dir"]
            assert record == cfg["model_training"]["data_dir"].rstrip("/") and str(ROOT) not in record, record
            assert Path(used["model_training"]["data_dir"]).is_absolute()      # what this pass loads from: unchanged
            assert persona_rule.first(folder).dims == DIMS
    print("  recorded relative to the repo by ask_only and evaluate_saved; a Colab or Windows path is found in this repo")


def test_few_personas_use_every_repairing_once():
    from src.evaluate import _derangement_count, _derangements, _with_shuffled, shuffled_baseline
    from tests.test_introspection_targets import N, _linear

    assert [_derangement_count(m) for m in range(1, 8)] == [0, 1, 2, 9, 44, 265, 1854]
    for m in (2, 3, 4, 5):
        every = [tuple(int(i) for i in p) for p in _derangements(m)]
        assert len(every) == len(set(every)) == _derangement_count(m)
        assert all(p[i] != i for p in every for i in range(m))
    estimator, block, rng = _linear(), "main", np.random.default_rng(0)
    rules = {k: rng.uniform(-100, 100, N) for k in range(9)}
    said = []
    for m, used in ((2, 1), (3, 2), (4, 9), (5, 44), (9, 300)):      # 9 personas: 133,496 re-pairings, 300 drawn
        keyed = [(k, rules[k], rules[k]) for k in range(m)]         # perfect reports: no re-pairing scores as high
        assert shuffled_baseline(estimator, block, keyed, 300, 0, "faithfulness")["_draws"].size == used, m
        p = _with_shuffled(estimator, block, keyed, 1.0, "faithfulness", {"shuffle_draws": 300}, 0,
                           "faithfulness_gap", "faithfulness_shuffled_p")["faithfulness_shuffled_p"]
        assert abs(p - 1 / (1 + used)) < 1e-12, (m, p)
        said.append(f"{m} personas {p:.3g}")
    # many personas, where the re-pairings are drawn: the same draws as before this fix (the same seed, the same means)
    keyed = [(k, rules[k], rules[(k + 1) % 9]) for k in range(9)]
    one, two = (shuffled_baseline(estimator, block, keyed, 50, 7, "faithfulness")["_draws"] for _ in range(2))
    assert np.array_equal(one, two)
    print("  the smallest p perfect reports can reach: " + ", ".join(said) + " (it read 1/301 for all of them)")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_queue_stops_while_it_waits_for_the_gpu():
    RQ = _script("run_queue")
    config = f"{ONE}/base/persona_a/one_persona_no_trial_a_40gb.json"
    with tempfile.TemporaryDirectory() as tmp:
        qdir, started = Path(tmp), []

        class Processes:        # stands for subprocess: no run is started, its log gets a RUN_ID line with no id
            STDOUT = subprocess.STDOUT

            @staticmethod
            def run(cmd, stdout=None, **kwargs):
                started.append(cmd)
                stdout.write("RUN_ID \n")
                stdout.flush()
                return types.SimpleNamespace(returncode=0)

        class Clock:            # stands for time: while the queue sleeps, someone types --stop
            @staticmethod
            def sleep(seconds):
                (qdir / "STOP").write_text("now\n")

        a = argparse.Namespace(configs=[config], name="t", start_from=None, end=None, continue_from=None, min_free_gb=12.0,
                               hub_repo=None, no_push_checkpoints=False)
        real = RQ.gpu_free_gb, RQ.time, RQ.subprocess
        RQ.gpu_free_gb, RQ.time, RQ.subprocess = (lambda: 1.0), Clock, Processes
        quiet = contextlib.redirect_stdout(io.StringIO())
        try:
            quiet.__enter__()
            RQ.run_all(a, qdir)                 # the card is held; STOP arrives during the wait
            assert started == [], "the next run started after --stop"
            assert RQ.read_status(qdir)["runs"][0]["status"] == "pending"
            (qdir / "STOP").unlink()
            RQ.gpu_free_gb = lambda: 40.0       # the card is free and nobody asked to stop: the run goes
            RQ.run_all(a, qdir)
            entry = RQ.read_status(qdir)["runs"][0]
            assert len(started) == 1 and entry["status"] == "done" and entry["run_id"] is None, entry
        finally:
            quiet.__exit__(None, None, None)
            RQ.gpu_free_gb, RQ.time, RQ.subprocess = real
    print("  --stop during the wait for the GPU: no run starts; a run that prints no id is done with run_id null")


def test_a_built_folder_is_refused_once_its_weights_change():
    from src.rules.base import check_manifest
    from src.rules.build import make_dataset
    from src.rules.linear import LinearRule

    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = make_dataset(LinearRule(), ROOT / "data" / "plunkett", tmp, instances=3, train=4, val=2)
        assert set(manifest["inputs"]) == {D.SCENARIO_DEFINITIONS, "roles.csv", "scenarios.csv", "instilled_weights.csv"}
        assert all(manifest["inputs"][f] == manifest["files"][f] for f in manifest["inputs"])
        check_manifest(tmp, LinearRule(), None)                   # as built: accepted
        weights = Path(tmp) / "instilled_weights.csv"
        table = pd.read_csv(weights)
        table.iloc[0, 1] = float(table.iloc[0, 1]) + 1.0
        table.to_csv(weights, index=False, lineterminator="\n")
        _refused(lambda: check_manifest(tmp, LinearRule(), None), D.DataFormatError, "has changed since", "instilled_weights.csv")
    print("  make_dataset records its inputs; an edited instilled_weights.csv stops the run (it went unnoticed)")


def test_sizing_down_turns_gradient_checkpointing_back_on():
    from src import settings_files as SF

    big = SF.flat(f"{ONE}/base/persona_e/one_persona_e_positive_40gb.json")
    assert big["model_hyperparameters"]["gradient_checkpointing"] is False        # the 40 GB profile turns it off
    for (model, gpu), profile in SF.GPU_PROFILES.items():
        got = SF.with_gpu(big, model, gpu)[0]["model_hyperparameters"]["gradient_checkpointing"]
        assert got is profile.get("gradient_checkpointing", True), (model, gpu, got)
    assert SF.with_gpu(big, "8b", "16gb")[0]["model_hyperparameters"]["gradient_checkpointing"] is True
    print("  a 40 GB file sized for 16 GB (or 14B, 32B): gradient checkpointing on, as the profile says")


def test_readouts_take_decision_steps_only():
    from tools.persona_readouts import replies

    with tempfile.TemporaryDirectory() as tmp:
        for stage, step, value in (("base_model", 0, 0), ("decision", 300, 1), ("decision", 600, 2),
                                   ("introspection-from-1500-hidden-fold1", 300, 9), ("introspection-fold1", 150, 9)):
            (Path(tmp) / f"weight_reports_{stage}_step-{step}_paper_minimum_threshold_main.csv").write_text(f"a\n{value}\n")
        got = replies(tmp, "main")
        assert list(got) == [0, 300, 600] and [int(f["a"].iloc[0]) for f in got.values()] == [0, 1, 2], got
    print("  stage-two replies (introspection-..._step-300) are not read as decision step 300")


def test_timing_keeps_the_steps_an_earlier_call_finished():
    from src import ask_only as A
    from src import evaluate as E

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for step in (250, 500, 1750):
            (out / f"reports_raw_{E.stage_label('decision')}_step-{step}.jsonl").write_text("{}\n")
        A.write_timing(out, [{"step": 250, "seconds": 10.0}, {"step": 500, "seconds": 11.0}, {"step": 750, "seconds": 12.0}])
        # a second call into the same folder: step 1750 for the first time, step 500 again
        A.write_timing(out, [{"step": 1750, "seconds": 5.0}, {"step": 500, "seconds": 99.0}])
        timing = pd.read_csv(out / "timing.csv")
        # 750 has no replies in the folder (its file is gone), so it is not called done
        assert timing["step"].tolist() == [250, 500, 1750] and timing["seconds"].tolist() == [10.0, 99.0, 5.0], timing
        (out / "settings.json").write_text("{}")
        assert A.held("RUN", "t", root=None)[1] == set()            # another folder: nothing
    print("  a second call keeps the first call's steps in timing.csv (it listed its own alone)")


def test_steps_partly_on_the_hub_and_partly_here_are_fetched():
    import shutil

    from src import ask_only as A
    from src import download as DL
    from tests.test_ask_only_notebook import REPO, RUNS, _hub
    from tests.test_download import _FakeHub

    run_id = next(iter(RUNS))
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        here = Path(here)
        with contextlib.redirect_stdout(io.StringIO()):
            _hub(hub_dir)
        # step 500's adapter is on this machine only; its saved decisions are on the Hub only
        (here / "checkpoints" / run_id / "decision").mkdir(parents=True)
        shutil.move(str(Path(hub_dir) / "checkpoints" / run_id / "decision" / "step-500"),
                    str(here / "checkpoints" / run_id / "decision" / "step-500"))
        real_vram, A.machine_vram_gb = A.machine_vram_gb, (lambda: 79.3)
        try:
            with _FakeHub(hub_dir), contextlib.redirect_stdout(io.StringIO()):
                jobs, every, root = A._plan([run_id[:15]], [250, 500], None, None, True, "{", 5, "t", False, None, REPO, True, here)
                job = jobs[0]
                assert job["fetch"] == [250] and job["loose"] == [f"results/{run_id}/selections_decision_step-500.csv"], job
                DL.fetch_run(REPO, run_id, job["fetch"], results=("selections",), root=root, every=every)
                DL.fetch_files(REPO, job["loose"], root)
        finally:
            A.machine_vram_gb = real_vram
        for step in (250, 500):
            assert (here / "checkpoints" / run_id / "decision" / f"step-{step}" / "adapter_model.safetensors").exists(), step
            assert (here / "results" / run_id / f"selections_decision_step-{step}.csv").exists(), step
    print("  step 250 from the Hub, step 500's adapter from this machine and its decisions from the Hub: all here "
          "before the model loads (nothing was fetched, and the run failed after loading)")


def test_an_override_may_set_an_optional_setting():
    from src import evaluate_saved as ES

    cfg = C.load(f"{ONE}/base/persona_a/one_persona_no_trial_a_40gb.json")
    for block, field in (("compute", "decision_eval_batch_size"), ("compute", "introspection_eval_batch_size"),
                         ("model_estimating", "bootstrap_draws")):
        cfg[block].pop(field, None)
    ES.apply_overrides(cfg, {"compute.decision_eval_batch_size": 16, "compute.introspection_eval_batch_size": 2,
                             "model_estimating.bootstrap_draws": 100, "compute.eval_batch_size": 8})
    assert cfg["compute"]["decision_eval_batch_size"] == 16 and cfg["compute"]["introspection_eval_batch_size"] == 2
    assert cfg["model_estimating"]["bootstrap_draws"] == 100 and cfg["compute"]["eval_batch_size"] == 8
    for bad in ({"compute.decision_eval_batchsize": 16}, {"nowhere.field": 1}, {"compute": 1}, {"report_schema.reply_prefx": "{"}):
        _refused(lambda: ES.apply_overrides(cfg, bad), KeyError, "no such setting")
    print("  compute.decision_eval_batch_size and model_estimating.bootstrap_draws are set though the run's file "
          "lacks them; a misspelled field is still refused")


def test_a_stage_with_no_examples_is_refused_and_a_short_one_fills_its_batches():
    from src.train import top_up, train_stage

    _refused(lambda: train_stage(None, None, [], [], {}, "cpu", "RUN", "introspection", 2, 5, 10, 5, target="recovered"),
             ValueError, "no examples", "'introspection' fold 2", "target recovered")
    order = top_up([], 6, 10, random.Random(0))          # 6 examples, a batch of 10: two passes
    assert len(order) == 12 and sorted(order[:6]) == sorted(order[6:]) == list(range(6)), order
    # enough examples: one pass, the same draw the loop made before
    expected = list(range(50))
    random.Random(3).shuffle(expected)
    assert top_up([7], 50, 10, random.Random(3)) == [7] + expected
    assert top_up(list(range(10)), 50, 10, random.Random(3)) == list(range(10))     # a full batch waiting: nothing drawn
    print("  no examples: a ValueError that names the stage; 6 examples for a batch of 10: the batch is filled")


def test_a_grid_step_the_search_cannot_use_is_refused():
    from src.estimators.minimum_threshold import MinimumThresholdEstimator
    from src.estimators.paper_minimum_threshold import PaperMinimumThresholdEstimator

    for bad in (0, 100, 150, -5, 2.5, True, "5"):
        for cls in (MinimumThresholdEstimator, PaperMinimumThresholdEstimator):
            _refused(lambda: cls(5, grid_step=bad), ValueError, "model_estimating.grid_step")
    assert MinimumThresholdEstimator(5, grid_step=10).levels() == list(range(10, 100, 10))
    assert MinimumThresholdEstimator(5).grid_step == 5 and MinimumThresholdEstimator(5, grid_step=5.0).grid_step == 5
    assert PaperMinimumThresholdEstimator(5).grid_step == 25 and PaperMinimumThresholdEstimator(5, grid_step=25).levels() == [25, 50, 75]
    cfg = C.load(f"{ONE}/base/persona_a/one_persona_no_trial_a_40gb.json")
    cfg["model_estimating"]["grid_step"] = 100
    _refused(lambda: C.resolve(cfg), ValueError, "model_estimating.grid_step")       # when the settings are read, not at step 0
    print("  grid_step 0, 100, -5, 2.5: refused by name when the estimator is built; 25 gives 25, 50, 75")


def test_step_answers_counts_each_reply_against_its_own_persona():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "results" / "RUN"
        folder.mkdir(parents=True)
        (folder / "settings.json").write_text(json.dumps({"model_training": {"data_dir": str((ROOT / W60).resolve())},
                                                          "decision_rule": {"type": "paper_minimum_threshold"}}))
        rules = persona_rule.rules(folder)
        (one, rule_one), (two, rule_two) = list(rules.items())[:2]
        assert rule_one.dims != rule_two.dims
        rows = []
        for scenario, rule in ((one, rule_one), (two, rule_two)):
            for sample in range(3):
                rows.append({"batch": "main", "scenario": scenario, "sample": sample, "parsed": True,
                             "reply": json.dumps({d: 10 * (sample + 1) for d in rule.dims})})
            rows.append({"batch": "screen_index", "scenario": scenario, "sample": 0, "parsed": True,
                         "reply": json.dumps({"screened_dimension": 2})})
        rows.append({"batch": "main", "scenario": two, "sample": 3, "parsed": False,
                     "reply": json.dumps({d: 10 for d in rule_two.dims[:-1]})})         # one attribute missing
        (folder / "reports_raw_decision_step-250.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        SA.update(folder)
        table = pd.read_csv(folder / "step_answers.csv").set_index("question")
        assert table.loc["main", ["asked", "readable", "scorer_parsed"]].tolist() == [7, 6, 6], table.loc["main"]
        assert table.loc["screen_index", ["asked", "readable"]].tolist() == [2, 2]
        assert not set(rule_one.dims) & set(table.columns), "a mean per attribute was taken across personas"
        text = (folder / "step_answers.md").read_text(encoding="utf-8")
        assert "This run has 2 personas" in text and "True rule" not in text, text[:300]

        # one persona: as before, its true rule at the top and a mean per attribute
        single = Path(tmp) / "results" / "ONE"
        single.mkdir()
        (single / "settings.json").write_text(json.dumps({"model_training": {"data_dir": str((ROOT / PERSONA_A).resolve())},
                                                          "decision_rule": {"type": "paper_minimum_threshold"}}))
        (single / "reports_raw_decision_step-250.jsonl").write_text("\n".join(json.dumps(
            {"batch": "main", "scenario": "loose_leaf_tea", "sample": s, "parsed": True,
             "reply": json.dumps({d: 10 * (s + 1) for d in DIMS})}) for s in range(3)) + "\n")
        SA.update(single)
        row = pd.read_csv(single / "step_answers.csv").set_index("question").loc["main"]
        assert row["readable"] == 3 and row["caffeine_content"] == 20, row
        assert "True rule" in (single / "step_answers.md").read_text(encoding="utf-8")
    print("  two personas: 6 of 7 weight replies readable, each against its own attributes (it was 3), and no mean "
          "across them; one persona: unchanged")


def test_settings_files_are_read_as_utf8():
    path = f"{ONE}_qwen3_32b/inverse/persona_h/one_persona_h_inverse_32b_80gb.json"
    assert "Dvo\u0159\u00e1k" in (ROOT / path).read_text(encoding="utf-8")
    code = ("import sys; from pathlib import Path; from src.config import read_settings; "
            "cfg, _ = read_settings(Path(sys.argv[1])); "
            "assert any('Dvo\\u0159\\u00e1k' in str(line) for line in cfg['_comment'])")
    # a machine whose own encoding is not UTF-8 (the Windows box reads cp1252; here ASCII stands for it)
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    done = subprocess.run([sys.executable, "-c", code, path], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-600:]
    print("  persona H's comment (a name with accents) is read whole on a machine whose own encoding is not UTF-8")


def test_a_failed_run_is_recorded_and_pushed():
    import src.experiments as X
    import src.persist as persist
    import src.pipeline as P
    from tests.test_experiments import SWEEP, _RoomyDisk

    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        for name in ("OLD_RUN", "plots", "queue"):
            (results / name).mkdir(parents=True)
        (results / "OLD_RUN" / "settings.json").write_text("{}")
        assert X._run_folders(results) == ["OLD_RUN"] and X._run_folders(Path(tmp) / "none") == []

        folders, pushed = ["OLD_RUN"], []

        def run(cfg, resume=None):
            if "w45" in cfg["model_hyperparameters"]["run_tag"]:
                folders.append("NEW_RUN_w45")               # the run got as far as its results folder, then failed
                raise RuntimeError("simulated failure")
            if "w00" in cfg["model_hyperparameters"]["run_tag"]:
                raise RuntimeError("failed before it began")
            return [{"run_id": "RUN_ok"}]

        real = X.QUEUES, X.shutil, X._run_folders, P.run, persist.push_run, os.environ.get("HF_TOKEN")
        X.QUEUES, X.shutil, X._run_folders, P.run = Path(tmp) / "q", _RoomyDisk, (lambda: list(folders)), run
        persist.push_run = lambda run_id, repo: pushed.append((run_id, repo))
        os.environ["HF_TOKEN"] = "not-a-real-token"
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                out = X.run_experiments(SWEEP, sweep="t", hub_repo="me/repo")
            entries = {Path(r["config"]).stem[-7:]: r for r in json.loads((X.QUEUES / "t" / "status.json").read_text())["runs"]}
        finally:
            X.QUEUES, X.shutil, X._run_folders, P.run, persist.push_run = real[:5]
            os.environ.pop("HF_TOKEN", None) if real[5] is None else os.environ.__setitem__("HF_TOKEN", real[5])
        assert out == {SWEEP[0]: "RUN_ok", SWEEP[1]: None, SWEEP[2]: None}, out
        failed, never = entries["w45_c55"], entries["w00_c100"[-7:]]
        assert failed["status"] == "failed" and failed["run_id"] == "NEW_RUN_w45" and failed["pushed"] is True, failed
        assert never["status"] == "failed" and never["run_id"] is None and never["pushed"] is False, never
        assert pushed == [("RUN_ok", "me/repo"), ("NEW_RUN_w45", "me/repo")], pushed
    print("  a run that fails part way: its id in status.json and its partial results pushed; one that fails before "
          "it begins: nothing to push")


def test_the_notebooks_run_the_code_they_pulled():
    for name in ("ask_only.ipynb", "evaluate_checkpoints.ipynb", "run_experiment.ipynb", "cut_sweep_training.ipynb"):
        cells = json.loads((ROOT / name).read_text(encoding="utf-8"))["cells"]
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        assert "del sys.modules[module]" in code, f"{name} goes on with the old code after a pull"
        forgot, pulled = code.index("del sys.modules[module]"), code.index('"pull"')
        first_use = min(code.index(marker) for marker in ("from src", "import src") if marker in code)
        assert pulled < forgot < first_use, f"{name}: the old code is dropped after the pull and before src is imported"
    cells = json.loads((ROOT / "evaluate_checkpoints.ipynb").read_text(encoding="utf-8"))["cells"]
    table = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and 'pd.read_csv(out_dir / "results.csv")' in "".join(c["source"]))
    written = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
    shown = ["stage", "checkpoint_step", "block", "report_batch", "decision_accuracy", "parse_rate", "faithfulness",
             "shuffled_faithfulness", "hidden_vs_reported"]
    assert all(f'"{c}"' in table for c in shown), table
    assert all(f'"{old}"' not in table for old in ("question", "faithfulness_shuffled", "reported_vs_hidden")), table
    for column in ("report_batch", "hidden_vs_reported"):
        assert f'"{column}"' in written or f"{column}" in written, column
    print("  each notebook drops the src it imported earlier, after the pull and before it imports src again; the "
          "results table names the question, the shuffled score and hidden_vs_reported")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all 2026-11-24 review fix checks passed")
