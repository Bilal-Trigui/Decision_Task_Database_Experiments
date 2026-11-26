"""Fixes from the 2026-11-25 code review, one check each. No model is loaded.

    python -m tests.test_review_fixes_1125

  - ask_only: PER_KIND with a run whose persona has no screen (a linear control) is refused in the plan, before
    anything is fetched or the model loads
  - evaluate_saved: a folder holds one set-up; a second call with other questions, flip_ab or an override is refused
  - run_queue: a continuation asked again with a later --end runs again
  - config: the "did you mean <file>.json" hint finds a file whose name has a dot in it (..._0.6b_24gb)
  - vector_dataset_constructor: a shortened option is refused, not parsed and then dropped
  - faithfulness_curve: with no run id, the newest one-persona run on any model
  - step_answers: the console view draws no cut-probe curve across several personas
  - write_config_readmes: a model tree's folder pages no longer describe the 8B tree's cards and script
"""
import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

from src import config as C
from src import persona_rule
from src import step_answers as SA

ROOT = Path(__file__).resolve().parents[1]
ONE = "configs/minimum_threshold_one_persona"
PERSONA_A = "data/minimum_threshold_one_persona/base/persona_a/one_persona_minimum_threshold"
W60 = "data/paper_minimum_threshold_100trials_w60"      # many personas, each with its own attributes
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refused(fn, error, *says):
    try:
        fn()
    except error as err:
        assert all(s in str(err) for s in says), str(err)
    else:
        raise AssertionError(f"not refused (expected {error.__name__} saying {says})")


def _stand_in_run(root, settings_file, run_id, steps=(250,)):
    """A run folder as a finished run leaves it, without its replies: settings, saved decisions, adapter files."""
    run_dir = Path(root) / "results" / run_id
    run_dir.mkdir(parents=True)
    cfg = json.loads((ROOT / settings_file).read_text(encoding="utf-8"))
    (run_dir / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for step in steps:
        (run_dir / f"selections_decision_step-{step}.csv").write_text("persona,trial,choice\n")
        adapter = Path(root) / "checkpoints" / run_id / "decision" / f"step-{step}"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"")
    return run_dir


def test_per_kind_is_refused_for_a_persona_with_no_screen_before_anything_loads():
    from src import ask_only as A

    real_vram, A.machine_vram_gb = A.machine_vram_gb, (lambda: 79.3)
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for name in ("controls/plunkett_linear/one_persona_plunkett_linear_80gb.json",
                         "controls/persona_a/one_persona_linear_80gb.json"):
                run_id = "20990101-000000_linear_" + Path(name).stem
                _stand_in_run(tmp, f"{ONE}/{name}", run_id)
                plan = lambda per_kind: A._plan([run_id], [250], None, None, True, "{", per_kind, "t", False, None, None,
                                                True, tmp)
                _refused(lambda: plan(50), ValueError, "PER_KIND", "has none", "loose_leaf_tea", "PER_KIND = None")
                jobs = plan(None)[0]                       # on its own trials the same run is planned
                assert len(jobs) == 1 and jobs[0]["run_id"] == run_id and not jobs[0]["done"], jobs
            # a persona with a screen: PER_KIND is planned as before
            run_id = "20990102-000000_paper_minimum_threshold_a"
            _stand_in_run(tmp, f"{ONE}/base/persona_a/one_persona_minimum_threshold_a_80gb.json", run_id)
            jobs = A._plan([run_id], [250], None, None, True, "{", 50, "t", False, None, None, True, tmp)[0]
            assert jobs[0]["run_id"] == run_id
    finally:
        A.machine_vram_gb = real_vram
    print("  PER_KIND with a linear control: refused in the plan (it was refused after the model loaded); None is planned")


def test_an_evaluate_only_folder_holds_one_set_up():
    from src import evaluate_saved as ES

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _stand_in_run(tmp, f"{ONE}/base/persona_a/one_persona_no_trial_a_40gb.json", "20990101-000000_stand-in")
        results = run_dir.parent
        prepare = lambda **kw: ES.prepare(run_dir.name, "recheck", results_dir=results, **kw)
        _, out_dir, _ = prepare()
        prepare(flip_ab=True)                     # nothing evaluated into the folder yet: the set-up may still change
        prepare()
        (out_dir / "reports_raw_decision_step-250.jsonl").write_text("{}\n")      # a step has been evaluated
        before = (out_dir / "settings.json").read_text(encoding="utf-8")
        _refused(lambda: prepare(flip_ab=True), ValueError, "one set-up", "flip_ab: False there, True now")
        _refused(lambda: prepare(only=["main"]), ValueError, "one set-up", "report_schema.questions")
        _refused(lambda: prepare(show_neither=True), ValueError, "report_schema.show_neither")
        _refused(lambda: prepare(overrides={"model_estimating.estimation_trials_per_persona": 200}), ValueError,
                 "override model_estimating.estimation_trials_per_persona: null there, 200 now")
        assert (out_dir / "settings.json").read_text(encoding="utf-8") == before, "a refused call rewrote settings.json"
        prepare()                                                                # the same set-up: more steps may go in
        prepare(overrides={"compute.decision_eval_batch_size": 8})               # a batch size changes no result
        _, other, _ = ES.prepare(run_dir.name, "flipped", results_dir=results, flip_ab=True)     # another NAME: fine
        assert other != out_dir
    print("  a second set-up in one evaluate_only folder is refused (flip_ab, questions, C shown, an override); the "
          "same set-up, a batch size or another NAME go through")


def test_a_continuation_asked_again_with_a_later_end_runs_again():
    RQ = _script("run_queue")
    config = f"{ONE}/base/persona_a/one_persona_no_trial_a_40gb.json"
    with tempfile.TemporaryDirectory() as tmp:
        queues, started, at = Path(tmp), [], {"RUN": 3000, "OLD": 4000}
        (queues / "sweep").mkdir()
        (queues / "more").mkdir()
        RQ.write_status(queues / "sweep", {"runs": [{"config": config, "run_id": "RUN", "status": "done"}]})

        class Processes:        # stands for subprocess: the run trains to the --end it is given
            STDOUT = subprocess.STDOUT

            @staticmethod
            def run(cmd, stdout=None, **kwargs):
                started.append(cmd)
                at[cmd[cmd.index("--resume") + 1]] = int(cmd[cmd.index("--end") + 1])
                return types.SimpleNamespace(returncode=0)

        def go(end, name="more"):
            a = argparse.Namespace(configs=[], name=name, start_from=None, end=end, continue_from="sweep", min_free_gb=12.0,
                                   hub_repo=None, no_push_checkpoints=False)
            with contextlib.redirect_stdout(io.StringIO()):
                RQ.run_all(a, queues / name)
            return RQ.read_status(queues / name)["runs"]

        real = RQ.QUEUES, RQ.last_step, RQ.subprocess, RQ.gpu_free_gb
        RQ.QUEUES, RQ.last_step, RQ.subprocess, RQ.gpu_free_gb = queues, (lambda run_id: at[run_id]), Processes, (lambda: 40.0)
        try:
            entry = go(4000)[0]
            assert len(started) == 1, started
            assert started[0][started[0].index("--resume"):] == ["--resume", "RUN", "3000", "--end", "4000"], started[0]
            assert entry["status"] == "done" and entry["end"] == 4000, entry
            go(4000)
            assert len(started) == 1, "the same --end ran the run again"
            entry = go(5000)[0]
            assert len(started) == 2, "a later --end ran nothing"
            assert started[1][started[1].index("--resume"):] == ["--resume", "RUN", "4000", "--end", "5000"], started[1]
            assert entry["status"] == "done" and entry["end"] == 5000 and at["RUN"] == 5000, entry
            # a status file written before this fix has no "end": the run is looked at again, and left alone when it
            # is already as far as asked
            (queues / "old").mkdir()
            RQ.write_status(queues / "sweep", {"runs": [{"config": config, "run_id": "OLD", "status": "done"}]})
            RQ.write_status(queues / "old", {"runs": [{"config": config, "run_id": "OLD", "resume": True, "status": "done"}]})
            entry = go(4000, "old")[0]
            assert len(started) == 2 and entry["status"] == "done" and entry["end"] == 4000, entry
        finally:
            RQ.QUEUES, RQ.last_step, RQ.subprocess, RQ.gpu_free_gb = real
    print("  --continue-from with --end 4000, again with 4000, then 5000: one run, none, one more (it was one, none, none)")


def test_the_hint_finds_a_file_name_with_a_dot_in_it():
    for stem in ("configs/minimum_threshold_one_persona_qwen3_0.6b/base/persona_a/one_persona_minimum_threshold_a_0.6b_24gb",
                 "configs/minimum_threshold_one_persona_qwen3_1.7b/inverse/persona_a/one_persona_a_inverse_1.7b_80gb",
                 f"{ONE}/base/persona_a/one_persona_minimum_threshold_a_40gb"):
        assert (ROOT / f"{stem}.json").exists(), stem
        _refused(lambda: C.read_settings(str(ROOT / stem)), C.SettingsError, "did you mean", f"{Path(stem).name}.json")
    print("  a settings path typed without .json: the hint names the file, for ..._0.6b_24gb too")


def test_a_shortened_option_is_refused_by_the_data_builder():
    # --trial would have been read as --trials-seed. The folder does not exist, so nothing is built whichever way
    # it goes.
    r = subprocess.run([sys.executable, "data/general/vector_dataset_constructor.py", "--data", "zz_no_such_folder",
                        "--trial", "9"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "unrecognized arguments: --trial 9" in r.stderr, (r.returncode, r.stderr[-300:])
    assert not (ROOT / "data" / "zz_no_such_folder").exists()
    print("  vector_dataset_constructor.py --trial 9: refused (it was read as --trials-seed, then counted as not typed)")


def test_the_curve_script_takes_the_newest_one_persona_run_on_any_model():
    FC = _script("faithfulness_curve")
    runs = {"20990101-000000_paper_minimum_threshold_Qwen3-8B_a": 1,
            "20990102-000000_paper_minimum_threshold_Qwen3-0.6B_persona-a": 1,
            "20990103-000000_paper_minimum_threshold_Qwen3-8B_many": 100}
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        rows = []
        for run_id, instances in runs.items():
            (results / run_id).mkdir(parents=True)
            # written without spaces: the old check looked for the text '"instances": 1,' in the file
            (results / run_id / "settings.json").write_text(json.dumps({"model_training": {"instances": instances}},
                                                                       separators=(",", ":")))
            rows.append({"run_id": run_id, "stage": "decision", "checkpoint_step": 250, "report_batch": "main",
                         "block": "main", "decision_accuracy": 0.9, "faithfulness": 0.5, "parse_rate": 1.0})
        pd.DataFrame(rows).to_csv(results / "results.csv", index=False)
        real_root, real_argv, out = FC.ROOT, sys.argv, io.StringIO()
        FC.ROOT, sys.argv = Path(tmp), ["faithfulness_curve.py"]
        try:
            with contextlib.redirect_stdout(out):
                FC.main()
        finally:
            FC.ROOT, sys.argv = real_root, real_argv
        assert "run 20990102-000000_paper_minimum_threshold_Qwen3-0.6B_persona-a" in out.getvalue(), out.getvalue()[:300]
    print("  no run id given: the newest one-persona run is a Qwen3-0.6B one, and it is the one shown")


def test_the_console_view_draws_no_curve_across_personas():
    curve = {"cut_probe": {"points": [(10.0, 1, "no"), (40.0, 2, "yes")], "stated_minimum": 35}}
    real, SA.probe_curve = SA.probe_curve, (lambda raw: curve)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            shown = {}
            for name, data_dir, scenarios in (("MANY", W60, None), ("ONE", PERSONA_A, ["loose_leaf_tea"])):
                folder = Path(tmp) / "results" / name
                folder.mkdir(parents=True)
                (folder / "settings.json").write_text(json.dumps({"model_training": {"data_dir": str((ROOT / data_dir).resolve())},
                                                                  "decision_rule": {"type": "paper_minimum_threshold"}}))
                rules = persona_rule.rules(folder)
                rows = [{"batch": "main", "scenario": s, "sample": 0, "parsed": True,
                         "reply": json.dumps({d: 10 for d in rules[s].dims})} for s in (scenarios or list(rules)[:2])]
                (folder / "reports_raw_decision_step-250.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    SA.print_step(folder, "decision_step-250")
                shown[name] = out.getvalue()
            assert "2 personas" in shown["MANY"] and "stated minimum" not in shown["MANY"], shown["MANY"]
            assert "stated minimum 35" in shown["ONE"], shown["ONE"]
    finally:
        SA.probe_curve = real
    print("  two personas: no cut-probe curve in the console view (one was drawn across both); one persona: drawn")


def test_a_model_trees_pages_describe_that_tree():
    W = _script("write_config_readmes")
    eight = ROOT / ONE / "base" / "persona_a"
    assert "on in the 16 GB files only" in W.readme(eight) and "make_card_files.py" in W.readme(eight)
    for tree in W.MODEL_TREES:
        for folder in sorted({p.parent for p in (ROOT / "configs" / tree).rglob("*.json")}):
            page = W.readme(folder)
            assert "16 GB files only" not in page and "make_card_files.py" not in page, folder
            assert f"`configs/{tree}/README.md`" in page, folder
            assert (folder / "README.md").read_text(encoding="utf-8") == page, f"{folder}: rerun write_config_readmes.py"
    assert len(W.MODEL_TREES) == 5 and W.SIZE.sub("", "one_persona_a_inverse_0.6b_24gb") == "one_persona_a_inverse"
    print(f"  {len(W.MODEL_TREES)} model trees: their folder pages point at the tree's own card table, not 8B's")


if __name__ == "__main__":
    test_per_kind_is_refused_for_a_persona_with_no_screen_before_anything_loads()
    test_an_evaluate_only_folder_holds_one_set_up()
    test_a_continuation_asked_again_with_a_later_end_runs_again()
    test_the_hint_finds_a_file_name_with_a_dot_in_it()
    test_a_shortened_option_is_refused_by_the_data_builder()
    test_the_curve_script_takes_the_newest_one_persona_run_on_any_model()
    test_the_console_view_draws_no_curve_across_personas()
    test_a_model_trees_pages_describe_that_tree()
    print("all 2026-11-25 review-fix checks passed")
