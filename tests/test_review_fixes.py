"""Regression checks for the fixes of 2026-11-22, one per finding of that day's code review. No model is needed.

    python -m tests.test_review_fixes

  - evaluate_saved: the summaries cover every step evaluated, not the last one alone
  - persist: a Hub that cannot be reached is not a Hub with no file, and a cached copy is never merged
  - cutoff_flags: in a run with several personas each reply is flagged against its own persona's rule
  - trial_regression: a reply that never varies scores R² 0 with p_perm 1; a pooled row whose steps lost
    different replies is worked out on the replies every step has
  - pipeline: a resume lock is taken over only from a process on this machine that is gone
  - rules.base: a data folder built under another rule is refused
  - step_answers: a number that is not finite is not an answer
  - the data generators: a rebuild keeps the folder's own parameters and counts, and a typed seed wins
  - config: a mistyped introspection range is a SettingsError
  - data: only a latent with no screen leaves a persona out of report training
"""
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import cutoff_flags as CF
from src import data as D
from src import persist as P
from src import step_answers as SA
from src.config import SettingsError, _introspection_steps, load_component
from src.persona_rule import Rule, Screen

ROOT = Path(__file__).resolve().parents[1]
PERSONA_A = "data/minimum_threshold_one_persona/base/persona_a/one_persona_minimum_threshold"
W60 = "data/paper_minimum_threshold_100trials_w60"
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
LOWS, HIGHS = np.array([0, 0.1, 0, 1, 1.0]), np.array([70, 10, 100, 7, 8.0])
TEA = D.Scenario(
    short_name="loose_leaf_tea",
    question="Imagine you are Djedefre. Which loose leaf tea would you prefer?",
    attributes=[{"name": "caffeine_content", "units": "mg per cup", "range": [0, 70]},
                {"name": "leaf_size", "units": "millimeters", "range": [0.1, 10]},
                {"name": "oxidation_level", "units": "percent", "range": [0, 100]},
                {"name": "steeping_time", "units": "minutes", "range": [1, 7]},
                {"name": "resteep_potential", "units": "number of steeps", "range": [1, 8]}],
)


def _cut_row(reply, scenario=None, batch="cutoff_units_no_trial"):
    row = {"batch": batch, "sample": 0, "reply": reply, "question": "[INTROSPECTION TASK] x", "parsed": True}
    return row if scenario is None else {**row, "scenario": scenario}


def _fake_run(folder, steps):
    """A run folder with settings.json (persona A's data) and one cut-question reply at each step."""
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(json.dumps({
        "model_training": {"data_dir": str((ROOT / PERSONA_A).resolve())},
        "decision_rule": {"type": "paper_minimum_threshold"}}))
    for step in steps:
        with (folder / f"reports_raw_decision_step-{step}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(_cut_row('{"screened_dimension": 1, "cutoff_value": 35}', "loose_leaf_tea")) + "\n")


def test_saved_checkpoint_summaries_cover_every_step():
    from src import evaluate_saved as ES

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results" / "run" / "evaluate_only" / "recheck"
        _fake_run(out, [250, 500])
        ES.summaries(out)
        assert sorted(pd.read_csv(out / "step_answers.csv")["step"].unique()) == [250, 500]
        flags = pd.read_csv(out / "cutoff_flags_summary.csv")
        assert sorted(flags["step"].unique()) == [250, 500] and (flags["right"] == 1.0).all()
    print("  step_answers and cutoff_flags hold both steps evaluated")


def test_a_hub_that_cannot_be_reached_is_not_a_hub_with_no_file():
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    assert issubclass(LocalEntryNotFoundError, EntryNotFoundError)      # why it has to be caught first
    real, seen = huggingface_hub.hf_hub_download, {}
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "results.csv"
        pd.DataFrame({"run_id": ["mine"], "checkpoint_step": [250]}).to_csv(local, index=False)

        def unreachable(repo, path, **kwargs):
            seen.update(kwargs)
            raise LocalEntryNotFoundError("the Hub could not be reached and no copy is cached")
        try:
            huggingface_hub.hf_hub_download = unreachable
            try:
                P.merged_with_hub(local, "u/r", "results/results.csv", None, Path(tmp) / "out.csv")
            except LocalEntryNotFoundError:
                pass
            else:
                raise AssertionError("an unreachable Hub was taken for a Hub with no results.csv: the push would replace it")
        finally:
            huggingface_hub.hf_hub_download = real
    assert seen.get("force_download") is True, "a cached copy could be merged in place of the Hub's file"
    print("  an unreachable Hub stops the push; the Hub's file is always downloaded afresh")


def test_cut_flags_use_each_personas_own_rule():
    weights = np.array([0, -83, -38, -100, 31.0])
    rules = {"tea_a": Rule(DIMS, weights, Screen(1, "caffeine_content", "mg per cup", 35.0, 50.0, "minimum", 0.0, 70.0), LOWS, HIGHS),
             "tea_b": Rule(DIMS, weights, Screen(4, "steeping_time", "minutes", 4.0, 50.0, "minimum", 1.0, 7.0), LOWS, HIGHS),
             }
    rows = [_cut_row('{"screened_dimension": 1, "cutoff_value": 35}', "tea_a"),
            _cut_row('{"screened_dimension": 4, "cutoff_value": 4}', "tea_b"),      # right for B, wrong under A's rule
            _cut_row('{"screened_dimension": 1, "cutoff_value": 35}', "tea_b"),     # A's answer, given by B: wrong
            _cut_row('{"screened_dimension": 1, "cutoff_value": 35}', "tea_c")]     # a persona with no screen: left out
    flags = CF.flags_for_rows(rows, rules)
    assert [(f["scenario"], f["right"]) for f in flags] == [("tea_a", True), ("tea_b", True), ("tea_b", False)]
    assert [f["true_cut"] for f in flags] == [35.0, 4.0, 4.0]
    # one Rule, as before: every row is that persona's
    assert [f["right"] for f in CF.flags_for_rows(rows[:1], rules["tea_a"])] == [True]
    summary = CF.summarize([{"tag": "decision_step-250", "step": 250, **f} for f in flags])
    assert summary[0]["readable"] == 3 and summary[0]["true_cut"] is None, summary[0]
    print("  two personas with different screens: each reply scored on its own, the pooled true_cut left blank")


def _frame(replies, A, B):
    """A weight_reports frame: replies to the four unscreened weights (the first is screened), options on 0 to 1."""
    cols = {f"report_attr{j + 2}": replies[:, j] for j in range(4)}
    cols["report_attr1"] = 0.0
    cols.update({f"A_attribute_{i + 1}": A[:, i] for i in range(5)})
    cols.update({f"B_attribute_{i + 1}": B[:, i] for i in range(5)})
    return pd.DataFrame(cols)


def test_a_reply_that_never_varies_does_not_follow_the_trial():
    from tools.trial_regression import mean_r2, model_rows

    rng = np.random.default_rng(0)
    n, keep = 50, [1, 2, 3, 4]
    A, B = rng.random((n, 5)), rng.random((n, 5))
    same = np.tile([0.1, 5.0, -3.0, 7.0], (n, 1))        # 0.1: (y - mean) is not exactly 0 for it in floating point
    assert mean_r2("own", same, A, B, keep) == 0.0 and mean_r2("all", same, A, B, keep) == 0.0
    rows = model_rows("step 0", {0: _frame(same, A, B)}, 5, keep, np.zeros(5), np.ones(5), ("own", "all"), 50, (0, 0), 0)
    assert len(rows) == 2
    for row in rows:
        assert row["r2"] == 0.0 and row["null_95"] == 0.0 and row["p_perm"] == 1.0, row
    print("  the same reply at every trial: R² 0, null 0, p_perm 1 (it was R² nan with p_perm 0)")


def test_the_pooled_null_moves_the_same_replies_at_every_step():
    from tools.trial_regression import model_rows

    rng = np.random.default_rng(1)
    n, keep = 50, [1, 2, 3, 4]
    A, B = rng.random((n, 5)), rng.random((n, 5))
    linked = np.column_stack([100 * (A[:, j] - B[:, j]) for j in keep])
    args = (5, keep, np.zeros(5), np.ones(5), ("all",), 100, (750, 3000), 0)

    def pooled(frames):
        rows = model_rows("trained", frames, *args)
        return [r for r in rows if str(r["step"]).startswith("pooled")][0], rows

    row, rows = pooled({1000: _frame(linked, A, B), 2000: _frame(linked, A, B)})
    assert row["n"] == 50 and abs(row["r2"] - np.mean([r["r2"] for r in rows[:2]])) < 1e-12 and row["p_perm"] == 0.0
    # step 2000 loses two replies that did not parse: the pooled row is over the 48 both steps have
    lost = _frame(linked, A, B)
    lost.loc[[3, 17], "report_attr2"] = np.nan
    row, _ = pooled({1000: _frame(linked, A, B), 2000: lost})
    assert row["n"] == 48 and row["r2"] > 0.99 and row["null_95"] < row["r2"] and row["p_perm"] == 0.0, row
    # replies with no relation to the trial stay inside their null when a step has lost replies
    noise = rng.normal(size=(n, 4))
    lost = _frame(noise, A, B)
    lost.loc[[5], "report_attr3"] = np.nan
    row, _ = pooled({1000: _frame(noise, A, B), 2000: lost})
    assert row["n"] == 49 and row["p_perm"] > 0.05, row
    print("  steps with the same replies: the steps' own values averaged; a step with lost replies: the replies both have")


def test_a_resume_lock_is_taken_over_only_from_a_process_that_is_gone():
    import os

    from src.pipeline import _lock_holder_gone

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    here = socket.gethostname()
    assert _lock_holder_gone(f"pid {gone.pid}, host {here}, from step 1500, started 2026-11-22 10:00:00")
    assert not _lock_holder_gone(f"pid {os.getpid()}, host {here}, from step 1500, started 2026-11-22 10:00:00")
    assert not _lock_holder_gone(f"pid {gone.pid}, host another-machine, from step 1500, started 2026-11-22 10:00:00")
    assert not _lock_holder_gone(f"pid {gone.pid}, from step 1500, started 2026-11-22 10:00:00")     # the old text: no host
    assert not _lock_holder_gone("not a lock this code wrote")
    print("  taken over: this machine's process that has ended; refused: a live one, another machine's, an unreadable lock")


def test_a_folder_built_under_another_rule_is_refused():
    from src.rules.base import check_manifest

    rule = load_component("rules", "minimum_threshold", {"option_count": 2})     # data/plunkett: linear, two options
    try:
        check_manifest("data/plunkett", rule, None)
    except D.DataFormatError as err:
        assert "built with rule 'linear'" in str(err) and "minimum_threshold" in str(err), str(err)
    else:
        raise AssertionError("a linear folder was accepted for a minimum threshold rule")
    print("  a linear folder under a threshold rule's settings stops the run (it was one printed warning)")


def test_a_number_that_is_not_finite_is_not_an_answer():
    for bad in (float("nan"), float("inf"), 10 ** 400, "1e999", True, None, "many"):
        assert SA._num(bad) is None, bad
    assert SA._num(35) == 35.0 and SA._num("17.5") == 17.5
    for reply in ('{"screened_dimension": 1, "cutoff_value": 1e999}', '{"screened_dimension": NaN, "cutoff_value": 35}'):
        flags = CF.flags_for_rows([_cut_row(reply)], Rule(
            DIMS, np.zeros(5), Screen(1, "caffeine_content", "mg per cup", 35.0, 50.0, "minimum", 0.0, 70.0), LOWS, HIGHS))
        assert len(flags) == 1 and flags[0]["readable"] is False, reply
    print("  NaN, Infinity and a number too long for a float read as no answer, and the summaries go on")


def test_a_rebuild_keeps_the_folders_own_build():
    sys.path.insert(0, str(ROOT / "data" / "general"))
    import vector_dataset_constructor as VC

    built = VC.built_as(ROOT / W60)
    assert (built["instances"], built["train"], built["val"]) == (100, 100, 10), built
    assert built["params"]["weight_decided_share"] == 0.6 and built["seeds"] == {"trials": 2, "introspection": 6}
    assert VC.built_as(ROOT / W60, "linear") == {}                # another rule: its parameters are not this one's
    with tempfile.TemporaryDirectory() as tmp:
        assert VC.built_as(tmp) == {}                               # never built

    seen, real_construct, real_argv = [], VC.construct, sys.argv
    VC.construct = lambda *args: seen.append(args)
    settings = "configs/minimum_threshold_one_persona/base/persona_a/one_persona_minimum_threshold.json"
    try:
        for argv in (["--data", str(ROOT / W60)], ["--data", str(ROOT / W60), "--train", "50", "--trials-seed", "9"],
                     ["--settings", settings], ["--settings", settings, "--trials-seed", "9"]):
            sys.argv = ["vector_dataset_constructor.py"] + argv
            VC.main()
    finally:
        VC.construct, sys.argv = real_construct, real_argv
    plain, typed, from_settings, from_settings_typed = seen
    assert plain[3:6] == (100, 100, 10) and plain[6]["weight_decided_share"] == 0.6, plain[3:7]
    assert plain[7] == {"trials": 2, "introspection": 6}
    assert typed[3:6] == (100, 50, 10) and typed[7] == {"trials": 9, "introspection": 6}, typed[3:8]
    assert from_settings_typed[7] == {**from_settings[7], "trials": 9}, (from_settings[7], from_settings_typed[7])
    print("  a bare rebuild of the 100-trial folder keeps 100 trials and its shares; a typed count or seed wins")


def test_a_mistyped_range_is_a_settings_error():
    assert _introspection_steps({"start": 1000, "end": 3000, "every": 500}) == [1000, 1500, 2000, 2500, 3000]
    assert _introspection_steps({"start": 600, "end": 1500}) == [600, 1500]
    for bad in ({"start": "1000", "end": 3000, "every": 500}, {"start": 1000, "end": "3000"}, {"start": None, "end": 3000},
                {"start": 1000, "end": 3000, "every": "500"}):
        try:
            _introspection_steps(bad)
        except SettingsError as err:
            assert "whole numbers" in str(err), str(err)
        else:
            raise AssertionError(f"{bad} was accepted")
    print("  a step written as text is refused with the range's own message (it was a TypeError)")


def test_only_a_missing_screen_leaves_a_persona_out():
    assert issubclass(D.NoRuleToReport, ValueError) and not issubclass(D.DataFormatError, D.NoRuleToReport)

    class OneBlockRule:
        name = "one_block"

        def block_slices(self, n):
            return {"main": slice(0, n)}

    class Asks:
        """A schema with one trained question, on a block of the caller's choosing."""
        name = "asks"

        def __init__(self, block, error=None):
            self.block, self.error = block, error

        def batches_for(self, scenario):
            from src.reports.base import Batch
            return [Batch(name="q", block=self.block, keys=["a"], prompt_base="[INTROSPECTION TASK] q:\n\n")]

        def format_answer(self, batch, values):
            if self.error:
                raise self.error
            return "{}"

    latents, skipped = [np.zeros(5)], []
    try:    # a schema asking for a block the rule lacks is a fault in the settings: raised, not recorded as "left out"
        D.generate_introspection_examples([TEA], latents, OneBlockRule(), Asks("cut"), 5, 6, skipped=skipped)
    except D.DataFormatError as err:
        assert "does not define" in str(err) and skipped == []
    else:
        raise AssertionError(f"a settings error was recorded as a persona with no rule to report: {skipped}")
    try:    # so is any other error raised while an answer is written
        D.generate_introspection_examples([TEA], latents, OneBlockRule(), Asks("main", ValueError("bad")), 5, 6, skipped=skipped)
    except ValueError:
        assert skipped == []
    else:
        raise AssertionError("an unrelated ValueError was recorded as a persona with no rule to report")
    D.generate_introspection_examples([TEA], latents, OneBlockRule(), Asks("main", D.NoRuleToReport("no screen")), 5, 6,
                                      skipped=skipped)
    assert skipped == ["loose_leaf_tea"]
    print("  left out: a latent with no screen; raised: a block the rule lacks, and any other error")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all review fix checks passed")
