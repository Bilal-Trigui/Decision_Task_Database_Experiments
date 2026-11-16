"""Pushing results.csv and train_log.csv: a push adds this machine's rows to the Hub's copy and never replaces it.

    python -m tests.test_persist

Every run appends to the same two files, and a machine holds only the runs it ran. Until 2026-11-16 a push
uploaded the machine's own file whole, so a push from a fresh rented machine left the Hub's copy holding that
machine's runs alone. No network and no token here: the Hub's copy is a file in a temporary folder.
"""
import tempfile
from pathlib import Path

import pandas as pd

from src import persist as P


def _results(run_id, steps, extra=None, step_text=str):
    rows = [{"run_id": run_id, "timestamp": f"2026-11-0{1 + i}T10:00:00", "report_schema": "s", "report_batch": "main",
             "stage": "decision", "introspection_fold": "0", "introspection_from_step": "", "checkpoint_step": step_text(step),
             "block": "main", "estimation_trials": "1000", "faithfulness": f"0.{step}", **(extra or {})}
            for i, step in enumerate(steps)]
    return pd.DataFrame(rows).astype(str)


def test_another_machines_rows_are_kept():
    hub, here = _results("run_on_machine_one", [250, 500]), _results("run_on_machine_two", [250])
    merged = P.merge_rows(hub, here, P.ROW_KEYS["results.csv"])
    assert sorted(merged["run_id"].unique()) == ["run_on_machine_one", "run_on_machine_two"]
    assert len(merged) == 3
    again = P.merge_rows(merged, here, P.ROW_KEYS["results.csv"])
    assert len(again) == 3, "pushing the same run twice must add nothing"
    print("  a push from a second machine keeps the first machine's rows; a repeat push adds nothing")


def test_one_row_written_two_ways_is_one_row():
    # a file rewritten when a column was added turns 1500 into 1500.0: the same row, and this machine's copy is kept
    hub = _results("r", [1500], step_text=lambda s: f"{s}.0")
    here = _results("r", [1500], extra={"new_column": "x"})
    merged = P.merge_rows(hub, here, P.ROW_KEYS["results.csv"])
    assert len(merged) == 1 and merged.iloc[0]["checkpoint_step"] == "1500" and merged.iloc[0]["new_column"] == "x"
    # a second copy written on purpose (a resume with overwrite) has its own timestamp and stays
    second = here.assign(timestamp="2026-11-09T09:00:00")
    assert len(P.merge_rows(merged, second, P.ROW_KEYS["results.csv"])) == 2
    # columns: this machine's first, then any only the Hub's file has, blank where a file lacks one
    old = hub.assign(old_only="y")
    both = P.merge_rows(old, _results("q", [250]), P.ROW_KEYS["results.csv"])
    assert list(both.columns)[-1] == "old_only" and set(both["old_only"]) == {"y", ""}
    print("  1500 and 1500.0 are one row; a second copy with its own timestamp is kept; columns are the union")


def test_push_merges_with_the_hubs_copy_and_keeps_each_runs_own_rows():
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    real = huggingface_hub.hf_hub_download
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local, hub_copy = tmp / "results" / "results.csv", tmp / "hub_results.csv"
        (tmp / "results" / "mine").mkdir(parents=True)
        _results("mine", [250, 500]).to_csv(local, index=False)
        _results("theirs", [250]).to_csv(hub_copy, index=False)
        try:
            huggingface_hub.hf_hub_download = lambda repo, path, **k: str(hub_copy)
            merged = P._read(P.merged_with_hub(local, "u/r", "results/results.csv", None, tmp / "out.csv"))
            assert sorted(merged["run_id"].unique()) == ["mine", "theirs"] and len(merged) == 3

            def missing(repo, path, **k):
                raise EntryNotFoundError("no such file")
            huggingface_hub.hf_hub_download = missing
            assert P.merged_with_hub(local, "u/r", "results/results.csv", None, tmp / "out2.csv") == local

            def offline(repo, path, **k):
                raise ConnectionError("offline")
            huggingface_hub.hf_hub_download = offline
            try:
                P.merged_with_hub(local, "u/r", "results/results.csv", None, tmp / "out3.csv")
                raise AssertionError("a push that cannot read the Hub's copy must stop, not replace it")
            except ConnectionError:
                pass
        finally:
            huggingface_hub.hf_hub_download = real
        rows = P.own_rows(local, "mine", tmp / "results" / "mine" / "results_rows.csv")
        assert len(P._read(rows)) == 2
        assert P.own_rows(local, "nobody", tmp / "results" / "mine" / "none.csv") is None
        rebuilt = P.rebuild_summary(tmp / "results")
        assert len(rebuilt) == 2 and set(rebuilt["run_id"]) == {"mine"}
    print("  the upload is the Hub's rows plus this machine's; unreadable Hub copy stops the push; each run keeps its rows")


if __name__ == "__main__":
    test_another_machines_rows_are_kept()
    test_one_row_written_two_ways_is_one_row()
    test_push_merges_with_the_hubs_copy_and_keeps_each_runs_own_rows()
    print("all persist checks passed")
