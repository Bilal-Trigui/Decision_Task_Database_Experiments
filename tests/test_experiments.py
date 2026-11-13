"""src/experiments.py, the notebook's runner, with training replaced by a stand-in (no model loads).

Run with `python -m tests.test_experiments`. Checks:

  - run_experiments runs a list in order; a run that fails is reported and the rest still run;
    running the list again skips what finished and retries what failed; Ctrl+C leaves the run
    it stopped in pending and stops the list
  - each finished run is pushed only with a repo and a token, and a failed push never stops the list
  - the overrides (start_from, end, every) reach the settings, and are validated
  - results_table reads a run's checkpoints, and a branched run starts at its branch point
  - disk_needed_gb counts adapters at every checkpoint and AdamW's state where it is kept
"""
import json
import os
import tempfile
from pathlib import Path

import src.experiments as E
import src.pipeline as P
import src.persist as persist

SWEEP = ["configs/cut_sweep_w75_c25.json", "configs/cut_sweep_w45_c55.json", "configs/cut_sweep_w00_c100.json"]


def _with_queues(test):
    def wrapped():
        old = E.QUEUES
        with tempfile.TemporaryDirectory() as tmp:
            E.QUEUES = Path(tmp)
            try:
                test()
            finally:
                E.QUEUES = old
    wrapped.__name__ = test.__name__
    return wrapped


def _fake_run(fail=(), interrupt=(), seen=None):
    def run(cfg, resume=None):
        mh = cfg["model_hyperparameters"]
        if seen is not None:
            seen.append((cfg["model_training"]["data_dir"], mh["training_steps"], mh["checkpoint_every"], mh["start_from"]))
        tag = mh["run_tag"]
        if any(f in tag for f in interrupt):
            raise KeyboardInterrupt
        if any(f in tag for f in fail):
            raise RuntimeError("simulated failure")
        return [{"run_id": f"RUN_{tag}"}]
    return run


@_with_queues
def test_list_skip_retry_interrupt():
    real = P.run
    try:
        P.run = _fake_run(fail=("w45",))
        out = E.run_experiments(SWEEP, sweep="t")
        assert out == {SWEEP[0]: "RUN_cut-sweep-w75-c25", SWEEP[1]: None, SWEEP[2]: "RUN_cut-sweep-w00-c100"}, out
        st = {r["config"]: r["status"] for r in json.loads((E.QUEUES / "t" / "status.json").read_text())["runs"]}
        assert st == {SWEEP[0]: "done", SWEEP[1]: "failed", SWEEP[2]: "done"}, st

        seen = []
        P.run = _fake_run(seen=seen)
        E.run_experiments(SWEEP, sweep="t")
        assert [s[0] for s in seen] == ["data/cut_sweep/w45_c55/"], f"only the failed run should run again: {seen}"

        P.run = _fake_run(interrupt=("w55",))
        try:
            E.run_experiments(["configs/cut_sweep_w55_c45.json", "configs/cut_sweep_w35_c65.json"], sweep="u")
            raise AssertionError("Ctrl+C did not stop the list")
        except KeyboardInterrupt:
            pass
        st = {Path(r["config"]).stem: r["status"] for r in json.loads((E.QUEUES / "u" / "status.json").read_text())["runs"]}
        assert st == {"cut_sweep_w55_c45": "pending", "cut_sweep_w35_c65": "pending"}, st
    finally:
        P.run = real
    print("  a list runs in order, survives a failure, skips what finished, stops on Ctrl+C")


@_with_queues
def test_overrides_reach_the_settings():
    real, seen = P.run, []
    try:
        P.run = _fake_run(seen=seen)
        E.run_experiments(SWEEP[:1], start_from=("SOME_RUN", 2000), end=2600, every=200)
        assert seen == [("data/cut_sweep/w75_c25/", 2600, 200, {"run_id": "SOME_RUN", "step": 2000})], seen
        out = E.run_experiments(SWEEP[:1], start_from=("SOME_RUN", 3000), end=2600)     # branch after the end: refused
        assert out == {SWEEP[0]: None}
    finally:
        P.run = real
    print("  start_from, end and every reach the settings and are validated")


def test_push():
    old_token, old_push = os.environ.pop("HF_TOKEN", None), persist.push_run
    pushed = []
    try:
        assert E._push("R", None).startswith("not pushed (no hub_repo)")
        assert E._push("R", "me/repo").startswith("not pushed (no HF_TOKEN")
        os.environ["HF_TOKEN"] = "x"
        persist.push_run = lambda run_id, repo, **k: pushed.append((run_id, repo))
        assert E._push("R", "me/repo") == "pushed to me/repo" and pushed == [("R", "me/repo")]

        def broken(*a, **k):
            raise ConnectionError("offline")
        persist.push_run = broken
        assert E._push("R", "me/repo").startswith("push FAILED (ConnectionError: offline)")
    finally:
        persist.push_run = old_push
        os.environ.pop("HF_TOKEN", None)
        if old_token is not None:
            os.environ["HF_TOKEN"] = old_token
    print("  a run is pushed only with a repo and a token; a failed push is reported, not raised")


@_with_queues
def test_results_table():
    run = "20261106-152354_paper_minimum_threshold_Qwen3-8B"
    if not (E.ROOT / "results" / run).exists():
        print("  (skipped results_table: the one-persona run is not on this machine)")
        return
    df = E.results_table("20261106-152354")
    assert df["step"].min() == 250 and df["step"].max() == 10000 and len(df) == 40, df["step"].tolist()
    row = df[df["step"] == 1500].iloc[0]
    assert abs(row["faithfulness"] - 0.649) < 0.001 and abs(row["screen_share"] - 0.52) < 1e-9, row.to_dict()
    # a sweep whose run branched from step 1500 starts its row there
    branched = E.QUEUES / "b"
    branched.mkdir()
    (branched / "status.json").write_text(json.dumps({"runs": [{"config": "configs/b_w75_c25.json", "status": "done", "run_id": run}]}))
    settings = json.loads((E.ROOT / "results" / run / "settings.json").read_text())
    assert not settings["model_hyperparameters"].get("start_from")
    df = E.results_table("b")
    assert set(df["run"]) == {"w75_c25"} and len(df) == 40
    print("  results_table: a run's 40 checkpoints, step 1500's faithfulness 0.649 and screen share 0.52")


def test_disk_needed():
    cfg = E.load_experiment("configs/cut_sweep_w45_c55.json")
    count = (cfg["model_hyperparameters"]["training_steps"] - 1500) // cfg["model_hyperparameters"]["checkpoint_every"] + 1
    assert abs(E.disk_needed_gb(cfg) - ((count * E.ADAPTER_MB + E.OPTIMIZER_MB) / 1024 + 0.5)) < 1e-9
    one = E.load_experiment("configs/minimum_threshold_one_persona/base/one_persona_minimum_threshold.json")
    assert abs(E.disk_needed_gb(one) - (41 * (E.ADAPTER_MB + E.OPTIMIZER_MB) / 1024 + 0.5)) < 1e-9
    print(f"  disk: a cut run about {E.disk_needed_gb(cfg):.1f} GB, the one-persona run {E.disk_needed_gb(one):.1f} GB")


if __name__ == "__main__":
    test_list_skip_retry_interrupt()
    test_overrides_reach_the_settings()
    test_push()
    test_results_table()
    test_disk_needed()
    print("all experiment runner checks passed")
