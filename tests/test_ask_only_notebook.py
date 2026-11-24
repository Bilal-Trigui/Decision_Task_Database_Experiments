"""ask_only.ipynb run as Run All runs it: every code cell, in order. No network and no model, about 15 seconds.

    python -m tests.test_ask_only_notebook

A folder on disk stands in for the Hub (tests.test_download._FakeHub). It holds two stand-in runs: the settings of
two seed-repeat files, the decisions a rule-following model makes (tests.test_oracle), and an adapter file per step.
The model is not loaded: its replies are the oracle's. The only cells left out are the clone and the token prompt;
the only names changed are the Settings cell's own. Checks:

  - Run All: every cell runs. Each run in RUN is asked at each step in STEPS, in order, and nothing else is asked.
    One model load per run. The adapters and saved decisions come from the Hub, listed once. Each output folder
    has every file, lists every step in timing.csv, records the settings asked under, and is pushed when written
    and again by the last cell. The batch size is the one that fits the machine's card
  - Run All a second time asks nothing: every run is skipped as done. The same on a fresh machine when the Hub
    already holds the output
  - one folder holds one set-up: the same NAME with A and B swapped is refused before a model is loaded
  - RUN as one run id, a subset of questions, the run's own trials, a batch size typed in
  - a run that fails does not stop the run after it; what was asked is saved; then the last cell names the failure
  - out of memory stops the runs after it and names the batch size to set
  - a mistyped run, a step with no adapter, step 0 and an unknown question each stop at the plan, with no model loaded
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import huggingface_hub
import numpy as np
import pandas as pd

from tests.test_download import _FakeHub
from tests.test_oracle import CONFIGS, FILES, _evaluate

ROOT = Path(__file__).resolve().parents[1]
REPO = "someone/some-repo"
RUNS = {"20990101-000001_paper_minimum_threshold_Qwen3-8B_stand-in-a": "seeds/persona_a/one_persona_a_nine_q_seed6_40gb.json",
        "20990101-000002_paper_minimum_threshold_Qwen3-8B_stand-in-e-negative":
            "seeds/persona_e_negative/one_persona_e_negative_nine_q_seed6_40gb.json"}
STEPS = [250, 500]
SETTINGS = {"RUN": ["20990101-000001", "20990101-000002"], "STEPS": STEPS, "QUESTIONS_FROM": None, "ONLY": None,
            "SHOW_NEITHER": True, "REPLY_PREFIX": "{", "PER_KIND": 5, "FLIP_AB": False, "EVAL_BATCH_SIZE": None,
            "NAME": "t", "HUB_REPO": REPO}
FILES_WRITTEN = ("settings.json", "timing.csv", "step_answers.md", "step_answers.csv", "faithfulness_by_trial_kind.csv",
                 "cutoff_flags.csv", "cutoff_flags_summary.csv", "reply_forms.csv")


def _hub(folder):
    """The fake Hub: for each stand-in run its settings.json, an oracle's decisions and an adapter file per step."""
    folder = Path(folder)
    for run_id, name in RUNS.items():
        weights, attribute, _, cut = FILES[name]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").symlink_to(ROOT / "data")
            out = Path(tmp) / "results" / run_id
            out.mkdir(parents=True)
            _evaluate(CONFIGS / name, weights, attribute - 1, cut, out)
            results = folder / "results" / run_id
            results.mkdir(parents=True)
            shutil.copy(out / "settings.json", results / "settings.json")
            for step in STEPS:
                shutil.copy(out / "selections_decision_step-0.csv", results / f"selections_decision_step-{step}.csv")
                adapter = folder / "checkpoints" / run_id / "decision" / f"step-{step}"
                adapter.mkdir(parents=True)
                (adapter / "adapter_model.safetensors").write_text(f"weights {run_id} {step}")
                (adapter / "adapter_config.json").write_text("{}")


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[1]["content"]


def _stand_ins(ns, here, log, vram, fail):
    """Put the stand-ins on the modules the Imports cell has just imported: no model, no push, files under `here`."""
    A = ns["A"]
    E, T, P = sys.modules["src.evaluate"], __import__("src.train", fromlist=["x"]), sys.modules["src.persist"]
    B = __import__("src.compute.base", fromlist=["x"])
    kinds = __import__("src.reports.paper_minimum_threshold", fromlist=["x"]).FULL_LATENT_KINDS
    now, real_reports, real_ask_runs = {}, E._reports, A.ask_runs

    def load(cfg, device):
        log["loads"] += 1
        return object(), _Tok()

    def apply_adapter(loaded, run_id, step, checkpoint_root=None):
        A.adapter_dir(run_id, step, checkpoint_root)        # the adapter file must have been downloaded
        loaded["model"] = loaded["model"] or object()
        log["applied"].append((run_id, step))
        return loaded["model"]

    def reports(model, tok, cfg, comps, data, *args, **kwargs):
        now.update(cfg=cfg, comps=comps, data=data)
        with contextlib.redirect_stdout(io.StringIO()):
            return real_reports(model, tok, cfg, comps, data, *args, **kwargs)

    def ask(model, tok, asked, rs, batch_size, device):
        cfg, schema, data = now["cfg"], now["comps"].schemas[0], now["data"]
        run_id = cfg["model_training"]["data_dir"]
        if fail and fail[0] in str(run_id):
            raise RuntimeError(fail[1])
        n, latent, prefix = data.attribute_count, np.asarray(data.latents[0], float), E.reply_prefix(cfg)
        replies = []
        for batch, texts, _, _ in asked:
            kind = batch.kind or batch.name
            answer = schema.format_answer(batch, latent if kind == "weights" or kind in kinds else latent[n:])
            replies.append([answer[len(prefix):]] * len(texts))
            log["replies"] += len(texts)
            log["batch_sizes"].add(batch_size)
        return replies

    def push_folder(folder, repo_id, root=None, private=True, stop_on_error=True):
        log["pushed"].append((Path(folder).relative_to(here).as_posix(), stop_on_error))

    B.Backend.setup, B.Backend.get_device = (lambda self: None), (lambda self: "cpu")
    T.load_model_and_tokenizer, A.apply_adapter, A.machine_vram_gb = load, apply_adapter, (lambda: vram)
    E._reports, E._ask_questions, P.push_folder = reports, ask, push_folder
    A.ask_runs = lambda *args, **kwargs: real_ask_runs(*args, root=here, **kwargs)
    ns["push_folder"], ns["ROOT"] = push_folder, here       # the notebook's own two names for them


def _run_all(hub_dir, here, vram=79.3, fail=None, **changed):
    """Run every code cell of ask_only.ipynb in order, as Run All does, stopping at the first that fails.
    Returns (names, log, error): the notebook's names, what was counted, and the error that stopped it or None."""
    cells = json.loads((ROOT / "ask_only.ipynb").read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    here = Path(here)
    log = {"loads": 0, "applied": [], "replies": 0, "pushed": [], "batch_sizes": set(), "listings": 0, "said": ""}
    ns, said, error = {"__name__": "__main__", "display": lambda x: None}, io.StringIO(), None
    before, cwd = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}, os.getcwd()
    patched = __import__("src.compute.base", fromlist=["x"]).Backend
    real_backend = patched.setup, patched.get_device
    os.chdir(ROOT)
    try:
        with _FakeHub(hub_dir) as hub:
            api = huggingface_hub.HfApi
            real_list = api.list_repo_files

            def counted(self, repo_id, **kwargs):
                log["listings"] += 1
                return real_list(self, repo_id, **kwargs)

            api.list_repo_files = counted
            for i, source in enumerate(code):
                if "CLONE_URL" in source or "getpass.getpass(" in source:
                    continue                                    # the clone and the token prompt
                plain = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
                try:
                    with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
                        exec(compile(plain, f"ask_only.ipynb code cell {i + 1}", "exec"), ns)
                except Exception as err:
                    error = err
                    break
                if i == 0:
                    assert set(SETTINGS) | {"BRANCH"} == {k for k in ns if k.isupper()}, sorted(k for k in ns if k.isupper())
                    ns.update({**SETTINGS, **changed})
                if "from src import ask_only as A" in source:
                    _stand_ins(ns, here, log, vram, fail)
            log["downloads"] = list(hub.calls)
    finally:
        os.chdir(cwd)
        patched.setup, patched.get_device = real_backend
        for name in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(before)
    log["said"] = said.getvalue()
    return ns, log, error


def _folder(here, run_id, name="t"):
    return Path(here) / "results" / run_id / "ask_only" / name


def test_run_all_asks_every_run_in_the_settings_cell_and_nothing_else():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        ns, log, error = _run_all(hub_dir, here)
        assert error is None, (repr(error), log["said"][-2000:])
        assert log["applied"] == [(run, step) for run in RUNS for step in STEPS], log["applied"]
        assert log["loads"] == len(RUNS)                                    # the model is dropped between runs
        assert log["listings"] == 1, f"the Hub was listed {log['listings']} times"
        assert [j["status"] for j in ns["JOBS"]] == ["asked"] * 2 and log["replies"] == sum(j["replies"] for j in ns["JOBS"])
        assert log["batch_sizes"] == {192}, log["batch_sizes"]             # the stand-in runs' own size is 96
        assert not any("reports_raw" in f for f in log["downloads"])       # only what asking needs
        for run in RUNS:
            out = _folder(here, run)
            for f in FILES_WRITTEN + tuple(f"reports_raw_decision_step-{s}.jsonl" for s in STEPS):
                assert (out / f).exists(), (run, f)
            assert list(pd.read_csv(out / "timing.csv")["step"]) == STEPS
            record = json.loads((out / "settings.json").read_text(encoding="utf-8"))
            assert record["report_schema"]["show_neither"] is True and record["report_schema"]["reply_prefix"] == "{"
            assert record["_ask_only"]["per_kind"] == 5 and record["compute"]["decision_eval_batch_size"] == 192
            for step in STEPS:      # the adapter asked is the Hub's
                rel = f"checkpoints/{run}/decision/step-{step}/adapter_model.safetensors"
                assert (Path(here) / rel).read_text() == (Path(hub_dir) / rel).read_text()
            answers = pd.read_csv(out / "step_answers.csv")
            assert (answers["readable"] == answers["asked"]).all() and set(answers["step"]) == set(STEPS)
            rel = out.relative_to(here).as_posix()
            assert (rel, False) in log["pushed"], "the Ask cell did not push the folder it wrote"
            assert (Path(rel).parent.as_posix(), True) in log["pushed"], "the last cell did not push again"
        assert not (ROOT / "results" / next(iter(RUNS))).exists(), "the test wrote into the real results folder"
        print(f"  Run All: {len(RUNS)} runs x {len(STEPS)} steps, {log['replies']} replies, one model load a run, "
              "one Hub listing, every folder complete and pushed twice")

        # Run All again: nothing left to ask
        ns, log, error = _run_all(hub_dir, here)
        assert error is None and log["loads"] == 0 and log["replies"] == 0, (repr(error), log["loads"])
        assert [j["status"] for j in ns["JOBS"]] == ["skipped: already done"] * 2

        # a fresh machine, the Hub already holding the output: skipped as well
        for run in RUNS:
            shutil.copytree(_folder(here, run), _folder(hub_dir, run))
        with tempfile.TemporaryDirectory() as fresh:
            ns, log, error = _run_all(hub_dir, fresh)
            assert error is None and log["loads"] == 0, (repr(error), log["said"][-1500:])
            assert [j["status"] for j in ns["JOBS"]] == ["skipped: already done"] * 2
            assert not any("adapter" in f for f in log["downloads"]), "adapters were downloaded for runs already done"
        print("  Run All again, here or on a fresh machine with the output on the Hub: every run skipped, no model loaded")

        # one folder, one set-up
        ns, log, error = _run_all(hub_dir, here, FLIP_AB=True)
        assert isinstance(error, ValueError) and "flip_ab" in str(error) and "another NAME" in str(error), repr(error)
        assert log["loads"] == 0
        ns, log, error = _run_all(hub_dir, here, FLIP_AB=True, NAME="t-flipped")
        assert error is None and log["loads"] == 2
        raw = [json.loads(l) for l in (_folder(here, next(iter(RUNS)), "t-flipped") / "reports_raw_decision_step-250.jsonl")
               .read_text(encoding="utf-8").splitlines()]
        plain = [json.loads(l) for l in (_folder(here, next(iter(RUNS))) / "reports_raw_decision_step-250.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
        shown = [(a["A_values"], a["B_values"]) for a in plain if "A_values" in a]
        swapped = [(b["B_values"], b["A_values"]) for b in raw if "A_values" in b]
        assert shown and shown == swapped, "FLIP_AB did not show the same pairs with A and B swapped"
        print("  the same NAME with A and B swapped is refused before a model loads; under its own NAME the same pairs, swapped")


def test_other_settings_run_too():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        run = next(iter(RUNS))
        ns, log, error = _run_all(hub_dir, here, RUN="20990101-000001", STEPS=[500], PER_KIND=None, EVAL_BATCH_SIZE=24,
                                  ONLY=["screen_index", "screen_cut_units_no_trial"], SHOW_NEITHER=False, REPLY_PREFIX="")
        assert error is None, (repr(error), log["said"][-1500:])
        assert log["applied"] == [(run, 500)] and log["batch_sizes"] == {24}
        assert log["replies"] == 50 + 1                     # the run's own 50 pairs, and one question shown no pair
        record = json.loads((_folder(here, run) / "settings.json").read_text(encoding="utf-8"))
        assert set(record["report_schema"]["questions"]) == {"screen_index", "screen_cut_units_no_trial"}
        assert record["report_schema"]["show_neither"] is False and record["_ask_only"]["per_kind"] is None
        # no GPU: the run's own batch size is kept
        ns, log, error = _run_all(hub_dir, here, vram=None, RUN=run, STEPS=[250], NAME="own-size")
        assert error is None and log["batch_sizes"] == {96}, (repr(error), log["batch_sizes"])
        print("  one run id, one step, two questions, the run's own 50 pairs, a typed batch size; no GPU keeps the run's size")


def test_a_failed_run_does_not_stop_the_next_and_is_named_last():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        first, second = RUNS
        ns, log, error = _run_all(hub_dir, here, fail=("persona_a/", "a stand-in failure"))
        assert isinstance(error, RuntimeError) and first in str(error) and "a stand-in failure" in str(error), repr(error)
        assert [j["status"].split(":")[0] for j in ns["JOBS"]] == ["FAILED", "asked"]
        assert list(pd.read_csv(_folder(here, second) / "timing.csv")["step"]) == STEPS
        assert not (_folder(here, first) / "timing.csv").exists()
        assert (_folder(here, second).relative_to(here).as_posix(), False) in log["pushed"]
        assert any(stop for _, stop in log["pushed"]), "the last cell stopped before it saved"
        # started again with the failure gone: only the run that failed is asked
        ns, log, error = _run_all(hub_dir, here)
        assert error is None and log["applied"] == [(first, s) for s in STEPS], (repr(error), log["applied"])
        print("  a run that fails: the next is still asked and pushed, the last cell saves and then names the failure; "
              "a second Run All asks only the run that failed")

    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        ns, log, error = _run_all(hub_dir, here, fail=("persona_a/", "CUDA out of memory. Tried to allocate 2.00 GiB"))
        assert isinstance(error, RuntimeError) and "EVAL_BATCH_SIZE = 96" in str(error), repr(error)
        assert [j["status"].split(":")[0] for j in ns["JOBS"]] == ["FAILED", "not asked"] and log["loads"] == 1
        print("  out of memory: the runs after it are not tried, and the batch size to set is named")


def test_a_wrong_setting_stops_at_the_plan():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        for changed, kind, wanted in (
                ({"RUN": ["20990101-000001", "20990909"]}, ValueError, "20990909"),
                ({"RUN": ["20990101-000001", "20990101-000001"]}, ValueError, "twice"),
                ({"STEPS": [250, 750]}, FileNotFoundError, "[750]"),
                ({"STEPS": [0, 250]}, ValueError, "Step 0"),
                ({"ONLY": ["main", "not_a_question"]}, ValueError, "not_a_question"),
                ({"PER_KIND": 0}, ValueError, "PER_KIND"),
                ({"NAME": "a/b"}, ValueError, "NAME"),
                ({"EVAL_BATCH_SIZE": 0}, ValueError, "eval_batch_size"),
                ({"QUESTIONS_FROM": "configs/no_such_file.json"}, Exception, "no_such_file")):
            ns, log, error = _run_all(hub_dir, here, **changed)
            assert isinstance(error, kind) and wanted in str(error), (changed, repr(error))
            assert log["loads"] == 0 and not log["applied"], changed
            assert not any("adapter" in f for f in log["downloads"]), (changed, "adapters were downloaded before the plan failed")
        print("  a mistyped run, a run named twice, a step with no adapter, step 0, an unknown question, a bad PER_KIND, "
              "NAME, batch size or questions file: each stops at the plan, nothing downloaded but settings, no model loaded")


def test_the_batch_size_follows_the_card():
    from src import ask_only as A

    for vram, size in ((15.9, 4), (22.5, 48), (39.6, 96), (47.5, 96), (79.3, 192), (95.6, 192)):   # what cards report
        assert A.card_batch_size("Qwen/Qwen3-8B", vram) == size, (vram, A.card_batch_size("Qwen/Qwen3-8B", vram))
    assert A.card_batch_size("Qwen/Qwen3-8B", 10.0) is None and A.card_batch_size("Qwen/Qwen3-8B", None) is None
    assert A.card_batch_size("Qwen/Qwen3-14B", 79.3) == 32 and A.card_batch_size("some/other-model", 79.3) is None
    print("  Qwen3-8B: 4 on 16 GB, 48 on 24 GB, 96 on 40 GB, 192 on 80 GB and above; no profile or no GPU: the run's own")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all ask_only notebook checks passed")
