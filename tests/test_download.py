"""src/download.py: the download notebook's code, as the other notebooks call it. No network and no model.

    python -m tests.test_download

A folder on disk stands in for the Hub repo (huggingface_hub's HfApi and hf_hub_download are replaced while the
checks run). Checks:

  - a link is read as download_checkpoint.ipynb reads it: a file's page, a download link, a folder page
  - a run id or its timestamp prefix names one run, and two matches are refused
  - fetch_run downloads each step's whole folder, settings.json and the results files asked for, to the same
    paths as on the Hub; step 0 needs no adapter; a second call downloads nothing; a step the Hub lacks is refused
  - a file that differs from the Hub's is caught by the check
  - with the Hub out of reach, fetch_run goes on only when the adapters and settings.json are already here
  - evaluate_checkpoints and cut_sweep_training call it: each has the Download cell, placed before the cell that
    needs the files; ask_only's ask_runs fetches each run's files before it asks that run
  - a caller that already has the repo's file list hands it to fetch_run, which then lists nothing; a "too many
    requests" answer from the Hub is waited out
"""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

from src import download as DL

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261116-221120_paper_minimum_threshold_Qwen3-8B_e-negative"
OTHER = "20261116-221042_paper_minimum_threshold_Qwen3-8B_e-positive"
REPO = "someone/some-repo"


def _hub(folder):
    """A fake Hub repo on disk: two runs, with adapters at steps 1500 and 3000 of the first."""
    files = {f"results/{RUN_ID}/settings.json": "{}", f"results/{OTHER}/settings.json": "{}",
             f"results/{RUN_ID}/selections_base_model_step-0.csv": "persona,trial\n",
             f"checkpoints/{OTHER}/decision/step-1500/adapter_model.safetensors": "other weights"}
    for step in (1500, 3000):
        files[f"checkpoints/{RUN_ID}/decision/step-{step}/adapter_model.safetensors"] = f"weights {step}"
        files[f"checkpoints/{RUN_ID}/decision/step-{step}/adapter_config.json"] = "{}"
        files[f"checkpoints/{RUN_ID}/decision/step-{step}/training_state.pt"] = f"adamw {step}"
        files[f"results/{RUN_ID}/selections_decision_step-{step}.csv"] = "persona,trial\n"
        files[f"results/{RUN_ID}/reports_raw_decision_step-{step}.jsonl"] = "{}\n"
    for name, text in files.items():
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(files)


class _FakeHub:
    """Replaces huggingface_hub.HfApi and hf_hub_download with reads of a folder; `calls` counts the downloads."""

    def __init__(self, folder, reachable=True):
        self.folder, self.reachable, self.calls = Path(folder), reachable, []

    def __enter__(self):
        self.real = huggingface_hub.HfApi, huggingface_hub.hf_hub_download
        hub = self

        class Api:
            def list_repo_files(self, repo_id, **kwargs):
                if not hub.reachable:
                    raise ConnectionError("offline")
                return sorted(str(p.relative_to(hub.folder).as_posix()) for p in hub.folder.rglob("*") if p.is_file())

            def get_paths_info(self, repo_id, paths, **kwargs):
                out = []
                for path in paths:
                    data = (hub.folder / path).read_bytes()
                    lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if path.endswith(".safetensors") else None
                    out.append(SimpleNamespace(path=path, size=len(data), lfs=lfs))
                return out

        def download(repo_id, filename, local_dir=None, **kwargs):
            hub.calls.append(filename)
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hub.folder / filename, dest)
            return str(dest)

        huggingface_hub.HfApi, huggingface_hub.hf_hub_download = Api, download
        return self

    def __exit__(self, *exc):
        huggingface_hub.HfApi, huggingface_hub.hf_hub_download = self.real


def test_links_are_read_as_the_download_notebook_reads_them():
    base = f"https://huggingface.co/{REPO}"
    step = f"checkpoints/{RUN_ID}/decision/step-1500"
    assert DL.parse_link(f"{base}/blob/main/{step}/adapter_model.safetensors") == (REPO, "model", "main", f"{step}/adapter_model.safetensors", False)
    assert DL.parse_link(f"{base}/resolve/main/{step}/adapter_model.safetensors?download=true")[3:] == (f"{step}/adapter_model.safetensors", False)
    assert DL.parse_link(f"{base}/tree/main/{step}") == (REPO, "model", "main", step, True)
    assert DL.parse_link(f"https://huggingface.co/datasets/{REPO}/tree/v2/data")[1:3] == ("dataset", "v2")
    try:
        DL.parse_link("https://example.com/a/b")
    except ValueError:
        pass
    else:
        raise AssertionError("a link to another site was accepted")
    with tempfile.TemporaryDirectory() as tmp:
        every = _hub(tmp)
    # one file inside a step folder is the whole step folder, unless whole_checkpoint is off
    assert len(DL.files_for(every, REPO, "main", f"{step}/adapter_model.safetensors", False, True)) == 3
    assert DL.files_for(every, REPO, "main", f"{step}/adapter_model.safetensors", False, False) == [f"{step}/adapter_model.safetensors"]
    assert len(DL.files_for(every, REPO, "main", f"checkpoints/{RUN_ID}/decision", True, True)) == 6
    print("  a file's page, a download link and a folder page; one file of a step is its whole folder")


def test_a_prefix_names_one_run():
    with tempfile.TemporaryDirectory() as tmp:
        every = _hub(tmp)
    assert DL.resolve_run(every, "20261116-221120") == RUN_ID and DL.resolve_run(every, RUN_ID) == RUN_ID
    for bad in ("20261116", "20270101"):       # two runs, and none
        try:
            DL.resolve_run(every, bad)
        except ValueError as err:
            assert "matches" in str(err)
        else:
            raise AssertionError(f"{bad} was taken for one run")
    print("  a timestamp prefix or the whole id; two matches or none refused")


def test_the_steps_a_run_can_be_trained_on_from_are_listed():
    with tempfile.TemporaryDirectory() as hub_dir:
        every = _hub(hub_dir)
        assert DL.steps_of(every, RUN_ID) == [1500, 3000] and DL.steps_of(every, "no-such-run") == []
        with _FakeHub(hub_dir):
            assert DL.adapter_steps(REPO, "20261116-221120") == (RUN_ID, [1500, 3000])
            assert DL.adapter_steps(REPO, OTHER) == (OTHER, [1500])
    print("  the full run id and the steps it has an adapter for")


def test_fetch_run_brings_what_a_notebook_reads():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        here = Path(here)
        with _FakeHub(hub_dir) as hub:
            got = DL.fetch_run(REPO, "20261116-221120", [0, 1500], results=("selections", "reports_raw"), root=here)
            assert got == RUN_ID
            step = here / "checkpoints" / RUN_ID / "decision" / "step-1500"
            assert sorted(p.name for p in step.iterdir()) == ["adapter_config.json", "adapter_model.safetensors", "training_state.pt"]
            results = sorted(p.name for p in (here / "results" / RUN_ID).iterdir())
            assert results == ["reports_raw_decision_step-1500.jsonl", "selections_base_model_step-0.csv",
                               "selections_decision_step-1500.csv", "settings.json"], results
            assert not (here / "checkpoints" / RUN_ID / "decision" / "step-3000").exists()     # not asked for
            assert not (here / "checkpoints" / OTHER).exists() and not (here / "results" / OTHER).exists()
            first = len(hub.calls)
            assert first == 7
            DL.fetch_run(REPO, RUN_ID, [0, 1500], results=("selections", "reports_raw"), root=here)     # all here already
            assert len(hub.calls) == first
            # settings.json alone, for a plan that only reads the settings
            DL.fetch_run(REPO, OTHER, [], root=here)
            assert (here / "results" / OTHER / "settings.json").exists() and not (here / "checkpoints" / OTHER).exists()
            try:
                DL.fetch_run(REPO, RUN_ID, [2000], root=here)
            except FileNotFoundError as err:
                assert "[1500, 3000]" in str(err), str(err)
            else:
                raise AssertionError("a step the Hub has no adapter for was accepted")
            # a file that is not the Hub's is caught
            (step / "adapter_model.safetensors").write_text("weights 15oo")
            try:
                DL.fetch_run(REPO, RUN_ID, [1500], root=here)
            except RuntimeError as err:
                assert "do not match the Hub" in str(err)
            else:
                raise AssertionError("an adapter that differs from the Hub's passed the check")
    print("  the step's whole folder, settings.json and the step's results files; nothing twice; a changed file caught")


def test_an_unreachable_hub_is_fine_only_when_the_files_are_here():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        _hub(hub_dir)
        with _FakeHub(hub_dir):
            DL.fetch_run(REPO, RUN_ID, [1500], root=here)
        with _FakeHub(hub_dir, reachable=False):
            assert DL.fetch_run(REPO, "20261116-221120", [0, 1500], root=here) == RUN_ID
            for run, steps in ((RUN_ID, [3000]), (OTHER, [1500])):       # a step, and a run, that are not here
                try:
                    DL.fetch_run(REPO, run, steps, root=here)
                except ConnectionError:
                    pass
                else:
                    raise AssertionError(f"went on without the Hub although {run} {steps} is not here")
    print("  offline with the adapter and settings.json here: goes on; offline without them: stops")


def test_the_notebooks_download_before_they_need_the_files():
    for name, needs in (("evaluate_checkpoints.ipynb", "A.find_run(RUN)"),
                        ("cut_sweep_training.ipynb", "branch_checkpoint(cfg)")):
        cells = json.loads((ROOT / name).read_text(encoding="utf-8"))["cells"]
        code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        download = [i for i, s in enumerate(code) if "DL.fetch_run(HUB_REPO" in s and "from src import download as DL" in s]
        first_use = [i for i, s in enumerate(code) if needs in s]
        assert download and first_use and download[0] < first_use[0], (name, download, first_use)
        assert len(download) == 1, (name, download)        # one Download cell a notebook
        assert any("HUB_REPO =" in s for s in code[:download[0]]), f"{name}: HUB_REPO is not set before the Download cell"
        ids = [c.get("id") for c in cells]
        assert len(ids) == len(set(ids)), f"{name}: two cells share an id"
    # ask_only has no Download cell: its Ask cell hands HUB_REPO to ask_runs, which fetches each run's adapters and
    # saved decisions just before it asks that run (tests.test_ask_only_notebook runs it against a fake Hub)
    cells = json.loads((ROOT / "ask_only.ipynb").read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    ask = [i for i, s in enumerate(code) if "A.ask_runs(RUN, STEPS" in s and "hub_repo=HUB_REPO" in s]
    assert len(ask) == 1 and any("HUB_REPO =" in s for s in code[:ask[0]]), ask
    runs = (ROOT / "src" / "ask_only.py").read_text(encoding="utf-8").split("\ndef ask_runs(")[1].split("\ndef ")[0]
    assert runs.index('DL.fetch_run(hub_repo, j["run_id"], steps') < runs.index('ask(j["run_id"], steps'), \
        "ask_runs asks a run before it has fetched the run's adapters"
    print("  evaluate_checkpoints and cut_sweep_training: HUB_REPO set, then Download, then the cells that read the "
          "files; ask_only: each run fetched before it is asked")


def test_one_listing_serves_several_runs_and_too_many_requests_is_waited_out():
    with tempfile.TemporaryDirectory() as hub_dir, tempfile.TemporaryDirectory() as here:
        every = _hub(hub_dir)
        # with the repo's file list in hand, fetch_run does not list again (an unreachable listing would raise)
        with _FakeHub(hub_dir, reachable=False):
            assert DL.fetch_run(REPO, "20261116-221120", [1500], results=("selections",), root=here, every=every) == RUN_ID
            assert (Path(here) / "checkpoints" / RUN_ID / "decision" / "step-1500" / "adapter_model.safetensors").exists()

    class TooMany(Exception):
        def __init__(self, headers, status=429):
            super().__init__("too many requests")
            self.response = SimpleNamespace(status_code=status, headers=headers)

    def hub_answering(answers):
        class Api:
            def list_repo_files(self, repo_id, **kwargs):
                answer = answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return Api

    real, waits = huggingface_hub.HfApi, []
    try:
        # the wait the Hub names, in either header it uses, plus a second; then the listing
        huggingface_hub.HfApi = hub_answering([TooMany({"Retry-After": "30"}), TooMany({"RateLimit": '"api";r=0;t=12'}), ["a/b"]])
        with contextlib.redirect_stdout(io.StringIO()):
            assert DL.listing(REPO, sleep=waits.append) == ["a/b"] and waits == [31, 13], waits
        # it gives up after `tries`, and another error is never waited on
        for answers, kind in (([TooMany({})] * 4, TooMany), ([TooMany({}, status=500)], TooMany), ([ConnectionError("offline")], ConnectionError)):
            huggingface_hub.HfApi, waits = hub_answering(list(answers)), []
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    DL.listing(REPO, sleep=waits.append)
            except kind:
                assert len(waits) == (3 if len(answers) == 4 else 0), (answers, waits)
            else:
                raise AssertionError("a listing that kept failing returned")
    finally:
        huggingface_hub.HfApi = real
    print("  fetch_run with the file list in hand lists nothing; a 429 is waited out as the Hub asks, three times at most")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all download checks passed")
