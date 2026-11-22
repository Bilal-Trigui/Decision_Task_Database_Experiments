"""Download checkpoints, and the run files a notebook reads, from the Hugging Face repo runs are pushed to.

The code of download_checkpoint.ipynb as functions, so the notebooks that need a saved adapter can fetch it
themselves:

    from src import download as DL
    RUN = DL.fetch_run(HUB_REPO, RUN, STEPS)                                             # evaluate_checkpoints
    RUN = DL.fetch_run(HUB_REPO, RUN, STEPS, results=("selections", "reports_raw"))      # ask_only
    DL.fetch_run(HUB_REPO, run_id, [step])                                               # cut_sweep_training
    DL.fetch_links(["https://huggingface.co/<user>/<repo>/tree/main/checkpoints/<run>/decision/step-1500"])

fetch_run takes a run id or its timestamp prefix and the decision steps wanted, and downloads whatever of these is
not here yet (when the Hub cannot be reached it goes on if the adapters and settings.json are already here)

  checkpoints/<run>/decision/step-<N>/   every file of each step's folder (the adapter, its config, and the
                                         training state when the run saved it); step 0 is the base model: nothing
  results/<run>/settings.json            which every notebook that reads a run needs
  results/<run>/<kind>_<tag>.<ext>       for each kind in `results`, each step's file: "selections" (the run's saved
                                         decisions; ask_only refits the rule from them, evaluate_checkpoints compares
                                         with them) and "reports_raw" (the run's saved replies, for ask_only's
                                         reproduction check)

The Hub repo mirrors this tree, so every file is written to the same path here as it has on the Hub. Files already
here are kept unless overwrite is set. Every file fetched or kept is then checked against the Hub: its size, and
SHA-256 for the large (LFS) ones such as adapter_model.safetensors.

A public repo needs no token. For a private one, set HF_TOKEN in the environment (or Colab's secrets) first.
"""
import hashlib
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"^/(?:(?P<type>datasets|spaces)/)?(?P<repo>[^/]+/[^/]+)(?:/(?P<kind>blob|resolve|tree)/(?P<rev>[^/]+)(?:/(?P<path>.*))?)?/?$")
STEP = re.compile(r"^(?P<folder>.*/step-\d+)/[^/]+$")
RESULT_FILES = {"selections": "selections_{tag}.csv", "reports_raw": "reports_raw_{tag}.jsonl"}


def token():
    """HF_TOKEN from the environment, or from Colab's secrets panel when the notebook runs there; else None."""
    found = os.environ.get("HF_TOKEN") or None
    if found is None:
        try:
            from google.colab import userdata
            found = userdata.get("HF_TOKEN")
        except Exception:
            pass
    return found


def parse_link(link):
    """(repo_id, repo_type, revision, path, is_folder) for a huggingface.co link."""
    u = urlparse(link.strip())
    if u.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(f"not a huggingface.co link: {link}")
    m = LINK.match(unquote(u.path))
    if not m:
        raise ValueError(f"cannot read this link: {link}")
    repo_type = {"datasets": "dataset", "spaces": "space"}.get(m["type"], "model")
    path = (m["path"] or "").strip("/")
    return m["repo"], repo_type, m["rev"] or "main", path, m["kind"] in (None, "tree")


def files_for(every, repo_id, rev, path, is_folder, whole_checkpoint):
    """Every Hub file the link names (from `every`, the repo's file list): the file, or everything under the folder."""
    if not is_folder:
        if path not in every:
            raise FileNotFoundError(f"{path} is not in {repo_id} at {rev}")
        step = STEP.match(path)
        if whole_checkpoint and step:
            path, is_folder = step["folder"], True
        else:
            return [path]
    prefix = f"{path}/" if path else ""
    got = sorted(f for f in every if f.startswith(prefix))
    if not got:
        raise FileNotFoundError(f"nothing under {path or 'the repo root'} in {repo_id} at {rev}")
    return got


def run_of(path):
    """The run id of a checkpoints/<run>/... or results/<run>/... path, else None."""
    parts = path.split("/")
    return parts[1] if len(parts) > 2 and parts[0] in ("checkpoints", "results") else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_run(every, run, repo_id="the Hub repo"):
    """The full run id for a run id or its timestamp prefix, among the runs in `every` (the repo's file list)."""
    runs = sorted({run_of(f) for f in every} - {None})
    hits = [r for r in runs if r == run] or [r for r in runs if r.startswith(run)]
    if len(hits) != 1:
        raise ValueError(f"run {run!r} matches {len(hits)} runs in {repo_id}: {hits[:10]}")
    return hits[0]


def local_run(root, run):
    """The full run id when results/ and checkpoints/ under `root` hold exactly one run matching `run`, else None."""
    names = set()
    for top in ("results", "checkpoints"):
        folder = Path(root) / top
        if folder.is_dir():
            names |= {p.name for p in folder.iterdir() if p.is_dir()}
    hits = [n for n in sorted(names) if n == run] or [n for n in sorted(names) if n.startswith(run)]
    return hits[0] if len(hits) == 1 else None


def adapters_here(root, run_id, steps):
    """True when settings.json and every step's adapter (weights and config) are already under `root`."""
    root = Path(root)
    folders = [root / "checkpoints" / run_id / "decision" / f"step-{step}" for step in steps if step]
    return (root / "results" / run_id / "settings.json").exists() and all(
        (f / "adapter_config.json").exists() and any(f.glob("*.safetensors")) for f in folders)


def steps_of(every, run_id):
    """The decision steps of `run_id` that `every` (the repo's file list) has an adapter for, in order."""
    found = (re.match(rf"checkpoints/{re.escape(run_id)}/decision/step-(\d+)/.*\.safetensors$", f) for f in every)
    return sorted({int(m.group(1)) for m in found if m})


def adapter_steps(repo_id, run):
    """(run_id, steps): the full id of `run` on the Hub (a run id or its timestamp prefix) and the decision steps
    it has an adapter for there. For choosing the step a run is trained on from."""
    from huggingface_hub import HfApi

    every = HfApi().list_repo_files(repo_id, repo_type="model", token=token())
    run_id = resolve_run(every, run, repo_id)
    return run_id, steps_of(every, run_id)


def step_tag(step):
    """The tag a run's files carry for a decision step; step 0 is the base model."""
    return "base_model_step-0" if step == 0 else f"decision_step-{step}"


def files_for_run(every, run_id, steps, results=()):
    """(wanted, missing): the Hub files fetch_run downloads for these steps, and the steps the Hub has no adapter for."""
    listed = set(every)
    wanted, missing = [], []
    for step in steps:
        if step:
            prefix = f"checkpoints/{run_id}/decision/step-{step}/"
            got = sorted(f for f in every if f.startswith(prefix))
            if not any(f.endswith(".safetensors") for f in got):
                missing.append(step)
            wanted += got
    extra = [f"results/{run_id}/settings.json"]
    for kind in results:
        if kind not in RESULT_FILES:
            raise ValueError(f"results kind {kind!r} is not one of {sorted(RESULT_FILES)}")
        extra += [f"results/{run_id}/" + RESULT_FILES[kind].format(tag=step_tag(step)) for step in steps]
    return wanted + [f for f in extra if f in listed], missing


def fetch_files(repo_id, wanted, root=None, repo_type="model", rev="main", overwrite=False, dry_run=False, label=None):
    """Download the Hub files `wanted` to the same paths under `root`. Returns [(repo_id, repo_type, rev, file)]
    for every file now here (fetched or kept), which check() takes; [] on a dry run."""
    from huggingface_hub import HfApi, hf_hub_download

    root, tok = Path(root or ROOT), token()
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        print(f"{label or repo_id}\n  nothing to download")
        return []
    sizes = {i.path: i.size for i in HfApi().get_paths_info(repo_id, wanted, repo_type=repo_type, revision=rev, token=tok)}
    new = [f for f in wanted if overwrite or not (root / f).exists()]
    print(f"{label or repo_id}\n  {repo_id} ({repo_type}) at {rev}: {len(wanted)} file(s), {sum(sizes.values()) / 1e9:.2f} GB; "
          f"{len(new)} to download ({sum(sizes[f] for f in new) / 1e9:.2f} GB), {len(wanted) - len(new)} already here")
    if dry_run:
        for f in wanted:
            print(f"  {'would fetch' if f in new else 'have      '}  {f} ({sizes[f] / 1e6:.1f} MB)")
        return []
    done = []
    # Downloaded to a temporary folder and copied, so no .cache/huggingface folder is left inside checkpoints/.
    with tempfile.TemporaryDirectory() as tmp:
        for f in wanted:
            dest = root / f
            if dest.exists() and not overwrite:
                print(f"  kept     {f} (already here; overwrite replaces it)")
                done.append((repo_id, repo_type, rev, f))
                continue
            local = hf_hub_download(repo_id, f, repo_type=repo_type, revision=rev, local_dir=tmp, token=tok)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dest)
            print(f"  fetched  {f} ({dest.stat().st_size / 1e6:.1f} MB)")
            done.append((repo_id, repo_type, rev, f))
    return done


def fetch(link, root=None, whole_checkpoint=True, also_settings=True, overwrite=False, dry_run=False):
    """Download what one huggingface.co link names (a file's page, its download link, or a folder page)."""
    from huggingface_hub import HfApi

    repo_id, repo_type, rev, path, is_folder = parse_link(link)
    every = HfApi().list_repo_files(repo_id, repo_type=repo_type, revision=rev, token=token())    # listed once per link
    wanted = files_for(every, repo_id, rev, path, is_folder, whole_checkpoint)
    if not all(f.startswith("checkpoints/") for f in wanted):
        print("  note: not every file is under checkpoints/; each goes to the same path here as on the Hub")
    if also_settings:
        listed = set(every)
        runs = {run_of(f) for f in wanted} - {None}
        wanted += sorted(s for s in (f"results/{r}/settings.json" for r in runs) if s in listed and s not in wanted)
    return fetch_files(repo_id, wanted, root, repo_type, rev, overwrite, dry_run, label=link)


def check(downloaded, root=None):
    """Check every downloaded file against the Hub: size, and SHA-256 for the large (LFS) ones. Returns how many differ."""
    from huggingface_hub import HfApi

    root, tok, api = Path(root or ROOT), token(), HfApi()
    by_repo = defaultdict(list)
    for repo_id, repo_type, rev, f in downloaded:
        by_repo[(repo_id, repo_type, rev)].append(f)
    bad = 0
    for (repo_id, repo_type, rev), files in by_repo.items():
        for info in api.get_paths_info(repo_id, files, repo_type=repo_type, revision=rev, token=tok):
            here = root / info.path
            ok, how = here.stat().st_size == info.size, "size"
            if ok and getattr(info, "lfs", None):
                ok, how = sha256(here) == info.lfs.sha256, "sha256"
            bad += not ok
            if not ok:
                print(f"  BAD ({how}) {info.path}")
    steps = sorted({str((root / f).parent.relative_to(root)) for *_, f in downloaded if "/step-" in f})
    for s in steps:
        names = {p.name for p in (root / s).iterdir()}
        if not ("adapter_config.json" in names and any(n.endswith(".safetensors") for n in names)):
            print(f"  note: {s} is not a complete adapter (it holds {', '.join(sorted(names))})")
    if downloaded:
        print(f"  all {len(downloaded)} file(s) match the Hub" if not bad
              else f"  {bad} problem(s): delete those files and download again with overwrite=True")
    return bad


def fetch_links(links, root=None, whole_checkpoint=True, also_settings=True, overwrite=False, dry_run=False, verify=True):
    """Download every link (see fetch) and check the files against the Hub. Returns what check() takes."""
    done = []
    for link in links:
        done += fetch(link, root, whole_checkpoint, also_settings, overwrite, dry_run)
    if verify and done:
        if check(done, root):
            raise RuntimeError("some downloaded files do not match the Hub (listed above)")
    return done


def fetch_run(repo_id, run, steps, results=(), root=None, overwrite=False, dry_run=False, verify=True):
    """Download a run's adapters at `steps`, its settings.json and the `results` files of those steps (see the
    module docstring), unless they are already here. `run` is a run id or its timestamp prefix; returns the full
    run id. A step the Hub has no adapter for is an error that lists the steps it has."""
    from huggingface_hub import HfApi

    try:
        every = HfApi().list_repo_files(repo_id, repo_type="model", token=token())
    except Exception as err:
        # the Hub cannot be reached: go on only when what the notebook loads is already on this machine
        run_id = local_run(root or ROOT, run)
        if run_id and adapters_here(root or ROOT, run_id, steps):
            print(f"{repo_id} could not be reached ({type(err).__name__}); the adapters for steps {list(steps)} of "
                  f"{run_id} and its settings.json are already here, so nothing was downloaded or checked")
            return run_id
        raise
    run_id = resolve_run(every, run, repo_id)
    wanted, missing = files_for_run(every, run_id, list(steps), results)
    if missing:
        raise FileNotFoundError(f"{repo_id} has no adapter for steps {missing} of {run_id}; it has {steps_of(every, run_id)}")
    done = fetch_files(repo_id, wanted, root, overwrite=overwrite, dry_run=dry_run,
                       label=f"{run_id}: steps {list(steps)}" + (f", with {', '.join(results)}" if results else ""))
    if verify and done:
        if check(done, root):
            raise RuntimeError("some downloaded files do not match the Hub (listed above)")
    return run_id
