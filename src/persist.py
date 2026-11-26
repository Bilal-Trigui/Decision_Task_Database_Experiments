"""Saving a run off the machine it ran on.

A rented GPU box is deleted at expiry and there is no recovery, so results and
adapters have to leave it while the run is still going rather than at the end.
This uploads a run to the Hugging Face Hub, which needs no browser and no OAuth
dance on a headless box: only HF_TOKEN in the environment.

    python -m src.persist --repo <user>/<name>                # every run present
    python -m src.persist --repo <user>/<name> --run-id <id>  # one run
    python -m src.persist --repo <user>/<name> --no-checkpoints
    python -m src.persist --repo <user>/<name> --restore-history   # once: rows lost to earlier pushes, put back

results.csv and train_log.csv are shared by every run, and a machine holds only the runs it ran, so a push
adds this machine's rows to the Hub's copy rather than replacing it (merged_with_hub). Until 2026-11-16 a
push replaced it: the copy on the Hub then held only the last machine's runs, and the earlier rows are in
the Hub repo's history and on the machines that ran them.

Results are small enough to push often. Adapters are tens of megabytes each, so
`--no-checkpoints` is the cheap call to repeat mid-run. Reads none of the six
settings.
"""
import argparse
import os
import tempfile
from pathlib import Path

CREDENTIALS_MESSAGE = "LOGIN CREDENTIALS NEEDED HERE"


def _api():
    """An HfApi authenticated from HF_TOKEN, or a clear stop if it is missing."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"{CREDENTIALS_MESSAGE}: HF_TOKEN")
        raise RuntimeError(
            "src.persist needs HF_TOKEN in the environment: a write token from "
            "https://huggingface.co/settings/tokens"
        )
    return HfApi(token=token), token


SUMMARY_FILES = ("results.csv", "train_log.csv")    # one row per checkpoint and per training step, every run's
# What makes a row of each file the row it is. Two copies of one row can differ in how a number is written (a file
# rewritten when a column was added turns 1500 into 1500.0), so rows are matched on these, not cell by cell.
# timestamp is part of a results row's key, so a second copy written on purpose (a resume with overwrite) is kept.
ROW_KEYS = {
    "results.csv": ["run_id", "timestamp", "report_schema", "report_batch", "stage", "introspection_fold",
                    "introspection_from_step", "introspection_target", "checkpoint_step", "block", "estimation_trials"],
    "train_log.csv": ["run_id", "stage", "fold", "introspection_from_step", "introspection_target", "step",
                      "resumed_from_step"],
}


def _canonical(cell):
    """A key cell as text, with a whole number written one way (1500.0 and 1500 are the same step)."""
    text = str(cell).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].lstrip("-").isdigit() else text


def _read(path):
    """A summary file as text cells, so nothing is reformatted on the way through."""
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def merge_rows(remote, local, keys=None):
    """The Hub's rows, then this machine's rows that the Hub does not already hold, as one frame.

    Columns are this machine's, then any only the Hub's file has; a cell a file lacks is blank. With `keys`
    (ROW_KEYS) a row is the same row when its key cells match, and this machine's copy is the one kept;
    without, when every cell matches. Either way pushing a run twice adds nothing.
    """
    import pandas as pd

    columns = list(local.columns) + [c for c in remote.columns if c not in local.columns]
    both = pd.concat([remote.reindex(columns=columns, fill_value=""), local.reindex(columns=columns, fill_value="")],
                     ignore_index=True)
    keys = [k for k in (keys or []) if k in both.columns]
    if not keys:
        return both.drop_duplicates(ignore_index=True)
    key = both[keys].apply(lambda column: column.map(_canonical))
    return both[~key.duplicated(keep="last")].reset_index(drop=True)


def merged_with_hub(path, repo_id, path_in_repo, token, out):
    """`path` merged with the Hub's copy of it (merge_rows), written to `out`; `path` itself when the Hub has none.

    A failure to read the Hub's copy for any reason other than its absence is raised, not swallowed: uploading
    without it would replace every other machine's rows.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    try:
        # force_download: a copy cached earlier is never merged in place of the Hub's current file, which is what
        # hf_hub_download falls back to when the Hub cannot be reached
        remote = hf_hub_download(repo_id, path_in_repo, repo_type="model", token=token, force_download=True)
    except LocalEntryNotFoundError:
        # the Hub could not be reached (connection error, timeout). This class is a subclass of EntryNotFoundError,
        # so it has to be caught first: it does not mean the Hub has no such file
        raise
    except EntryNotFoundError:
        return Path(path)
    merged = merge_rows(_read(remote), _read(path), ROW_KEYS.get(Path(path_in_repo).name))
    merged.to_csv(out, index=False, lineterminator="\n")
    print(f"  {path_in_repo}: {len(merged)} rows ({len(_read(remote))} on the Hub, {len(_read(path))} here)")
    return Path(out)


def summary_history(repo_id, name, token=None):
    """Every version of results/<name> the Hub's history holds, merged oldest to newest (merge_rows).

    Until 2026-11-16 each push replaced the file, so the copy on the Hub held only the last machine's runs.
    The earlier versions are still in the repo's history; this puts their rows back together. Returns the
    merged frame and how many versions went into it.
    """
    from huggingface_hub import HfApi, hf_hub_download

    commits = HfApi(token=token).list_repo_commits(repo_id, repo_type="model")      # newest first
    versions = [c.commit_id for c in commits if c.title == f"results: {name}"][::-1]
    merged = None
    for revision in versions:
        frame = _read(hf_hub_download(repo_id, f"results/{name}", repo_type="model", revision=revision, token=token))
        merged = frame if merged is None else merge_rows(merged, frame, ROW_KEYS.get(name))
    return merged, len(versions)


def restore_history(repo_id, results_dir="results", upload=True):
    """Put back on the Hub every row its history holds of results.csv and train_log.csv, with this machine's rows.

    Run once, with HF_TOKEN set, from the machine whose results/ folder holds the most runs:
        python -m src.persist --repo <user>/<name> --restore-history
    Nothing is removed: the result is the union of every version in the history and the local files. With
    upload False nothing is sent and the merged frames are returned, to look at first.
    """
    token = os.environ.get("HF_TOKEN") or None
    out = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in SUMMARY_FILES:
            merged, versions = summary_history(repo_id, name, token)
            if merged is None:
                continue
            local = Path(results_dir) / name
            if local.exists():
                merged = merge_rows(merged, _read(local), ROW_KEYS.get(name))
            out[name] = merged
            runs = merged["run_id"].nunique() if "run_id" in merged else 0
            print(f"  results/{name}: {len(merged)} rows, {runs} runs, from {versions} versions in the history"
                  + (" and this machine's file" if local.exists() else ""))
            if upload:
                api, _ = _api()
                path = Path(scratch) / name
                merged.to_csv(path, index=False, lineterminator="\n")
                api.upload_file(path_or_fileobj=str(path), path_in_repo=f"results/{name}", repo_id=repo_id,
                                repo_type="model", commit_message=f"results: {name}")
    return out


def own_rows(path, run_id, out):
    """The rows of `run_id` in a summary file, written into the run's own folder (nothing when it has none)."""
    frame = _read(path)
    if "run_id" not in frame or not Path(out).parent.is_dir():
        return None
    rows = frame[frame["run_id"] == run_id]
    if rows.empty:
        return None
    rows.to_csv(out, index=False, lineterminator="\n")
    return Path(out)


def rebuild_summary(results_dir="results", name="results.csv"):
    """Every run folder's own rows (own_rows) put back together as one frame: the summary file, rebuilt."""
    import pandas as pd

    parts = [_read(p) for p in sorted(Path(results_dir).glob(f"*/{name.replace('.csv', '_rows.csv')}"))]
    if not parts:
        return pd.DataFrame()
    columns = list(dict.fromkeys(c for part in parts for c in part.columns))
    return pd.concat([part.reindex(columns=columns, fill_value="") for part in parts], ignore_index=True)


def push_run(run_id, repo_id, results_dir="results", checkpoint_root="checkpoints",
             checkpoints=True, private=True):
    """Upload one run's results, plots and adapters to `repo_id`. Returns what was sent.

    The layout on the Hub mirrors the layout on disk, so a later download drops
    straight back into a working tree.
    """
    api, token = _api()
    results_dir, checkpoint_root = Path(results_dir), Path(checkpoint_root)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    sent = []

    # The run's own rows of the two summary files go into its folder first, so they travel with it and the
    # summaries can always be rebuilt from the run folders (rebuild_summary).
    for name in SUMMARY_FILES:
        if (results_dir / name).exists():
            own_rows(results_dir / name, run_id, results_dir / run_id / name.replace(".csv", "_rows.csv"))

    for folder, target in (
        (results_dir / run_id, f"results/{run_id}"),
        (results_dir / "plots", "results/plots"),
    ):
        if folder.is_dir():
            api.upload_folder(folder_path=str(folder), path_in_repo=target,
                              repo_id=repo_id, repo_type="model",
                              commit_message=f"results: {target}")
            sent.append(str(folder))

    # The two summary files hold every run's rows, and each machine has only the runs it ran. Uploading this
    # machine's copy as it stands would replace the Hub's with it and drop every other machine's runs, so the
    # Hub's rows are kept and this machine's added. After the run's folder, so a summary that cannot be merged
    # (the Hub's copy unreadable just now) never keeps the run itself off the Hub; the error is raised at the end.
    summary_error = None
    with tempfile.TemporaryDirectory() as scratch:
        for name in SUMMARY_FILES:
            path = results_dir / name
            if not path.exists():
                continue
            try:
                merged = merged_with_hub(path, repo_id, f"results/{name}", token, Path(scratch) / name)
                api.upload_file(path_or_fileobj=str(merged), path_in_repo=f"results/{name}",
                                repo_id=repo_id, repo_type="model",
                                commit_message=f"results: {name}")
                sent.append(str(path))
            except Exception as err:
                summary_error = summary_error or err
                print(f"  results/{name} NOT pushed ({type(err).__name__}: {err}); the run's own rows are in its folder")

    if checkpoints:
        folder = checkpoint_root / run_id
        if folder.is_dir():
            api.upload_folder(folder_path=str(folder), path_in_repo=f"checkpoints/{run_id}",
                              repo_id=repo_id, repo_type="model",
                              commit_message=f"checkpoints: {run_id}")
            sent.append(str(folder))

    print(f"pushed to https://huggingface.co/{repo_id}")
    for item in sent:
        print("  " + item)
    if not sent:
        print(f"  nothing found for run '{run_id}'")
    if summary_error is not None:
        raise summary_error
    return sent


def push_folder(folder, repo_id, root=None, private=True, stop_on_error=True):
    """Upload one folder to `repo_id`, at the path it has under the project folder (`root`). Returns that path.

    For output that is not a whole run: results/<run_id>/ask_only/<NAME>/ and results/<run_id>/evaluate_only/<NAME>/,
    which ask_only.ipynb and evaluate_checkpoints.ipynb write. Files the Hub holds that the folder does not are left
    as they are. With no repo named nothing is pushed. A push that fails raises, unless stop_on_error is off: then it
    prints NOT pushed with the reason and returns None, so a loop of jobs goes on and the folder can be pushed later.
    """
    folder = Path(folder).resolve()
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    if not repo_id:
        print(f"NOT pushed (no Hub repo named): {folder}")
        return None
    try:
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is not a folder")
        target = folder.relative_to(root).as_posix()    # a ValueError when the folder is outside the project
        api, _ = _api()
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(folder), path_in_repo=target, repo_id=repo_id, repo_type="model",
                          commit_message=f"results: {target}")
    except Exception as err:
        if stop_on_error:
            raise
        print(f"NOT pushed ({type(err).__name__}: {err}): {folder}")
        return None
    print(f"pushed {target} to https://huggingface.co/{repo_id}/tree/main/{target}")
    return target


# where finished runs are pushed; a branch source missing on this machine is fetched from here (BRANCH_SOURCE_REPO overrides)
DEFAULT_REPO = "ANONYMOUS/bdp-plunkett-qwen3"


def pull_branch_source(run_id, step, checkpoint_root="checkpoints", results_dir="results", repo_id=None):
    """Download one decision checkpoint (checkpoints/<run_id>/decision/step-<step>/) and the run's settings.json
    from the Hub into this tree, so a run whose start_from names it can branch on a fresh machine. Reading a public
    repo needs no token (HF_TOKEN is used if set). Returns the checkpoint folder, or None if the Hub has no such
    checkpoint. Files already present are left as they are."""
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    repo_id = repo_id or os.environ.get("BRANCH_SOURCE_REPO") or DEFAULT_REPO
    ckpt = f"checkpoints/{run_id}/decision/step-{step}"
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo_id, local_dir=tmp, token=os.environ.get("HF_TOKEN") or None,
                          allow_patterns=[f"{ckpt}/*", f"results/{run_id}/settings.json"])
        got = Path(tmp) / ckpt
        if not any(got.glob("*.safetensors")):
            return None
        dest = Path(checkpoint_root) / run_id / "decision" / f"step-{step}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in got.iterdir():
            if f.is_file() and not (dest / f.name).exists():
                shutil.copyfile(f, dest / f.name)
        settings = Path(tmp) / "results" / run_id / "settings.json"
        if settings.exists() and not (Path(results_dir) / run_id / "settings.json").exists():
            (Path(results_dir) / run_id).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(settings, Path(results_dir) / run_id / "settings.json")
    print(f"branch source {run_id} step {step} fetched from https://huggingface.co/{repo_id}")
    return dest


def run_ids(results_dir="results"):
    """Every run id that has a detail folder, oldest first."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(p.name for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="Hub repo id, <user>/<name>")
    parser.add_argument("--run-id", default=None, help="one run; default is every run present")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--no-checkpoints", action="store_true", help="results only, the cheap repeat call")
    parser.add_argument("--public", action="store_true", help="create the repo public rather than private")
    parser.add_argument("--restore-history", action="store_true",
                        help="rebuild results.csv and train_log.csv on the Hub from every version in its history")
    args = parser.parse_args()
    if args.restore_history:
        restore_history(args.repo, args.results_dir)
        return
    ids = [args.run_id] if args.run_id else run_ids(args.results_dir)
    if not ids:
        print(f"no runs found under {args.results_dir}")
        return
    for run_id in ids:
        push_run(run_id, args.repo, args.results_dir, args.checkpoint_root,
                 checkpoints=not args.no_checkpoints, private=not args.public)


if __name__ == "__main__":
    main()
