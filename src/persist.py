"""Saving a run off the machine it ran on.

A rented GPU box is deleted at expiry and there is no recovery, so results and
adapters have to leave it while the run is still going rather than at the end.
This uploads a run to the Hugging Face Hub, which needs no browser and no OAuth
dance on a headless box: only HF_TOKEN in the environment.

    python -m src.persist --repo <user>/<name>                # every run present
    python -m src.persist --repo <user>/<name> --run-id <id>  # one run
    python -m src.persist --repo <user>/<name> --no-checkpoints

Results are small enough to push often. Adapters are tens of megabytes each, so
`--no-checkpoints` is the cheap call to repeat mid-run. Reads none of the six
settings.
"""
import argparse
import os
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

    for name in ("results.csv", "train_log.csv"):
        path = results_dir / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"results/{name}",
                            repo_id=repo_id, repo_type="model",
                            commit_message=f"results: {name}")
            sent.append(str(path))

    for folder, target in (
        (results_dir / run_id, f"results/{run_id}"),
        (results_dir / "plots", "results/plots"),
    ):
        if folder.is_dir():
            api.upload_folder(folder_path=str(folder), path_in_repo=target,
                              repo_id=repo_id, repo_type="model",
                              commit_message=f"results: {target}")
            sent.append(str(folder))

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
    return sent


# where finished runs are pushed; a branch source missing on this machine is fetched from here (BRANCH_SOURCE_REPO overrides)
DEFAULT_REPO = "adamchanna/bdp-plunkett-qwen3"


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
    args = parser.parse_args()
    ids = [args.run_id] if args.run_id else run_ids(args.results_dir)
    if not ids:
        print(f"no runs found under {args.results_dir}")
        return
    for run_id in ids:
        push_run(run_id, args.repo, args.results_dir, args.checkpoint_root,
                 checkpoints=not args.no_checkpoints, private=not args.public)


if __name__ == "__main__":
    main()
