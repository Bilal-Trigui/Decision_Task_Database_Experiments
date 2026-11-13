"""Write a sweep of branch configs over the weights / cut share, and build each one's data folder.

    python scripts/make_cut_sweep.py                          # the default sweep below
    python scripts/make_cut_sweep.py --weights 75 50 25 0 --step 2750 --end 5000 --every 250 --name cut_sweep_2750

Every config extends --base (configs/minimum_threshold_one_persona/base/one_persona_minimum_threshold.json) and branches from
--source-run's decision step --step, training on to step --end, evaluated and saved every --every
steps. Each sets
    weight_decided_share  W%   trials both options pass the cut: the weights alone decide
    neither_share         C/2  trials both options fail the cut: answer C
    the rest              C/2  trials one option fails, and the cut changes the answer
                               (cut_overturns_only: the weights alone would have picked the other)
where C = 100 - W. Configs go to configs/<name>_w<W>_c<C>.json, data to data/<name>/w<W>_c<C>/.
A data folder that is already built under the same settings is left as it is. Prints the config
list to paste into the notebook's queue cell, or run with scripts/run_queue.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA = "data/minimum_threshold_one_persona/base/one_persona_minimum_threshold"      # the persona's weights, scenarios and roles
INPUTS = ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv")
DEFAULT_SOURCE = "20261106-152354_paper_minimum_threshold_Qwen3-8B"

# Windows text mode writes CRLF, which changes the manifest's hashes; build with LF line endings
LF_BUILD = """
import pathlib, runpy, sys
import pandas as pd
_write = pathlib.Path.write_text
pathlib.Path.write_text = lambda self, data, encoding=None, errors=None, newline=None: _write(self, data, encoding=encoding, errors=errors, newline="\\n")
_csv = pd.DataFrame.to_csv
def _to_csv(self, *a, **k):
    k.setdefault("lineterminator", "\\n")
    return _csv(self, *a, **k)
pd.DataFrame.to_csv = _to_csv
sys.path[:0] = [".", "data/general"]
sys.argv = ["vector_dataset_constructor.py", "--settings", sys.argv[1]]
runpy.run_path("data/general/vector_dataset_constructor.py", run_name="__main__")
"""


def write_lf(path, text):
    Path(path).write_bytes(text.encode())


def config_for(w, a):
    c = 100 - w
    stem = f"w{w:02d}_c{c:02d}"
    half = round(c / 200, 4)
    return stem, {
        "_comment": [
            f"Cut sweep: {w}% of training trials decided by the weights, {c}% by the cut ({c / 2:g}% C, both options",
            f"  below the cut; {c / 2:g}% one option below it, where the cut changes the answer from what the weights alone",
            f"  would pick). Branches from {a.source_run} decision step {a.step} (its adapter and AdamW",
            f"  state) and trains steps {a.step + 1} to {a.end}, evaluated and saved every {a.every}, over {a.data_root}/{stem}/.",
            f"  Everything else is {Path(a.base).name}'s. Written by scripts/make_cut_sweep.py.",
            "save_optimizer last: each checkpoint keeps its adapter; AdamW's state only at the last.",
        ],
        "extends": Path(a.base).name,
        "decision_rule": {
            "weight_decided_share": w / 100,
            "neither_share": half,
            "cut_overturns_only": True,
        },
        "model_training": {"data_dir": f"{a.data_root}/{stem}/"},
        "model_hyperparameters": {
            "training_steps": a.end,
            "checkpoint_every": a.every,
            "start_from": {"run_id": a.source_run, "step": a.step},
            "baseline_before_training": False,
            "save_optimizer": "last",
            "run_tag": f"{a.name.replace('_', '-')}-{stem.replace('_', '-')}",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=int, nargs="+", default=[75, 65, 55, 45, 35, 25, 15, 5, 0],
                        help="weight-decided percentages, one config each")
    parser.add_argument("--source-run", default=DEFAULT_SOURCE)
    parser.add_argument("--step", type=int, default=1500, help="the source run's checkpoint to branch from")
    parser.add_argument("--end", type=int, default=3000, help="the step each branch trains to")
    parser.add_argument("--every", type=int, default=100, help="evaluate and save a checkpoint every this many steps")
    parser.add_argument("--name", default="cut_sweep", help="config prefix and data folder name")
    parser.add_argument("--base", default="configs/minimum_threshold_one_persona/base/one_persona_minimum_threshold.json")
    parser.add_argument("--no-build", action="store_true", help="write the configs only")
    parser.add_argument("--quiet", action="store_true", help="one line in all, not the builder's log")
    a = parser.parse_args()
    a.data_root = f"data/{a.name}"
    if any(not 0 <= w < 100 for w in a.weights):
        sys.exit("--weights must each be from 0 to 99")
    ckpt = ROOT / "checkpoints" / a.source_run / "decision" / f"step-{a.step}"
    if not ckpt.exists():
        sys.exit(f"no checkpoint {ckpt}")

    written, built = [], 0
    say = (lambda *x: None) if a.quiet else print
    for w in a.weights:
        stem, cfg = config_for(w, a)
        path = ROOT / "configs" / f"{a.name}_{stem}.json"
        write_lf(path, json.dumps(cfg, indent=2) + "\n")
        written.append(path.relative_to(ROOT).as_posix())
        out = ROOT / a.data_root / stem
        out.mkdir(parents=True, exist_ok=True)
        for f in INPUTS:
            if not (out / f).exists():
                shutil.copyfile(ROOT / SOURCE_DATA / f, out / f)
        if a.no_build:
            continue
        if (out / "manifest.json").exists() and _built_under(path, out):
            say(f"{stem}: data already built")
            continue
        for f in out.iterdir():
            if f.name not in INPUTS:
                f.unlink()
        say(f"{stem}: building data")
        subprocess.run([sys.executable, "-c", LF_BUILD, path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL if a.quiet else None)
        built += 1
    if a.quiet:
        print(f"{len(written)} settings files ready, configs/{a.name}_*.json ({built} data folders built, "
              f"{len(written) - built} already there): from {a.source_run} step {a.step} to {a.end}, every {a.every}")
        return
    print("\nconfigs, in order:")
    for p in written:
        print(f'    "{p}",')


def _built_under(config, out):
    """True when the folder's manifest matches the config's rule and counts."""
    sys.path.insert(0, str(ROOT))
    from src.config import load, resolve
    from src.rules.base import check_manifest
    try:
        cfg = load(str(config))
        check_manifest(cfg["model_training"]["data_dir"], resolve(cfg).rule, cfg)
        m = json.loads((out / "manifest.json").read_text())
        return m["train_trials_per_persona"] == cfg["model_training"]["train_trials_per_persona"]
    except Exception:
        return False


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
