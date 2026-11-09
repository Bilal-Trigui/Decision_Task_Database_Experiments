"""Run one settings file outside the notebook (the one-persona run by default).

    python scripts/run_one_persona.py                                    # a fresh run
    python scripts/run_one_persona.py --settings configs/cut_sweep_w65_c35.json
    python scripts/run_one_persona.py --settings X --start-from <run_id> 2750 --end 5000
    python scripts/run_one_persona.py --resume <run_id> <step>           # carry on from a saved checkpoint

--start-from and --end override the file's model_hyperparameters.start_from and training_steps, so
one sweep's configs can branch from another checkpoint without being rewritten. Does what the
notebook's run cell does: loads the settings, sets the CUDA allocator the notebook sets, and calls
src.pipeline.run. Results go to results/<run_id>/, checkpoints to checkpoints/<run_id>/. Before a
fresh run it checks that the data folder is built and that the disk can hold every checkpoint, since
a full disk would fail a save part way through. The last line it prints is "RUN_ID <run_id>".
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))     # the notebook's model cache, so Qwen3-8B is not downloaded twice


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", default="configs/one_persona_minimum_threshold.json")
    parser.add_argument("--resume", nargs=2, metavar=("RUN_ID", "STEP"), default=None)
    parser.add_argument("--start-from", nargs=2, metavar=("RUN_ID", "STEP"), default=None,
                        help="branch from this run's decision checkpoint instead of the file's start_from")
    parser.add_argument("--end", type=int, default=None, help="train to this step instead of the file's training_steps")
    a = parser.parse_args()

    from src.experiments import describe, disk_needed_gb, load_experiment
    from src.pipeline import run

    cfg = load_experiment(a.settings, a.start_from, a.end)
    if a.start_from or a.end is not None:
        cfg["_source"] += " (start_from / training_steps from the command line)"
    mt = cfg["model_training"]
    if not (Path(mt["data_dir"]) / "manifest.json").exists():
        sys.exit(f"{mt['data_dir']} is not built; run: python data/general/vector_dataset_constructor.py --settings {a.settings}")
    describe(cfg)
    need_gb, free_gb = disk_needed_gb(cfg), shutil.disk_usage(ROOT).free / 1024 ** 3
    print(f"about {need_gb:.1f} GB needed for checkpoints and results; {free_gb:.1f} GB free")
    if free_gb < need_gb and not a.resume:
        sys.exit("not enough disk for every checkpoint; free some space first")
    resume = {"run_id": a.resume[0], "from_step": int(a.resume[1])} if a.resume else None
    rows = run(cfg, resume=resume)
    run_id = rows[-1]["run_id"] if rows else (resume or {}).get("run_id", "")
    print(f"RUN_ID {run_id}")


if __name__ == "__main__":
    main()
