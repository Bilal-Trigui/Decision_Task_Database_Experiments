#!/usr/bin/env python3
"""Run once, before run_experiment.ipynb, in the environment that will train.

    python run_me.py

1. Reports Python, CUDA, and the installed versions of torch, transformers,
   peft, bitsandbytes and accelerate.
2. Installs any of those that are missing. It never upgrades or downgrades a
   package that is already present, so the torch that Colab ships is kept.
3. Rewrites requirements.txt so the four pinned packages match this
   environment exactly (local build tags such as +cu126 are dropped).
4. The notebook's install cell then installs from that requirements.txt,
   which is a no-op for the pinned packages and installs the rest.

Reads none of the six settings.
"""
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PINNED = ["torch", "transformers", "peft", "bitsandbytes"]
UNPINNED = ["accelerate", "numpy", "pandas", "scikit-learn", "matplotlib"]
LINUX_ONLY = {"bitsandbytes"}


def version_of(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def main():
    root = Path(__file__).resolve().parent
    linux = platform.system() == "Linux"
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = f" ({torch.cuda.get_device_name(0)})" if cuda else ""
        print(f"torch {torch.__version__}, cuda available: {cuda}{name}")
    except ImportError:
        print("torch is not installed yet")

    missing = [
        p
        for p in PINNED + UNPINNED
        if version_of(p) is None and (linux or p not in LINUX_ONLY)
    ]
    if missing:
        print("installing missing packages:", " ".join(missing))
        pip_install(missing)

    lines = []
    for p in PINNED:
        v = version_of(p)
        marker = "; sys_platform == 'linux'" if p in LINUX_ONLY else ""
        if v is None:
            lines.append(f"{p}{marker}")
        else:
            lines.append(f"{p}=={v.split('+')[0]}{marker}")
    lines += UNPINNED
    (root / "requirements.txt").write_text(
        "# Pinned by run_me.py from the live environment.\n" + "\n".join(lines) + "\n"
    )
    print("wrote requirements.txt:")
    for line in lines:
        print("  " + line)
    print("done. Now run run_experiment.ipynb; its install cell uses this file.")


if __name__ == "__main__":
    main()
