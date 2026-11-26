#!/usr/bin/env python3
"""Run once, before run_experiment.ipynb, in the environment that will train.

    python run_me.py

1. Reports Python, the GPU driver's CUDA version, and the installed versions of
   torch, transformers, peft, bitsandbytes and accelerate.
2. Installs any of those that are missing. torch is installed from the PyTorch
   index that matches the driver's CUDA version, never from PyPI's default,
   because the default is built against the newest CUDA and a box with an
   older driver cannot initialise it. A torch that is present but cannot see a
   GPU the driver can see is reinstalled the same way.
3. Never upgrades or downgrades any other package that is already present.
4. Rewrites requirements.txt so the pinned packages match this environment
   exactly (local build tags such as +cu128 are dropped; the pin still resolves
   to the installed build, since pip matches a local version to a bare pin).
5. The notebook's install cell then installs from that requirements.txt, which
   is a no-op for the pinned packages and installs the rest.

Reads none of the six settings.
"""
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PINNED = ["torch", "transformers", "peft", "bitsandbytes"]
UNPINNED = ["accelerate", "numpy", "pandas", "scikit-learn", "matplotlib"]
LINUX_ONLY = {"bitsandbytes"}
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"


def version_of(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def pip(*args, check=True):
    return subprocess.run([sys.executable, "-m", "pip", *args], check=check)


def driver_cuda_tag():
    """'cu128' from the driver's 'CUDA Version: 12.8' line, or None with no driver."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return f"cu{match.group(1)}{match.group(2)}" if match else None


def torch_sees_cuda():
    """Checked in a fresh interpreter, so a reinstall can be verified in this run."""
    code = "import warnings,torch;warnings.simplefilter('ignore');print(int(torch.cuda.is_available()))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return result.stdout.strip() == "1"


def install_torch(tag):
    """Install torch for the driver's CUDA, replacing any build that shadows it."""
    if tag is None:
        print("installing torch from PyPI (no NVIDIA driver found; CPU build)")
        pip("install", "-q", "torch")
        return
    print(f"installing torch for the driver's CUDA ({tag}) from the PyTorch index")
    pip("uninstall", "-y", "-q", "torch", check=False)
    pip("install", "-q", "torch", "--index-url", TORCH_INDEX.format(tag=tag))


def main():
    root = Path(__file__).resolve().parent
    linux = platform.system() == "Linux"
    tag = driver_cuda_tag()
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"driver CUDA: {tag or 'none (no nvidia-smi)'}")

    reinstalled = False
    if version_of("torch") is None:
        install_torch(tag)
        reinstalled = True
    elif tag and not torch_sees_cuda():
        print(f"torch {version_of('torch')} is installed but cannot use the GPU the driver exposes;")
        print("its CUDA build does not match the driver. Replacing it.")
        install_torch(tag)
        reinstalled = True

    cuda = torch_sees_cuda()
    print(f"torch {version_of('torch')}, cuda available: {cuda}")
    if tag and not cuda:
        print("WARNING: torch still cannot see the GPU. Run `nvidia-smi` and check CUDA_VISIBLE_DEVICES.")

    missing = [
        p
        for p in PINNED[1:] + UNPINNED
        if version_of(p) is None and (linux or p not in LINUX_ONLY)
    ]
    if missing:
        print("installing missing packages:", " ".join(missing))
        pip("install", "-q", *missing)

    lines = []
    for p in PINNED:
        v = version_of(p)
        marker = "; sys_platform == 'linux'" if p in LINUX_ONLY else ""
        lines.append(f"{p}{marker}" if v is None else f"{p}=={v.split('+')[0]}{marker}")
    lines += UNPINNED
    (root / "requirements.txt").write_text(
        "# Pinned by run_me.py from the live environment.\n" + "\n".join(lines) + "\n"
    )
    print("wrote requirements.txt:")
    for line in lines:
        print("  " + line)
    if reinstalled:
        print()
        print("torch was (re)installed. RESTART THE KERNEL before the GPU check cell,")
        print("because a kernel that already imported the old torch keeps it until restart.")
    print("done. Now run run_experiment.ipynb; its install cell uses this file.")


if __name__ == "__main__":
    main()
