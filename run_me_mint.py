#!/usr/bin/env python3
"""Run once on a Linux Mint (or Ubuntu) machine, with the system Python, before anything else.

    python3 run_me_mint.py
    source .venv/bin/activate

Mint marks its system Python as externally managed, so pip refuses to install into it and
run_me.py stops at its first install. This does the same job inside a virtual environment:

1. Checks that python3 can make a virtual environment, and prints the apt line that fixes it
   when it cannot.
2. Creates .venv/ in the repo, or reuses the one already there.
3. Installs torch at the version requirements.txt pins, from the PyTorch index that fits this
   machine: the CPU build when there is no NVIDIA driver, otherwise the build for the driver's
   CUDA. Cards older than Volta (compute capability below 7.0, such as a GTX 1080) get the
   cu126 build at most, because the CUDA 12.8 builds no longer carry kernels for them and
   torch would see the GPU but fail on the first operation.
4. Installs the rest of requirements.txt, plus ipykernel so VS Code can run the notebooks on
   .venv's Python.
5. Checks the imports and, with a GPU, runs one operation on it.

Unlike run_me.py it leaves requirements.txt alone: those pins are meant to match the machine
that trains, not this one. Running it again is safe; pip skips what is already installed.
"""
import re
import subprocess
import sys
from pathlib import Path

from run_me import TORCH_INDEX, driver_cuda_tag

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"
PRE_VOLTA_MAX_TAG = "cu126"


def run(*cmd, check=True):
    sys.stdout.flush()  # keep this script's lines ahead of the child's output when piped
    return subprocess.run([str(c) for c in cmd], check=check)


def can_make_venv():
    """Mint ships python3 without ensurepip until python3-venv is installed."""
    result = subprocess.run([sys.executable, "-c", "import ensurepip"], capture_output=True)
    return result.returncode == 0


def compute_capability():
    """6.1 for a GTX 1080, 9.0 for an H100, None with no driver or no answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def pinned_torch():
    """The torch version requirements.txt pins, or None when it is unpinned."""
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        match = re.match(r"\s*torch==([^\s;]+)", line)
        if match:
            return match.group(1)
    return None


def torch_index(tag, capability):
    if tag is None:
        return "cpu"
    if capability is not None and capability < 7.0:
        # Compare the numbers in 'cu128' and 'cu126' rather than the strings.
        return min(tag, PRE_VOLTA_MAX_TAG, key=lambda t: int(t[2:]))
    return tag


def main():
    major, minor = sys.version_info[:2]
    if not can_make_venv():
        print(f"python{major}.{minor} cannot make a virtual environment yet. Install the venv module:")
        print(f"    sudo apt install python{major}.{minor}-venv")
        print("then run this again.")
        sys.exit(1)

    tag, capability = driver_cuda_tag(), compute_capability()
    print(f"python {major}.{minor}, driver CUDA: {tag or 'none'}, compute capability: {capability or 'n/a'}")

    if PY.exists():
        print(f"reusing {VENV}")
    else:
        print(f"creating {VENV}")
        run(sys.executable, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "-q", "--upgrade", "pip")

    index = torch_index(tag, capability)
    version = pinned_torch()
    spec = f"torch=={version}" if version else "torch"
    print(f"installing {spec} from the PyTorch {index} index")
    if index != tag and tag is not None:
        print(f"  ({index}, not the driver's {tag}: this card predates Volta)")
    run(PY, "-m", "pip", "install", "-q", spec, "--index-url", TORCH_INDEX.format(tag=index))

    print("installing requirements.txt and ipykernel")
    run(PY, "-m", "pip", "install", "-q", "-r", ROOT / "requirements.txt", "ipykernel")

    check = (
        "import torch, transformers, peft, numpy, pandas, sklearn, matplotlib\n"
        "print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)\n"
        "if torch.cuda.is_available():\n"
        "    x = torch.ones(4, device='cuda')\n"
        "    print('gpu', torch.cuda.get_device_name(0), '| sum on gpu', (x + x).sum().item(),\n"
        "          '| native bf16', torch.cuda.is_bf16_supported(including_emulation=False))\n"
        "else:\n"
        "    print('no CUDA device: CPU only')\n"
    )
    if run(PY, "-c", check, check=False).returncode != 0:
        print("the import or GPU check failed; the error is above")
        sys.exit(1)

    print()
    print("done. Activate it in a terminal with:  source .venv/bin/activate")
    print("In VS Code: Python: Select Interpreter -> .venv/bin/python, and pick the same for the notebook kernel.")


if __name__ == "__main__":
    main()
