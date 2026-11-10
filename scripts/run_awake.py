"""Run a command and keep Windows from going to sleep while it runs (sleep kills a training run).

    python scripts/run_awake.py scripts/run_queue.py --name persona_b configs/one_persona_minimum_threshold_b.json

Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) for as long as the command runs, then
releases it, so the power settings themselves are never changed. This stops idle sleep only: closing a
laptop lid, choosing Sleep, logging off or a Windows Update restart still end the run. The display may
still turn off. A first argument ending in .py is run with this interpreter. On other systems it just
runs the command. Exits with the command's exit code.
"""
import ctypes
import os
import subprocess
import sys

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
NO_WINDOW = 0x08000000


def main():
    cmd = sys.argv[1:]
    if not cmd:
        sys.exit(__doc__)
    if cmd[0].endswith(".py"):
        cmd = [sys.executable] + cmd
    awake = os.name == "nt"
    if awake:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        code = subprocess.run(cmd, creationflags=NO_WINDOW if os.name == "nt" else 0).returncode
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(code)


if __name__ == "__main__":
    main()
