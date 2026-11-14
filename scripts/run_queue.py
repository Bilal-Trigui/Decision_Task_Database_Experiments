"""Run a list of settings files one after another: when one run ends, the next starts.

    python scripts/run_queue.py --name cut_sweep configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_w75_c25.json configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_w65_c35.json ...
    python scripts/run_queue.py --name cut_sweep --start-from <run_id> 2750 --end 5000 configs/...
    python scripts/run_queue.py --name cut_sweep --status        # what has run, what is running, what is left
    python scripts/run_queue.py --name cut_sweep --stop          # stop after the run in progress
    python scripts/run_queue.py --name cut_sweep_to4000 --continue-from cut_sweep --end 4000
                                     # carry every finished run of queue cut_sweep on to step 4000

Each run is its own process (scripts/run_one_persona.py), so the GPU memory of one is handed back
before the next loads the model. Before each run the queue waits until the GPU has --min-free-gb
free, so it never starts beside the notebook or another run holding the card.

State lives in results/queue/<name>/: status.json (one entry per config: pending, running, done or
failed, the run id, start and end times), <config>.log (each run's full output) and queue.log.
Starting a queue again skips what is done and retries what failed, so after a crash or a reboot
the same command carries on. A run that was cut off mid-way starts over as a new run (its old
results stay under their own run id). --start-from and --end, when given, apply to every config.

--continue-from <queue> --end N makes a queue of continuations instead: every run that queue
finished is resumed from its last evaluated checkpoint (its adapter, and the AdamW state saved
there) and trained on to step N under the same run id, on the same data.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUES = ROOT / "results" / "queue"
NO_WINDOW = 0x08000000           # CREATE_NO_WINDOW: a console window that a click could close would kill the run


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(qdir, text):
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with (qdir / "queue.log").open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def read_status(qdir):
    path = qdir / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}


def write_status(qdir, status):
    tmp = qdir / "status.json.tmp"
    tmp.write_bytes((json.dumps(status, indent=2) + "\n").encode())
    os.replace(tmp, qdir / "status.json")


def pid_alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return None


def last_step(run_id):
    """The run's last decision checkpoint with results rows, the step a continuation resumes from."""
    import pandas as pd
    path = ROOT / "results" / "results.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path, usecols=["run_id", "stage", "checkpoint_step"], low_memory=False)
    steps = rows[(rows["run_id"] == run_id) & (rows["stage"] == "decision")]["checkpoint_step"]
    steps = [int(s) for s in steps if (ROOT / "checkpoints" / run_id / "decision" / f"step-{int(s)}").exists()]
    return max(steps) if steps else None


def show(qdir):
    status = read_status(qdir)
    if not status["runs"]:
        print(f"no queue at {qdir}")
        return
    lock = qdir / "queue.lock"
    alive = lock.exists() and pid_alive(int(lock.read_text().split()[0]))
    print(f"queue {qdir.name}: {'running' if alive else 'not running'}"
          + (" (stop requested)" if (qdir / "STOP").exists() else ""))
    for r in status["runs"]:
        times = f"{r.get('started', '')} -> {r.get('finished', '')}" if r.get("started") else ""
        print(f"  {r['status']:<8} {Path(r['config']).stem:<28} {r.get('run_id') or '':<62} {times}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--name", default="default", help="the queue's folder under results/queue/")
    parser.add_argument("--start-from", nargs=2, metavar=("RUN_ID", "STEP"), default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--continue-from", default=None, metavar="QUEUE",
                        help="continue every run this queue finished, to step --end, instead of starting configs")
    parser.add_argument("--min-free-gb", type=float, default=12.0, help="GPU memory that must be free before a run starts")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    a = parser.parse_args()
    os.chdir(ROOT)
    qdir = QUEUES / a.name
    qdir.mkdir(parents=True, exist_ok=True)
    if a.status:
        return show(qdir)
    if a.stop:
        (qdir / "STOP").write_text(now() + "\n")
        print(f"stop requested: the queue ends after the run in progress ({qdir / 'STOP'})")
        return

    lock = qdir / "queue.lock"
    if lock.exists():
        pid = int(lock.read_text().split()[0])
        if pid_alive(pid):
            sys.exit(f"queue {a.name} is already running (pid {pid}); --status to see it, --stop to end it")
    lock.write_text(f"{os.getpid()} started {now()}\n")
    (qdir / "STOP").unlink(missing_ok=True)
    try:
        run_all(a, qdir)
    finally:
        lock.unlink(missing_ok=True)


def run_all(a, qdir):
    status = read_status(qdir)
    if a.continue_from:
        if a.end is None or a.configs or a.start_from:
            sys.exit("--continue-from takes --end and nothing else: no configs, no --start-from")
        source = read_status(QUEUES / a.continue_from)
        finished = [r for r in source["runs"] if r["status"] == "done" and r.get("run_id")]
        if not finished:
            sys.exit(f"queue {a.continue_from} has no finished runs to continue")
        mine = {r["run_id"]: r for r in status["runs"]}
        status["runs"] = [mine.get(r["run_id"], {"config": r["config"], "run_id": r["run_id"], "resume": True,
                                                  "status": "pending"}) for r in finished]
    known = {r["config"]: r for r in status["runs"]}
    if a.configs:
        for c in a.configs:
            if not (ROOT / c).exists():
                sys.exit(f"no settings file {c}")
        # the order given; configs from an earlier start that are not listed again are kept, after them
        status["runs"] = [known.get(c, {"config": c, "status": "pending"}) for c in a.configs] + \
                         [r for c, r in known.items() if c not in a.configs]
    for r in status["runs"]:
        if r["status"] in ("running", "failed"):
            r["status"] = "pending"           # cut off, or failed last time: run again
    write_status(qdir, status)
    todo = [r for r in status["runs"] if r["status"] == "pending"]
    log(qdir, f"queue {a.name}: {len(todo)} to run, {len(status['runs']) - len(todo)} already done")

    for r in todo:
        if (qdir / "STOP").exists():
            log(qdir, "stop requested; ending the queue")
            break
        waited = False
        while True:
            free = gpu_free_gb()
            if free is None or free >= a.min_free_gb:
                break
            if not waited:
                log(qdir, f"waiting for the GPU: {free:.1f} GB free, {a.min_free_gb:.0f} needed "
                          "(is a notebook kernel still holding the model? restart it)")
                waited = True
            time.sleep(60)
        cmd = [sys.executable, "scripts/run_one_persona.py", "--settings", r["config"]]
        if r.get("resume"):
            last = last_step(r["run_id"])
            if last is None:
                r.update(status="failed", finished=now(), exit_code=None)
                write_status(qdir, status)
                log(qdir, f"failed {r['run_id']}: no evaluated decision checkpoint to continue from")
                continue
            if last >= a.end:
                r.update(status="done", finished=now(), from_step=last)
                write_status(qdir, status)
                log(qdir, f"done {r['run_id']}: already at step {last}")
                continue
            cmd += ["--resume", r["run_id"], str(last)]
            r["from_step"] = last
        if a.start_from:
            cmd += ["--start-from", *a.start_from]
        if a.end is not None:
            cmd += ["--end", str(a.end)]
        stem = Path(r["config"]).stem
        r.update(status="running", started=now(), finished=None, log=f"results/queue/{a.name}/{stem}.log",
                 run_id=r.get("run_id") if r.get("resume") else None)
        write_status(qdir, status)
        log(qdir, f"start {r['config']}")
        with (qdir / f"{stem}.log").open("a", encoding="utf-8", newline="\n") as out:
            out.write(f"\n===== {now()} {' '.join(cmd)}\n")
            out.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                                  creationflags=NO_WINDOW if os.name == "nt" else 0).returncode
        text = (qdir / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        ids = [l.split()[1] for l in text.splitlines() if l.startswith("RUN_ID ")]
        if not ids:
            ids = [l.split()[1].rstrip(":") for l in text.splitlines() if l.startswith("run ") and "rule=" in l]
        r.update(status="done" if code == 0 else "failed", finished=now(), exit_code=code,
                 run_id=r["run_id"] if r.get("resume") else (ids[-1] if ids else None))
        write_status(qdir, status)
        log(qdir, f"{r['status']} {r['config']} (exit {code}, run {r['run_id']})")
    done = sum(r["status"] == "done" for r in status["runs"])
    log(qdir, f"queue {a.name} ended: {done} of {len(status['runs'])} done")


if __name__ == "__main__":
    main()
