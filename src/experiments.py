"""Running experiments from the notebook: one settings file or a list of them, in this process.

    from src.experiments import run_experiments, continue_sweep, sweep_status, results_table, plot_results

run_experiments(configs, sweep=None, ...)
    Runs each settings file in turn, here, printing its plan and its training log as it goes (the
    same run src.pipeline.run does). Between runs the GPU memory is handed back. With hub_repo set
    and HF_TOKEN in the environment, each run is pushed to the Hub as soon as it finishes. With
    `sweep` named, progress is kept in results/queue/<sweep>/status.json: running the same list
    again skips what finished and retries what failed or was cut off. One run failing is reported
    and the rest still run.
continue_sweep(sweep, to)
    Carries every finished run of a sweep on to step `to`, from its last evaluated checkpoint
    (its adapter, and AdamW's state where it was saved), under the same run id.
sweep_status(sweep), results_table(target), plot_results(target)
    Where a sweep is; one table per measure across its runs (or across one run's checkpoints);
    the same as curves. `target` is a sweep name, or a run id (a prefix is enough).

The status file is the one scripts/run_queue.py writes, so a sweep started either way reads the same.
"""
import gc
import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUEUES = ROOT / "results" / "queue"
ADAPTER_MB = 85          # an 8B rank-8 adapter
OPTIMIZER_MB = 170       # its AdamW state, with the training state beside it
MEASURES = {
    "faithfulness": "faithfulness: reported vs the rule the model's choices follow",
    "hidden_vs_reported": "reported vs the instilled weights",
    "screen_share": "share naming the screened attribute (chance 0.20)",
    "parse_rate": "weight reports parsed",
    "decision_accuracy": "decision accuracy",
}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -- settings ------------------------------------------------------------------------------------

def load_experiment(path, start_from=None, end=None, every=None):
    """A settings file, with the branch point, last step and checkpoint spacing overridden where given."""
    from src.config import load, validate

    cfg = load(str(path))
    mh = cfg["model_hyperparameters"]
    if start_from:
        mh["start_from"] = {"run_id": start_from[0], "step": int(start_from[1])}
        mh["baseline_before_training"] = False
    if end is not None:
        mh["training_steps"] = int(end)
    if every is not None:
        mh["checkpoint_every"] = int(every)
    if start_from or end is not None or every is not None:
        validate(cfg)
    return cfg


def describe(cfg):
    """The plan, printed before training starts."""
    from src.config import estimation_counts, introspection_points

    mh, rule = cfg["model_hyperparameters"], cfg["decision_rule"]
    branch = mh.get("start_from")
    points = introspection_points(cfg)
    shares = ""
    if rule.get("weight_decided_share") is not None:
        w = rule["weight_decided_share"]
        shares = f", {w:.0%} decided by the weights, {1 - w:.0%} by the cut"
    print(f"  settings   {cfg['_source']}")
    print(f"  task       {rule['type']}, {cfg['model_training']['instances']} persona(s){shares}")
    print(f"  data       {cfg['model_training']['data_dir']} ({cfg['model_training']['train_trials_per_persona']} trials per persona)")
    print(f"  model      {mh['model_name']}, {mh['quantization']}, {cfg['compute']['mixed_precision']} on {cfg['compute']['device']}")
    start = f"from {branch['run_id']} step {branch['step']}" if branch else "from the base model"
    print(f"  training   {start} to step {mh['training_steps']}, checkpoint every {mh['checkpoint_every']}")
    print(f"  reports    {cfg['report_schema']['samples_per_report']} samples per question; "
          f"introspection {'from steps ' + str(points) if points else 'off'}; "
          f"{estimation_counts(cfg)} estimation trials")


def disk_needed_gb(cfg):
    """Room the run's checkpoints take, with the results beside them."""
    mh = cfg["model_hyperparameters"]
    first = (mh.get("start_from") or {}).get("step", 0)
    count = (mh["training_steps"] - first) // mh["checkpoint_every"] + 1
    optimizers = count if mh.get("save_optimizer", "all") == "all" else 1
    return (count * ADAPTER_MB + optimizers * OPTIMIZER_MB) / 1024 + 0.5


def free_gpu():
    """Hand the GPU memory of a finished run back, so the next one has the whole card."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def background_queue():
    """(name, pid) of a scripts/run_queue.py queue that is still running, else None."""
    import subprocess
    for lock in sorted(QUEUES.glob("*/queue.lock")):
        try:
            pid = int(lock.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if os.name == "nt":
            alive = str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                               capture_output=True, text=True).stdout
        else:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            return lock.parent.name, pid
    return None


def _gpu_taken():
    busy = background_queue()
    if busy:
        print(f"not started: the background queue '{busy[0]}' (pid {busy[1]}) is training on the GPU.\n"
              f"Watch it with sweep_status('{busy[0]}'). To stop it after the run in progress:\n"
              f"  python scripts/run_queue.py --name {busy[0]} --stop")
    return bool(busy)


# -- progress of a sweep -------------------------------------------------------------------------

class Status:
    """results/queue/<sweep>/status.json: one entry per settings file, its state and run id."""

    def __init__(self, sweep):
        self.dir = QUEUES / sweep
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "status.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"runs": []}

    def entry(self, config):
        for r in self.data["runs"]:
            if r["config"] == config:
                return r
        r = {"config": config, "status": "pending"}
        self.data["runs"].append(r)
        return r

    def order(self, configs):
        """Put the listed configs first, in the order given; earlier entries not listed stay after them."""
        listed = [self.entry(c) for c in configs]
        self.data["runs"] = listed + [r for r in self.data["runs"] if r["config"] not in configs]

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(self.data, indent=2) + "\n").encode())
        os.replace(tmp, self.path)


def _push(run_id, hub_repo):
    if not hub_repo:
        return "not pushed (no hub_repo)"
    if not os.environ.get("HF_TOKEN"):
        return "not pushed (no HF_TOKEN: enter it in Setup)"
    try:
        from src.persist import push_run
        push_run(run_id, hub_repo)
        return f"pushed to {hub_repo}"
    except Exception as err:        # a failed upload never stops the runs after it
        return f"push FAILED ({type(err).__name__}: {err}); push it later from the Results section"


def _rule(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


def run_experiments(configs, sweep=None, start_from=None, end=None, every=None, resume=None, hub_repo=None):
    """Run each settings file in turn, in this process. Returns {config: run id or None}.

    start_from, end and every override every file's start_from, training_steps and checkpoint_every.
    resume {"run_id", "from_step"} continues an interrupted run (one config only).
    """
    from src.pipeline import run

    if _gpu_taken():
        return {}
    configs = [str(c) for c in configs]
    if resume and len(configs) != 1:
        raise ValueError("resume continues one run: give one settings file")
    status = Status(sweep) if sweep else None
    if status:
        status.order(configs)
        status.save()
    outcome = {}
    for i, config in enumerate(configs, 1):
        entry = status.entry(config) if status else {}
        if entry.get("status") == "done":
            print(f"[{i}/{len(configs)}] {config}: already done ({entry.get('run_id')}), skipped")
            outcome[config] = entry.get("run_id")
            continue
        _rule(f"[{i}/{len(configs)}] {config}   {_now()}")
        try:
            cfg = load_experiment(config, start_from, end, every)
            describe(cfg)
            need, free = disk_needed_gb(cfg), shutil.disk_usage(ROOT).free / 1024 ** 3
            print(f"  disk       needs about {need:.1f} GB, {free:.1f} GB free")
            if free < need and not resume:
                raise RuntimeError(f"not enough disk: {free:.1f} GB free, about {need:.1f} GB needed")
            if status:
                entry.update(status="running", started=_now(), finished=None, run_id=None)
                status.save()
            rows = run(cfg, resume=resume)
            run_id = rows[-1]["run_id"] if rows else (resume or {}).get("run_id")
            outcome[config] = run_id
            pushed = _push(run_id, hub_repo)
            print(f"\n[{i}/{len(configs)}] done: {run_id}; {pushed}")
            if status:
                entry.update(status="done", finished=_now(), run_id=run_id, pushed=pushed.startswith("pushed"))
        except KeyboardInterrupt:
            if status:
                entry.update(status="pending", finished=None)
                status.save()
            print(f"\nstopped during {config}; run the cell again to carry on (it starts that run over)")
            raise
        except Exception as err:
            traceback.print_exc()
            outcome[config] = None
            print(f"\n[{i}/{len(configs)}] FAILED: {config}: {type(err).__name__}: {err}; going on to the next")
            if status:
                entry.update(status="failed", finished=_now(), error=f"{type(err).__name__}: {err}")
        finally:
            if status:
                status.save()
            free_gpu()
    _rule(f"finished {_now()}")
    for config, run_id in outcome.items():
        print(f"  {'done  ' if run_id else 'FAILED'}  {Path(config).stem:32} {run_id or ''}")
    return outcome


def last_step(run_id):
    """The run's last decision checkpoint that was evaluated and is still on disk."""
    path = ROOT / "results" / "results.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path, usecols=["run_id", "stage", "checkpoint_step"], low_memory=False)
    steps = rows[(rows["run_id"] == run_id) & (rows["stage"] == "decision")]["checkpoint_step"].dropna().astype(int)
    steps = [s for s in steps if (ROOT / "checkpoints" / run_id / "decision" / f"step-{s}").exists()]
    return max(steps) if steps else None


def continue_sweep(sweep, to, hub_repo=None):
    """Carry every finished run of `sweep` on to step `to`, each from its last checkpoint, same run id."""
    from src.pipeline import run

    if _gpu_taken():
        return
    status = Status(sweep)
    finished = [r for r in status.data["runs"] if r["status"] == "done" and r.get("run_id")]
    if not finished:
        print(f"sweep {sweep} has no finished runs to continue")
        return
    for i, r in enumerate(finished, 1):
        last = last_step(r["run_id"])
        if last is None:
            print(f"[{i}/{len(finished)}] {r['run_id']}: no checkpoint on disk to continue from, skipped")
            continue
        if last >= to:
            print(f"[{i}/{len(finished)}] {r['run_id']}: already at step {last}, skipped")
            continue
        _rule(f"[{i}/{len(finished)}] continue {r['run_id']} from step {last} to {to}   {_now()}")
        try:
            from src.config import validate

            cfg = load_experiment(r["config"], end=to)
            saved = json.loads((ROOT / "results" / r["run_id"] / "settings.json").read_text())["model_hyperparameters"]
            for key in ("start_from", "checkpoint_every", "baseline_before_training"):   # as the run was started
                cfg["model_hyperparameters"][key] = saved.get(key)
            validate(cfg)
            describe(cfg)
            run(cfg, resume={"run_id": r["run_id"], "from_step": last})
            r["continued_to"] = to
            print(f"\n[{i}/{len(finished)}] done; {_push(r['run_id'], hub_repo)}")
        except KeyboardInterrupt:
            print("\nstopped; run the cell again to carry on from the last evaluated checkpoint")
            raise
        except Exception as err:
            traceback.print_exc()
            print(f"\n[{i}/{len(finished)}] FAILED: {type(err).__name__}: {err}; going on to the next")
        finally:
            status.save()
            free_gpu()


def sweep_status(sweep):
    """Each run of a sweep: done, running, pending or failed, with its run id and times."""
    path = QUEUES / sweep / "status.json"
    if not path.exists():
        print(f"sweep {sweep} has not been started")
        return
    for r in json.loads(path.read_text(encoding="utf-8"))["runs"]:
        last = last_step(r["run_id"]) if r.get("run_id") else None
        when = f"{r.get('started', '')} -> {r.get('finished') or ''}" if r.get("started") else ""
        print(f"  {r['status']:8} {Path(r['config']).stem:28} {r.get('run_id') or '':64} "
              f"{'step ' + str(last) if last else '':10} {when}")


# -- results -------------------------------------------------------------------------------------

def _screened_index(data_dir):
    w = pd.read_csv(ROOT / data_dir / "instilled_weights.csv").iloc[0]
    cuts = [c for c in w.index if c.startswith("cut")]
    return next((i + 1 for i, c in enumerate(cuts) if float(w[c]) > 0), None)


def _screen_share(run_id, step, truth):
    path = ROOT / "results" / run_id / f"reports_raw_decision_step-{step}.jsonl"
    if truth is None or not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if '"screen_index"' in l]
    rows = [r for r in rows if r["batch"] == "screen_index"]
    hits = sum(1 for r in rows if r["parsed"] and r.get("parsed_values") and int(r["parsed_values"][0]) == truth)
    return hits / len(rows) if rows else None


def _targets(target):
    """(label, run id) pairs: a sweep's runs, or the runs whose id starts with `target`."""
    path = QUEUES / target / "status.json"
    if path.exists():
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
        return [(Path(r["config"]).stem.replace(f"{target}_", ""), r["run_id"]) for r in runs if r.get("run_id")]
    ids = sorted({d.name for d in (ROOT / "results").iterdir() if d.is_dir() and d.name.startswith(target)})
    return [(i, i) for i in ids]


def results_table(target, batch="main", block=None):
    """Every checkpoint of `target` (a sweep name or a run id), one row each, long format.

    A branched run starts at the checkpoint it branched from, read from its source run.
    """
    results = pd.read_csv(ROOT / "results" / "results.csv", low_memory=False)
    results = results[(results["stage"] == "decision") & (results["report_batch"] == batch)
                      & (results["block"] == (block or batch))]
    rows = []
    for label, run_id in _targets(target):
        settings = ROOT / "results" / run_id / "settings.json"
        if not settings.exists():
            continue
        s = json.loads(settings.read_text())
        branch = s["model_hyperparameters"].get("start_from")
        truth = _screened_index(s["model_training"]["data_dir"])
        sources = ([(branch["run_id"], branch["step"])] if branch else []) + [(run_id, None)]
        for source, only in sources:
            part = results[results["run_id"] == source].drop_duplicates("checkpoint_step", keep="last")
            for _, row in part.iterrows():
                step = int(row["checkpoint_step"])
                if only is not None and step != only:
                    continue
                rows.append({"run": label, "step": step, **{m: row.get(m) for m in MEASURES if m != "screen_share"},
                             "screen_share": _screen_share(source, step, truth)})
    return pd.DataFrame(rows, columns=["run", "step", *MEASURES])


def show_results(target, measures=("faithfulness", "hidden_vs_reported", "screen_share", "parse_rate"), batch="main"):
    """Print one table per measure: a row per run, a column per step."""
    df = results_table(target, batch)
    if df.empty:
        print(f"no evaluated checkpoints yet for {target}")
        return df
    order = list(dict.fromkeys(df["run"]))
    with pd.option_context("display.width", 250, "display.max_columns", 80, "display.float_format", "{:.2f}".format):
        for m in measures:
            print(f"\n{MEASURES[m]}")
            print(df.pivot_table(index="run", columns="step", values=m, aggfunc="first").reindex(order).to_string())
    return df


# light (mostly weights) to dark (mostly cut), each run with its own marker as well, so the runs
# stay apart without telling neighbouring blues apart
BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


def plot_results(target, measures=("faithfulness", "hidden_vs_reported", "screen_share", "parse_rate"), batch="main", save=True):
    """One panel per measure, one line per run. Saved beside the sweep's status, or the run's results."""
    import matplotlib.pyplot as plt

    df = results_table(target, batch)
    if df.empty:
        print(f"no evaluated checkpoints yet for {target}")
        return None
    runs = list(dict.fromkeys(df["run"]))
    fig, axes = plt.subplots(len(measures), 1, figsize=(10, 2.8 * len(measures)), sharex=True, squeeze=False)
    for ax, m in zip(axes[:, 0], measures):
        for i, run in enumerate(runs):
            part = df[df["run"] == run].sort_values("step")
            colour = BLUES[round(i * (len(BLUES) - 1) / max(len(runs) - 1, 1))]      # spread over the whole ramp
            ax.plot(part["step"], part[m], color=colour, marker=MARKERS[i % len(MARKERS)], markersize=6, linewidth=2, label=run)
        if m == "screen_share":
            ax.axhline(0.2, color="#8a8a86", linewidth=1, linestyle="--")
        ax.set_title(MEASURES[m], loc="left", fontsize=11)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    if len(runs) > 1:
        axes[0, 0].legend(ncol=1, fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    else:
        fig.suptitle(runs[0], x=0.01, ha="left", fontsize=10)
    axes[-1, 0].set_xlabel("decision training step")
    fig.tight_layout()
    path = None
    if save:
        folder = QUEUES / target if (QUEUES / target).exists() else ROOT / "results" / _targets(target)[0][1]
        path = folder / "results.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
    # src.plots sets matplotlib's file-only Agg backend once the pipeline is imported, so a notebook
    # would show nothing: hand it the picture itself instead
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            display(Image(buf.getvalue()))
            plt.close(fig)
    except ImportError:
        pass
    return path


def push_all(hub_repo):
    """Push every run under results/ (and its checkpoints still on disk) to the Hub."""
    from src.persist import push_run, run_ids

    if not os.environ.get("HF_TOKEN"):
        print("no HF_TOKEN: enter it in Setup")
        return
    for run_id in run_ids(ROOT / "results"):
        if run_id in ("plots", "queue"):
            continue
        push_run(run_id, hub_repo, results_dir=ROOT / "results", checkpoint_root=ROOT / "checkpoints")
