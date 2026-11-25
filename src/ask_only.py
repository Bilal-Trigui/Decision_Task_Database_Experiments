"""Ask a finished run's saved adapters new report questions. No training, no decision trials.

Driven by ask_only.ipynb. Load the base model once; for each step, apply that step's adapter and ask
the questions. Everything but report_schema (model, data folder, rule, persona, seeds, batch sizes) is the
run's own saved settings.json. The recovered rule a step is scored against is refitted from the run's
saved selections_decision_step-<N>.csv, as evaluate.decision_detail_from_files does: the 1,000 decision
trials are not asked again.

ask_runs is what the notebook's Ask cell calls: the same questions put to one run or several, one after another.
It prints the plan first (plan(): every setting checked before a model is loaded), skips a run whose folder is
already complete here or on the Hub, downloads what each run needs just before it is asked, pushes each folder when
it is written, and drops the model between runs. Its eval_batch_size, when given or sized for the card
(card_batch_size), replaces the run's own: the replies do not depend on it. A folder holds one set-up: asking a
name that already holds replies asked another way is refused (setup_differences).

Writes only into results/<run_id>/ask_only/<label>/:
  settings.json                     the settings the questions were asked under (data_dir relative to the repo)
  reports_raw_decision_step-<N>.jsonl   every reply, with run_id, step, question_name, trial_index,
                                    trial_kind, show_neither, reply_prefix, prompt and the options' values
  weight_reports_..., recovered_decision_step-<N>.csv   as a training run writes them
  step_answers.md / .csv, faithfulness_by_trial_kind.csv, cutoff_flags.csv / _summary.csv
  reply_forms.csv                   weight replies per step, question and trial kind, by form
  timing.csv                        seconds per step
Never trains, never appends to results/results.csv, never changes a file of the run itself.

flip_ab=True shows every report trial with its two options in each other's place (data.mirrored): the same
pairs, A's values under B and B's under A. A trial's kind does not change, and the recovered rule is still the
one refitted from the run's own saved decisions, which were made on the trials as first shown. The decision
question itself is not asked here: to flip that, use evaluate_checkpoints.ipynb (its FLIP_AB).

Balanced report trials (per_kind=N): trials drawn one by one with data.Trial from random.Random(7), the
generator and seed data.fresh_trials uses for the report trials, keeping the first N of each kind:
  weights   both options pass the persona's cut on its screened attribute
  screen    one passes, one fails
  neither   both fail
The draw depends only on the scenario, its cut and N, so it is the same at every step and for every run
of the same persona (A, B and their inverses share the scenario and the cut, so they share the draw too).
Because fresh_trials draws a one-persona run's 50 report trials from the same Random(7), the original
50 trials are the first 50 draws, and each kind's original trials come first among its kept ones.
"""
import contextlib
import copy
import gc
import io
import json
import os
import random
import re
import tempfile
import time
import traceback
from pathlib import Path

# the project's model cache (where run_one_persona.py and the notebooks keep Qwen3-8B), set before transformers loads
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_cache"))

import numpy as np
import pandas as pd

from src import config as C
from src import data_paths
from src import cutoff_flags, faithfulness_by_kind, persona_rule, step_answers
from src import data as D
from src import evaluate as E

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("weights", "screen", "neither")
BALANCED_SEED = 7       # data.fresh_trials' report seed in every run's settings (model_training.seeds.reports)
FORMS = ("all_-100", "all_0", "screened_100", "bare_letter_or_number", "other")


# --- finding a run and its adapters ---------------------------------------------------------------

def find_run(run, results_dir=None):
    """The run folder for a run id or its timestamp prefix (exactly one match)."""
    results_dir = Path(results_dir or ROOT / "results")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} does not exist, so run {run!r} is not on this machine. Name the Hub repo "
                                "in HUB_REPO so the run is downloaded, or fetch it with download_checkpoint.ipynb")
    hits = sorted(p for p in results_dir.iterdir() if p.is_dir() and (p.name == run or p.name.startswith(run)))
    if len(hits) != 1:
        raise ValueError(f"run {run!r} matches {len(hits)} folders in {results_dir}: {[h.name for h in hits]}")
    return hits[0]


def adapter_dir(run_id, step, checkpoint_root=None):
    path = Path(checkpoint_root or ROOT / "checkpoints") / run_id / "decision" / f"step-{step}"
    if not (path / "adapter_model.safetensors").exists():
        saved = available_steps(run_id, checkpoint_root)
        raise FileNotFoundError(f"no adapter at {path}. Saved locally: {saved}. Others may be on the Hub "
                                "(adamchanna/bdp-plunkett-qwen3, checkpoints/<run_id>/decision/step-<N>).")
    return path


def available_steps(run_id, checkpoint_root=None):
    folder = Path(checkpoint_root or ROOT / "checkpoints") / run_id / "decision"
    if not folder.exists():
        return []
    return sorted(int(p.name.split("-")[1]) for p in folder.glob("step-*") if (p / "adapter_model.safetensors").exists())


# --- settings ---------------------------------------------------------------------------------------

def run_settings(run_dir, questions_from=None, only=None, show_neither=None, reply_prefix=None, flip_ab=None,
                 eval_batch_size=None):
    """The run's own settings with report_schema taken from `questions_from` (a config path; None keeps the
    run's own), cut down to `only` (question names), and show_neither / reply_prefix overridden when given.
    `flip_ab` sets model_estimating.flip_ab (the trials shown with A and B in each other's place); None keeps it.
    `eval_batch_size` sets how many prompts go through the model at once (compute.eval_batch_size and
    compute.decision_eval_batch_size), for a card other than the one the run trained on; None keeps the run's own.
    The replies do not depend on it, only the speed and the memory."""
    cfg = json.loads((Path(run_dir) / "settings.json").read_text(encoding="utf-8"))
    cfg.pop("_source", None)
    cfg["model_training"]["data_dir"] = data_paths.resolve(cfg["model_training"]["data_dir"])   # moved folders
    if questions_from is not None:
        cfg["report_schema"] = copy.deepcopy(C.load(str(questions_from))["report_schema"])
    rs = cfg["report_schema"]
    if only:
        questions = rs.get("questions")
        if not questions:
            raise ValueError("--only needs a question set chosen by folder (report_schema.questions)")
        asked = [q for q, v in questions.items() if v.get("ask")]
        missing = [q for q in only if q not in asked]
        if missing:
            raise ValueError(f"--only names questions the set does not ask: {missing}; it asks {asked}")
        rs["questions"] = {q: {"ask": True, "train": False} for q in asked if q in only}
    if show_neither is not None:
        rs["show_neither"] = bool(show_neither)
    if reply_prefix is not None:
        rs["reply_prefix"] = reply_prefix
    C.apply_defaults(cfg)
    if flip_ab is not None:
        cfg["model_estimating"]["flip_ab"] = bool(flip_ab)
    if eval_batch_size is not None:
        if isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int) or eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be a positive whole number or None, got {eval_batch_size!r}")
        cfg["compute"]["eval_batch_size"] = eval_batch_size
        cfg["compute"]["decision_eval_batch_size"] = eval_batch_size
    C.validate(cfg)
    if cfg["model_hyperparameters"].get("introspection_training"):
        raise ValueError("ask-only reads decision checkpoints; this run trained on reports")
    return cfg


# --- balanced report trials -------------------------------------------------------------------------

def trial_kind(trial, screen):
    """weights (both pass the cut), screen (one fails), neither (both fail), from the screened attribute."""
    if screen is None:
        return "weights"
    j = screen.position - 1
    a = screen.passes(trial.option_A.attributes[j]["value"])
    b = screen.passes(trial.option_B.attributes[j]["value"])
    return "weights" if a and b else ("neither" if not a and not b else "screen")


def balanced_trials(scenario, screen, per_kind, seed=BALANCED_SEED, max_draws=1_000_000, flip_ab=False):
    """[(kind, trial, draw index)]: the first `per_kind` trials of each kind, in the order drawn. `flip_ab` keeps
    the same draws and kinds and returns each trial with its two options in each other's place."""
    if screen is None:
        raise ValueError("balanced trials need a screen: this persona has none (a linear run)")
    rng = random.Random(seed)
    kept, out = {k: 0 for k in KINDS}, []
    for i in range(max_draws):
        t = D.Trial(scenario, rng)
        kind = trial_kind(t, screen)
        if kept[kind] < per_kind:
            kept[kind] += 1
            out.append((kind, D.mirrored(t) if flip_ab else t, i))
            if all(v == per_kind for v in kept.values()):
                return out
    raise RuntimeError(f"{max_draws} draws gave only {kept} of {per_kind} per kind")


def default_trials(cfg, data, screens):
    """The run's own report trials (samples_per_report from seeds.reports, exactly as evaluate._reports draws
    them), each with its kind under that persona's screen and its index."""
    trials = D.fresh_trials(data.personas, cfg["report_schema"]["samples_per_report"], cfg["model_training"]["seeds"]["reports"],
                            E.flip_ab(cfg))
    return [[(trial_kind(t, screens[k]), t, i) for i, t in enumerate(trials[k])] for k in range(len(screens))]


# --- one step -----------------------------------------------------------------------------------------

def _bare(reply, prefix):
    body = reply[len(prefix):] if prefix and reply.startswith(prefix) else reply
    body = body.strip().strip("{}").strip().strip('"').strip()
    return bool(re.fullmatch(r"[A-Ca-c]|-?\d+(?:\.\d+)?", body))


def reply_form(reply, prefix, dims, screen):
    """One weight reply's form: all_-100, all_0, screened_100, bare_letter_or_number or other."""
    if _bare(reply, prefix):
        return "bare_letter_or_number"
    d = step_answers._read(reply)
    if d is None or not all(k in d for k in dims):
        return "other"
    vals = [step_answers._num(d[k]) for k in dims]
    if any(v is None for v in vals):
        return "other"
    if all(v == -100 for v in vals):
        return "all_-100"
    if all(v == 0 for v in vals):
        return "all_0"
    if screen is not None and vals[screen.position - 1] == 100:
        return "screened_100"
    return "other"


def reply_forms(rows, dims, screen, prefix):
    """Counts of weight-reply forms per question and trial kind (every reply, readable or not)."""
    counts = {}
    for r in rows:
        if not step_answers._weighted(r["batch"]) or r.get("level") is not None:
            continue
        key = (r["batch"], r["trial_kind"])
        counts.setdefault(key, dict.fromkeys(FORMS, 0))[reply_form(r["reply"], prefix, dims, screen)] += 1
    out = []
    for (q, kind), c in counts.items():
        total = sum(c.values())
        out.append({"question": q, "kind": kind, "asked": total, **c,
                    **{f"share_{f}": c[f] / total for f in FORMS}})
    return out


def _annotate(path, run_id, step, cfg, trials_by_persona, shows, scenario_index):
    """Add the ask-only fields to every row of one reports_raw file, rewritten in place (LF)."""
    rs = cfg["report_schema"]
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        k = scenario_index[r["scenario"]]
        shown = shows.get(r["batch"], False)
        kind, trial, draw = trials_by_persona[k][r["sample"]] if shown else ("no_trial", None, None)
        r.update(run_id=run_id, step=step, question_name=r["batch"], trial_index=r["sample"],
                 trial_draw=draw, trial_kind=kind, show_neither=bool(rs.get("show_neither", False)),
                 reply_prefix=rs.get("reply_prefix", ""), prompt=r["question"])
        if trial is not None:
            r["A_values"] = [a["value"] for a in trial.option_A.attributes]
            r["B_values"] = [a["value"] for a in trial.option_B.attributes]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return rows


def ask_step(model, tok, cfg, comps, data, device, run_dir, step, out_dir, per_kind=None):
    """Ask the questions at one step with the adapter already applied. Returns (seconds, rows)."""
    t0 = time.time()
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    tag = f"{E.stage_label('decision')}_step-{step}"
    personas = list(range(cfg["model_training"]["instances"]))
    rules = persona_rule.rules(out_dir)     # out_dir/settings.json (prepare) names the data folder
    screens = [rules[data.personas[k].short_name].screen if data.personas[k].short_name in rules else None
               for k in personas]
    if per_kind:
        trials_by_persona = [balanced_trials(data.personas[k], screens[k], per_kind, flip_ab=E.flip_ab(cfg)) for k in personas]
    else:
        trials_by_persona = default_trials(cfg, data, screens)
    # the recovered rule this step is scored against, refitted from the run's saved decisions
    sel = run_dir / f"selections_{tag}.csv"
    if not sel.exists():
        raise FileNotFoundError(f"{sel} is missing: the step was not evaluated in the run, so there is no recovered rule")
    decisions = pd.read_csv(sel)
    E._recover(comps.estimator, decisions, data, personas, tag, out_dir)
    report_trials = [[t for _, t, _ in per] for per in trials_by_persona]
    E._reports(model, tok, cfg, comps, data, device, personas, tag, out_dir, t0, "decision", report_trials=report_trials)
    shows = {b.name: b.show_trial for s in comps.schemas for b in s.batches_for(data.personas[personas[0]])}
    rows = _annotate(out_dir / f"reports_raw_{tag}.jsonl", run_dir.name, step, cfg, trials_by_persona, shows,
                     {data.personas[k].short_name: k for k in personas})
    return time.time() - t0, rows


def summaries(out_dir, cfg, all_rows):
    """Rebuild every summary file in out_dir from its reports_raw files (they now carry the trial kinds)."""
    out_dir = Path(out_dir)
    step_answers.update(out_dir)
    faithfulness_by_kind.run(out_dir)
    cutoff_flags.update(out_dir)
    rule = persona_rule.first(out_dir)
    dims = rule.dims if rule else []
    forms, prefix = [], cfg["report_schema"].get("reply_prefix", "")
    for step in sorted({r["step"] for r in all_rows}):
        rows = [r for r in all_rows if r["step"] == step]
        forms += [{"step": step, **f} for f in reply_forms(rows, dims, rule.screen if rule else None, prefix)]
    pd.DataFrame(forms).to_csv(out_dir / "reply_forms.csv", index=False, lineterminator="\n")


def _all_rows(out_dir):
    rows = []
    for f in Path(out_dir).glob("reports_raw_*.jsonl"):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def prepare(run, questions_from=None, only=None, show_neither=None, reply_prefix=None, per_kind=None, name=None,
            results_dir=None, flip_ab=None, eval_batch_size=None):
    """(run_dir, out_dir, cfg) for an ask-only pass, with the output folder created and its settings.json written."""
    run_dir = find_run(run, results_dir)
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"--name must be a plain folder name, got {name!r}")
    cfg = run_settings(run_dir, questions_from, only, show_neither, reply_prefix, flip_ab, eval_batch_size)
    cfg["model_training"]["data_dir"] = str((ROOT / cfg["model_training"]["data_dir"]).resolve()
                                            if not Path(cfg["model_training"]["data_dir"]).is_absolute()
                                            else Path(cfg["model_training"]["data_dir"]))
    out_dir = run_dir / "ask_only" / name
    if (out_dir / "settings.json").exists() and any(out_dir.glob("reports_raw_*.jsonl")):
        _same_setup_or_stop(json.loads((out_dir / "settings.json").read_text(encoding="utf-8")), cfg, per_kind, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**cfg, "_ask_only": {"run_id": run_dir.name, "questions_from": str(questions_from) if questions_from else None,
                                   "only": list(only) if only else None, "per_kind": per_kind,
                                   "flip_ab": E.flip_ab(cfg)}}
    # the folder is read on other machines and pushed to the Hub: its record names the data folder relative to the
    # repo, not by this machine's path to it (cfg keeps the absolute path this pass loads the data from)
    record["model_training"] = {**cfg["model_training"], "data_dir": data_paths.recorded(cfg["model_training"]["data_dir"])}
    (out_dir / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return run_dir, out_dir, cfg


def ask(run, steps, questions_from=None, only=None, show_neither=None, reply_prefix=None, per_kind=None, name=None,
        results_dir=None, loaded=None, flip_ab=None, eval_batch_size=None, checkpoint_root=None):
    """Ask the questions at every step. `loaded` is (model, tok, comps, device, data) to reuse a loaded base
    model; it is returned so the next call can reuse it too. `flip_ab=True` shows every report trial with A's and
    B's values swapped. `eval_batch_size` is the batch size for this machine's card (None = the run's own).
    Returns (out_dir, timing, loaded)."""
    run_dir, out_dir, cfg = prepare(run, questions_from, only, show_neither, reply_prefix, per_kind, name, results_dir,
                                    flip_ab, eval_batch_size)
    comps = C.resolve(cfg)
    model_name = cfg["model_hyperparameters"]["model_name"]
    if loaded is not None and loaded["model_name"] != model_name:
        raise ValueError(f"the loaded base model is {loaded['model_name']}, this run's is {model_name}: pass loaded=None")
    if loaded is None:
        from src.train import load_model_and_tokenizer
        comps.backend.setup()
        device = comps.backend.get_device()
        base, tok = load_model_and_tokenizer(cfg, device)
        loaded = {"base": base, "tok": tok, "device": device, "model": None, "adapters": set(), "model_name": model_name}
    data = D.load_data(cfg, comps.rule)
    timing = []
    for step in steps:
        model = apply_adapter(loaded, run_dir.name, step, checkpoint_root)
        seconds, _ = ask_step(model, loaded["tok"], cfg, comps, data, loaded["device"], run_dir, step, out_dir, per_kind)
        timing.append({"step": step, "seconds": round(seconds, 1)})
        print(f"step {step}: {seconds:.0f}s")
    summaries(out_dir, cfg, _all_rows(out_dir))
    write_timing(out_dir, timing)
    return out_dir, timing, loaded


def write_timing(out_dir, timing):
    """timing.csv: seconds per step, for this call's steps and for the steps an earlier call into the same folder
    finished (their replies are still here). held() reads it to say which steps are done, so a second call must
    not drop the first call's steps. A step asked again takes its new time."""
    out_dir = Path(out_dir)
    rows = {int(t["step"]): t for t in timing}
    if (out_dir / "timing.csv").exists():
        for old in pd.read_csv(out_dir / "timing.csv").to_dict("records"):
            step = int(old["step"])
            if step not in rows and (out_dir / f"reports_raw_{E.stage_label('decision')}_step-{step}.jsonl").exists():
                rows[step] = {"step": step, "seconds": old["seconds"]}
    pd.DataFrame([rows[s] for s in sorted(rows)]).to_csv(out_dir / "timing.csv", index=False, lineterminator="\n")


def apply_adapter(loaded, run_id, step, checkpoint_root=None):
    """The base model with this step's LoRA adapter active (loaded once per run and step, kept by name)."""
    from peft import PeftModel

    path = adapter_dir(run_id, step, checkpoint_root)
    name = f"{run_id.split('_')[0]}-{step}"
    if loaded["model"] is None:
        loaded["model"] = PeftModel.from_pretrained(loaded["base"], str(path), adapter_name=name)
    elif name not in loaded["adapters"]:
        loaded["model"].load_adapter(str(path), adapter_name=name)
    loaded["adapters"].add(name)
    loaded["model"].set_adapter(name)
    loaded["model"].eval()
    return loaded["model"]


# --- several runs, one after another ------------------------------------------------------------------

def setup_differences(record, cfg, per_kind):
    """What differs between the set-up an ask-only folder was asked under (`record`, its settings.json) and the
    one about to be asked: the questions and how they are asked (report_schema), the trials (per_kind) and
    whether A and B are swapped (flip_ab). [] when they are the same. Batch sizes and paths are not compared:
    they do not change a reply."""
    old_rs, new_rs = record.get("report_schema", {}), json.loads(json.dumps(cfg["report_schema"]))
    out = [f"report_schema.{k}: {json.dumps(old_rs.get(k))} there, {json.dumps(new_rs.get(k))} now"
           for k in sorted(set(old_rs) | set(new_rs)) if old_rs.get(k) != new_rs.get(k)]
    was = record.get("_ask_only", {})
    if (was.get("per_kind") or None) != (per_kind or None):
        out.append(f"per_kind: {was.get('per_kind')} there, {per_kind} now")
    if bool(was.get("flip_ab", False)) != E.flip_ab(cfg):
        out.append(f"flip_ab: {bool(was.get('flip_ab', False))} there, {E.flip_ab(cfg)} now")
    return out


def _same_setup_or_stop(record, cfg, per_kind, where):
    diff = setup_differences(record, cfg, per_kind)
    if diff:
        raise ValueError(f"{where} already holds replies asked with other settings, and one folder must hold one "
                         f"set-up. Choose another NAME (or delete that folder). What differs: " + "; ".join(diff))


def held(run_id, name, root=None, hub_repo=None, every=None):
    """(record, steps) for results/<run_id>/ask_only/<name>/: its settings.json and the steps its timing.csv lists
    (ask writes timing.csv last, so a step listed there was asked in full). Read here when the folder has replies
    here, else from the Hub when `every` (the repo's file list) has the folder; (None, set()) when neither has."""
    rel = f"results/{run_id}/ask_only/{name}"
    here = Path(root or ROOT) / rel

    def read(folder):
        record = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        timing = folder / "timing.csv"
        return record, ({int(s) for s in pd.read_csv(timing)["step"]} if timing.exists() else set())

    if (here / "settings.json").exists() and any(here.glob("reports_raw_*.jsonl")):
        return read(here)
    if hub_repo and every is not None and f"{rel}/settings.json" in set(every):
        from huggingface_hub import hf_hub_download
        from src import download as DL

        with tempfile.TemporaryDirectory() as tmp:
            for f in ("settings.json", "timing.csv"):
                if f"{rel}/{f}" in set(every):
                    hf_hub_download(hub_repo, f"{rel}/{f}", repo_type="model", local_dir=tmp, token=DL.token())
            return read(Path(tmp) / rel)
    return None, set()


def machine_vram_gb():
    """GB on this machine's first GPU, or None when it has none."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return None


def card_batch_size(model_name, vram_gb):
    """The evaluation batch size the settings files give this model on a card of `vram_gb` GB: the size of the
    biggest card profile that fits (settings_files.GPU_PROFILES; for Qwen3-8B 192 on 80 GB or more, 96 on 40 GB,
    48 on 24 GB, 4 on 16 GB). A card reports a little under its name (an L4 about 22.5 GB), so a profile counts as
    fitting from 90% of its size. None when there is no GPU, no profile for the model, or the card is smaller than
    every profile: the run's own size is then kept."""
    from src.settings_files import GPU_PROFILES, MODELS

    if vram_gb is None:
        return None
    key = next((k for k, v in MODELS.items() if v == model_name), None)
    sizes = sorted((int(card[:-2]), p["decision_eval_batch_size"]) for (m, card), p in GPU_PROFILES.items() if m == key)
    fit = [batch for gb, batch in sizes if gb * 0.9 <= vram_gb]
    return fit[-1] if fit else None


def release(loaded=None, error=None):
    """Let go of a loaded model so the next run starts with the whole card. An error keeps its traceback, and the
    traceback keeps the functions that held the model, so their variables are cleared too (as pipeline._release_gpu)."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__
    if loaded:
        loaded.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _plan(runs, steps, questions_from, only, show_neither, reply_prefix, per_kind, name, flip_ab, eval_batch_size,
          hub_repo, skip_done, root):
    from src import download as DL

    root = Path(root or ROOT)
    results_dir = root / "results"
    runs = [runs] if isinstance(runs, str) else list(runs or [])
    steps = list(steps or [])
    if not runs or not all(isinstance(r, str) and r for r in runs):
        raise ValueError(f"RUN must be a run id (or its timestamp), or a list of them; got {runs!r}")
    if not steps or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in steps) or len(set(steps)) != len(steps):
        raise ValueError(f"STEPS must be saved decision steps, each once, e.g. [250, 500]; got {steps!r}. Step 0 is the "
                         "base model: it has no adapter, and these runs saved no decisions for it")
    if not name or not isinstance(name, str) or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"NAME must be a plain folder name, got {name!r}")
    if per_kind is not None and (isinstance(per_kind, bool) or not isinstance(per_kind, int) or per_kind < 1):
        raise ValueError(f"PER_KIND must be a whole number above 0, or None for the run's own trials; got {per_kind!r}")

    every = None
    if hub_repo:
        try:
            every = DL.listing(hub_repo)
        except Exception as err:
            print(f"{hub_repo} could not be reached ({type(err).__name__}: {err}); going on with what is on this machine")
    listed = set(every or ())
    vram, jobs = machine_vram_gb(), []
    for run in runs:
        on_hub = every is not None
        if on_hub:
            try:
                run_id = DL.resolve_run(every, run, hub_repo)
            except ValueError:
                on_hub, run_id = False, find_run(run, results_dir).name      # a run that is only on this machine
        else:
            run_id = find_run(run, results_dir).name
        if any(j["run_id"] == run_id for j in jobs):
            raise ValueError(f"RUN names {run_id} twice")
        here = set(available_steps(run_id, root / "checkpoints"))
        there = set(DL.steps_of(every, run_id)) if on_hub else set()
        no_adapter = [s for s in steps if s not in here | there]
        if no_adapter:
            raise FileNotFoundError(f"{run_id} has no saved adapter for steps {no_adapter}. Saved: on this machine "
                                    f"{sorted(here)}" + (f", on {hub_repo} {sorted(there)}" if on_hub else ""))
        files = {s: f"results/{run_id}/selections_decision_step-{s}.csv" for s in steps}
        unscored = [s for s, f in files.items() if f not in listed and not (root / f).exists()]
        if unscored:
            raise FileNotFoundError(f"{run_id} has no saved decisions for steps {unscored} (selections_decision_step-<N>"
                                    ".csv): the run did not score those steps, so there is no rule to score replies against")
        # From the Hub: the adapters it has (fetch_run leaves alone what is already here) with those steps' saved
        # decisions, and the saved decisions of a step whose adapter is only on this machine. Before 2026-11-24
        # nothing was fetched unless the Hub had every step, so a run with some steps there and some here was
        # refused only after the model had loaded.
        fetch = [s for s in steps if s in there] if on_hub else []
        loose = [f for s, f in files.items() if s not in fetch and f in listed and not (root / f).exists()]
        if on_hub and f"results/{run_id}/settings.json" in listed:
            with contextlib.redirect_stdout(io.StringIO()):
                DL.fetch_run(hub_repo, run_id, [], root=root, every=every)          # settings.json, a few kB
        run_dir = find_run(run_id, results_dir)
        saved = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        size = eval_batch_size if eval_batch_size is not None else card_batch_size(saved["model_hyperparameters"]["model_name"], vram)
        cfg = run_settings(run_dir, questions_from, only, show_neither, reply_prefix, flip_ab, size)
        with contextlib.redirect_stdout(io.StringIO()):                              # two lines about the data, per run
            data = D.load_data(cfg, C.resolve(cfg).rule)
        record, finished = held(run_id, name, root, hub_repo, every)
        if record is not None:
            _same_setup_or_stop(record, cfg, per_kind, f"results/{run_id}/ask_only/{name}")
        jobs.append({"run": run, "run_id": run_id, "out_dir": run_dir / "ask_only" / name, "fetch": fetch, "loose": loose,
                     "replies": reply_count(cfg, data, per_kind) * len(steps),
                     "eval_batch_size": C.eval_batch_size(cfg, "decision"), "sized_for_card": eval_batch_size is None and size is not None,
                     "done": bool(skip_done) and set(steps) <= finished,
                     "show_neither": bool(cfg["report_schema"].get("show_neither", False)),
                     "reply_prefix": E.reply_prefix(cfg), "flip_ab": E.flip_ab(cfg)})

    first = jobs[0]
    sizes = sorted({j["eval_batch_size"] for j in jobs})
    card = f"this card has {vram:.0f} GB" if vram is not None else "no GPU on this machine"
    why = "set in EVAL_BATCH_SIZE" if eval_batch_size is not None else (f"sized for the card, {card}" if first["sized_for_card"]
                                                                       else f"each run's own, {card}")
    print(f'ask_only plan, output folder results/<run_id>/ask_only/{name}/')
    print(f'  prefill {first["reply_prefix"]!r}' + ("" if first["reply_prefix"] else " (none)")
          + f' | C line {"shown" if first["show_neither"] else "not shown"}'
          + (f" | {per_kind} pairs of each kind ({3 * per_kind} pairs a question)" if per_kind else " | each run's own pairs")
          + (" | A and B swapped" if first["flip_ab"] else " | A and B as drawn")
          + f' | eval batch {", ".join(map(str, sizes))} ({why})')
    for j in jobs:
        print(f'  {j["run_id"]:<72} {len(steps)} steps {j["replies"]:>7} replies' + ("   already done, skipped" if j["done"] else ""))
    todo = [j for j in jobs if not j["done"]]
    print(f"{len(jobs)} run(s), {len(todo)} to ask: {sum(j['replies'] for j in todo)} replies")
    return jobs, every, root


def plan(runs, steps, questions_from=None, only=None, show_neither=None, reply_prefix=None, per_kind=None, name=None,
         flip_ab=None, eval_batch_size=None, hub_repo=None, skip_done=True, root=None):
    """The jobs ask_runs would do, printed and returned, with nothing asked: one job per run. Everything that can
    be wrong with the settings is found here, before a model is loaded: a run that is not on the Hub or here, a step
    with no adapter or no saved decisions, a question name the set does not have, a folder that already holds
    replies asked another way. Downloads each run's settings.json (a few kB) and nothing else."""
    return _plan(runs, steps, questions_from, only, show_neither, reply_prefix, per_kind, name, flip_ab, eval_batch_size,
                 hub_repo, skip_done, root)[0]


def ask_runs(runs, steps, questions_from=None, only=None, show_neither=None, reply_prefix=None, per_kind=None, name=None,
             flip_ab=None, eval_batch_size=None, hub_repo=None, skip_done=True, root=None):
    """Ask every run in `runs` (a run id or its timestamp, or a list of them) the same questions at the same steps,
    one run after another. This is what ask_only.ipynb's Ask cell calls.

    First the plan (plan(): printed, and every setting checked before a model is loaded). Then, for each run:
      - its adapters and saved decisions are downloaded from `hub_repo` when they are not here (checked against the Hub)
      - ask(): the base model is loaded, each step's adapter applied, the questions asked
      - the output folder is pushed to `hub_repo` at the same path (a push that fails prints NOT pushed and the
        work goes on; the notebook's last cell pushes again)
      - the model is dropped, so one run's adapters never stay on the card for the next
    A run whose folder already lists every step (here or on the Hub) is skipped when `skip_done`, so a stopped
    notebook can be started again. A run that fails does not stop the runs after it, except when the card ran
    out of memory, which every later run would do too.

    `eval_batch_size`: a number, or None for the size that fits this machine's card (card_batch_size).
    Returns one dict per run, in order: run_id, out_dir, status ("asked", "skipped: already done" or
    "FAILED: <why>"), minutes, replies.
    """
    from src import download as DL
    from src.persist import push_folder

    jobs, every, root = _plan(runs, steps, questions_from, only, show_neither, reply_prefix, per_kind, name, flip_ab,
                              eval_batch_size, hub_repo, skip_done, root)
    steps = list(steps)
    todo, spent, stop = [j for j in jobs if not j["done"]], [], None
    for j in jobs:
        j["minutes"] = None
        if j["done"]:
            j["status"] = "skipped: already done"
            continue
        if stop:
            j["status"] = f"not asked: {stop}"
            continue
        print(f"\n=== run {todo.index(j) + 1} of {len(todo)}: {j['run_id']}", flush=True)
        started, loaded, failure = time.time(), None, None
        try:
            if j["fetch"]:
                DL.fetch_run(hub_repo, j["run_id"], j["fetch"], results=("selections",), root=root, every=every)
            if j["loose"]:
                DL.fetch_files(hub_repo, j["loose"], root, label=f"{j['run_id']}: saved decisions")
            _, _, loaded = ask(j["run_id"], steps, questions_from, only, show_neither, reply_prefix, per_kind, name,
                               results_dir=root / "results", flip_ab=flip_ab, eval_batch_size=j["eval_batch_size"],
                               checkpoint_root=root / "checkpoints")
        except Exception as err:
            traceback.print_exc()
            failure = f"{type(err).__name__}: {err}"
            if "out of memory" in failure.lower():
                half = max(1, j["eval_batch_size"] // 2)
                stop = f"the card ran out of memory at eval batch {j['eval_batch_size']}: set EVAL_BATCH_SIZE = {half} and run again"
                failure = stop
            release(loaded, err)
        else:
            release(loaded)
        j["minutes"] = round((time.time() - started) / 60, 1)
        if failure:
            j["status"] = f"FAILED: {failure}"
            print(f"FAILED {j['run_id']}: {failure}", flush=True)
            continue
        j["status"] = "asked"
        push_folder(j["out_dir"], hub_repo, root=root, stop_on_error=False)
        spent.append(time.time() - started)
        left = len([k for k in todo if "status" not in k])
        print(f"{j['run_id']}: {j['minutes']} min" + (f"; {left} run(s) left, about {left * sum(spent) / len(spent) / 60:.0f} min "
                                                      "at this pace" if left else ""), flush=True)
    failed = [j for j in jobs if j["status"].startswith(("FAILED", "not asked"))]
    print(f"\n{sum(j['status'] == 'asked' for j in jobs)} asked, {sum(j['done'] for j in jobs)} skipped, {len(failed)} failed")
    for j in failed:
        print(f"  {j['run_id']}: {j['status']}")
    return [{k: j[k] for k in ("run_id", "out_dir", "status", "minutes", "replies")} for j in jobs]


def stop_if_failed(jobs):
    """Raise, naming the runs, when any job of ask_runs failed. The notebook calls it last, after everything that
    was asked has been saved, so a Run All ends on the failure and not before the save."""
    failed = [j for j in jobs if not j["status"].startswith(("asked", "skipped"))]
    if failed:
        raise RuntimeError(f"{len(failed)} run(s) were not asked: " + " | ".join(f"{j['run_id']}: {j['status']}" for j in failed))


# --- checks -------------------------------------------------------------------------------------------

def reproduction(run_dir, out_dir, step):
    """Per question, the share of (sample, reply) pairs equal to the run's own reports_raw at that step."""
    tag = f"{E.stage_label('decision')}_step-{step}"
    def read(p):
        return {(r["batch"], r["sample"]): r["reply"] for r in
                (json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip())}
    theirs, mine = read(Path(run_dir) / f"reports_raw_{tag}.jsonl"), read(Path(out_dir) / f"reports_raw_{tag}.jsonl")
    out = []
    for q in sorted({b for b, _ in theirs}):
        keys = [k for k in theirs if k[0] == q]
        both = [k for k in keys if k in mine]
        same = sum(theirs[k] == mine[k] for k in both)
        out.append({"question": q, "saved": len(keys), "asked": len(both), "identical": same,
                    "match_rate": same / len(both) if both else float("nan")})
    extra = sorted({b for b, _ in mine} - {b for b, _ in theirs})
    return pd.DataFrame(out), extra


def report_table(out_dir, question="main"):
    """One row per step of an ask-only folder: the numbers the report-back asks for.
      both_pass_cos4      cosine of the mean weights reply on both-pass trials with the recovered weights, over the
                          four unscreened weights (faithfulness_by_trial_kind, cos_recovered_4), and its n
      both_fail_all_-100  share of replies on both-fail trials with all five at -100 (reply_forms, every reply)
      one_fail_screened_100   share of replies on one-fails trials with the screened attribute at 100
      cut_probe...        where each cut probe turns from rejected to accepted (one column per probe)
      readable            readable / asked per question (step_answers)
      no-trial replies    word for word, one column per no-trial question"""
    out_dir = Path(out_dir)
    fk = pd.read_csv(out_dir / "faithfulness_by_trial_kind.csv") if (out_dir / "faithfulness_by_trial_kind.csv").exists() else pd.DataFrame()
    forms = pd.read_csv(out_dir / "reply_forms.csv") if (out_dir / "reply_forms.csv").stat().st_size > 1 else pd.DataFrame()
    sa = pd.read_csv(out_dir / "step_answers.csv")
    rows = []
    for step in sorted(sa["step"].unique()):
        row = {"step": int(step)}
        if len(fk):
            w = fk[(fk["step"] == step) & (fk["question"] == question) & (fk["kind"] == "weights")]
            if len(w):
                row["both_pass_cos4"], row["both_pass_n"] = w.iloc[0]["cos_recovered_4"], int(w.iloc[0]["n"])
        if len(forms):
            f = forms[(forms["step"] == step) & (forms["question"] == question)]
            for kind, col, name in (("neither", "share_all_-100", "both_fail_all_-100"),
                                    ("screen", "share_screened_100", "one_fail_screened_100")):
                part = f[f["kind"] == kind]
                if len(part):
                    row[name] = part.iloc[0][col]
                    row[f"{name}_n"] = int(part.iloc[0]["asked"])
        s = sa[sa["step"] == step]
        # every cut probe (cut_probe, cut_probe_caffeine_full, cut_probe_steeping, ...): where it turns
        for _, r in s[s["question"].astype(str).str.startswith("cut_probe")].iterrows():
            row[r["question"]] = r["stated_minimum"]
        for _, r in s.iterrows():
            if str(r["question"]).startswith("cut_probe"):
                continue
            row[f"readable {r['question']}"] = f"{int(r['readable'])}/{int(r['asked'])}"
            if "reply" in r and isinstance(r["reply"], str):
                row[f"reply {r['question']}"] = r["reply"]
        rows.append(row)
    return pd.DataFrame(rows)


def reply_count(cfg, data, per_kind=None):
    """How many replies one step asks (every persona): trial questions x trials, no-trial ones once, probes once each level."""
    comps = C.resolve(cfg)
    k0 = data.personas[0]
    n_trials = 3 * per_kind if per_kind else cfg["report_schema"]["samples_per_report"]
    once = cfg["report_schema"]["report_temperature"] == 0
    total = 0
    for s in comps.schemas:
        for b in s.batches_for(k0):
            total += n_trials if b.show_trial else (1 if once else n_trials)
    return total * cfg["model_training"]["instances"]
