"""Ask a finished run's saved adapters new report questions. No training, no decision trials.

Driven by ask_only.ipynb. Load the base model once; for each step, apply that step's adapter and ask
the questions. Everything but report_schema (model, data folder, rule, persona, seeds, batch sizes) is the
run's own saved settings.json. The recovered rule a step is scored against is refitted from the run's
saved selections_decision_step-<N>.csv, as evaluate.decision_detail_from_files does: the 1,000 decision
trials are not asked again.

Writes only into results/<run_id>/ask_only/<label>/:
  settings.json                     the settings the questions were asked under (data_dir made absolute)
  reports_raw_decision_step-<N>.jsonl   every reply, with run_id, step, question_name, trial_index,
                                    trial_kind, show_neither, reply_prefix, prompt and the options' values
  weight_reports_..., recovered_decision_step-<N>.csv   as a training run writes them
  step_answers.md / .csv, faithfulness_by_trial_kind.csv, cutoff_flags.csv / _summary.csv
  reply_forms.csv                   weight replies per step, question and trial kind, by form
  timing.csv                        seconds per step
Never trains, never appends to results/results.csv, never changes a file of the run itself.

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
import copy
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
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

def run_settings(run_dir, questions_from=None, only=None, show_neither=None, reply_prefix=None):
    """The run's own settings with report_schema taken from `questions_from` (a config path; None keeps the
    run's own), cut down to `only` (question names), and show_neither / reply_prefix overridden when given."""
    cfg = json.loads((Path(run_dir) / "settings.json").read_text(encoding="utf-8"))
    cfg.pop("_source", None)
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


def balanced_trials(scenario, screen, per_kind, seed=BALANCED_SEED, max_draws=1_000_000):
    """[(kind, trial, draw index)]: the first `per_kind` trials of each kind, in the order drawn."""
    if screen is None:
        raise ValueError("balanced trials need a screen: this persona has none (a linear run)")
    rng = random.Random(seed)
    kept, out = {k: 0 for k in KINDS}, []
    for i in range(max_draws):
        t = D.Trial(scenario, rng)
        kind = trial_kind(t, screen)
        if kept[kind] < per_kind:
            kept[kind] += 1
            out.append((kind, t, i))
            if all(v == per_kind for v in kept.values()):
                return out
    raise RuntimeError(f"{max_draws} draws gave only {kept} of {per_kind} per kind")


def default_trials(cfg, data, screens):
    """The run's own report trials (samples_per_report from seeds.reports, exactly as evaluate._reports draws
    them), each with its kind under that persona's screen and its index."""
    trials = D.fresh_trials(data.personas, cfg["report_schema"]["samples_per_report"], cfg["model_training"]["seeds"]["reports"])
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
    rules = persona_rule.rules(out_dir)     # out_dir/settings.json (prepare) names the data folder absolutely
    screens = [rules[data.personas[k].short_name].screen if data.personas[k].short_name in rules else None
               for k in personas]
    if per_kind:
        trials_by_persona = [balanced_trials(data.personas[k], screens[k], per_kind) for k in personas]
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
            results_dir=None):
    """(run_dir, out_dir, cfg) for an ask-only pass, with the output folder created and its settings.json written."""
    run_dir = find_run(run, results_dir)
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"--name must be a plain folder name, got {name!r}")
    cfg = run_settings(run_dir, questions_from, only, show_neither, reply_prefix)
    cfg["model_training"]["data_dir"] = str((ROOT / cfg["model_training"]["data_dir"]).resolve()
                                            if not Path(cfg["model_training"]["data_dir"]).is_absolute()
                                            else Path(cfg["model_training"]["data_dir"]))
    out_dir = run_dir / "ask_only" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**cfg, "_ask_only": {"run_id": run_dir.name, "questions_from": str(questions_from) if questions_from else None,
                                   "only": list(only) if only else None, "per_kind": per_kind}}
    (out_dir / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return run_dir, out_dir, cfg


def ask(run, steps, questions_from=None, only=None, show_neither=None, reply_prefix=None, per_kind=None, name=None,
        results_dir=None, loaded=None):
    """Ask the questions at every step. `loaded` is (model, tok, comps, device, data) to reuse a loaded base
    model; it is returned so the next call can reuse it too. Returns (out_dir, timing, loaded)."""
    run_dir, out_dir, cfg = prepare(run, questions_from, only, show_neither, reply_prefix, per_kind, name, results_dir)
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
        model = apply_adapter(loaded, run_dir.name, step)
        seconds, _ = ask_step(model, loaded["tok"], cfg, comps, data, loaded["device"], run_dir, step, out_dir, per_kind)
        timing.append({"step": step, "seconds": round(seconds, 1)})
        print(f"step {step}: {seconds:.0f}s")
    summaries(out_dir, cfg, _all_rows(out_dir))
    pd.DataFrame(timing).to_csv(out_dir / "timing.csv", index=False, lineterminator="\n")
    return out_dir, timing, loaded


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
      cut_probe           where the cut probe turns from rejected to accepted
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
        probe = s[s["question"] == "cut_probe"]
        if len(probe):
            row["cut_probe"] = probe.iloc[0]["stated_minimum"]
        for _, r in s.iterrows():
            if r["question"] == "cut_probe":
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
