"""The model's answers to the report questions at every evaluated step, side by side.

For each checkpoint (reports_raw_<tag>.jsonl) this collects:
  the four original questions, shown a trial (50 samples each):
    main, screen_index, screen_cut_units, main_report_units
    -> how many replies were readable, the mean reported weight per dimension, the screened
       dimension most often named and the mean minimum given, and the most common reply
  any copy of them asked with the trial first (<question>_trial_first), beside the original
  the five no-trial questions (asked once each):
    main_threshold_met, main_threshold_not_met, screen_index_no_trial,
    screen_cut_units_no_trial, screen_cut_percent_no_trial
    -> the reply exactly as given, and its values when it is readable JSON
  the cut curve (cut_probe): one yes/no answer per level, and where it crosses from rejected to accepted

Replies are read here with json.loads, more leniently than the scorer: a reply the scorer rejects
(parsed false in reports_raw) is still shown, and `scorer_parsed` says how many the scorer accepted.
Values are the model's own numbers (minimum_value in the units it gave, not converted).

Writes results/<run>/step_answers.md (one section per step, the persona's true rule at the top)
and results/<run>/step_answers.csv (one row per step and question). evaluate._reports calls
update() after every checkpoint, so both files grow as a run goes; for a finished run:

    python tools/step_answers.py results/<run_id>
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src import persona_rule

WITH_TRIAL = ["main", "screen_index", "screen_cut_units", "screen_cut", "main_report_units", "cutoff_units", "cutoff_percent"]
NO_TRIAL = ["main_no_trial", "main_threshold_met", "main_threshold_not_met", "screen_index_no_trial",
            "screen_cut_units_no_trial", "screen_cut_percent_no_trial", "cutoff_units_no_trial", "cutoff_percent_no_trial"]
WEIGHT_QUESTIONS = {"main", "main_no_trial", "main_report_units", "main_threshold_met", "main_threshold_not_met"}
SCREEN_KEYS = ["screened_dimension", "minimum_value", "minimum_percent", "minimum_level", "cutoff_value", "cutoff_percent"]
MINIMA = ("minimum_value", "minimum_percent", "minimum_level", "cutoff_value", "cutoff_percent")   # units, %, fraction, units, %
PERCENT = ("minimum_percent", "cutoff_percent")
TRIAL_FIRST = "_trial_first"    # a copy of a question asked with the trial before it (question.json trial_first)


def _base(question):
    return question[: -len(TRIAL_FIRST)] if question.endswith(TRIAL_FIRST) else question


def _weighted(question):
    return _base(question) in WEIGHT_QUESTIONS


def _shows_trial(question):
    return _base(question) in WITH_TRIAL


def _step_of(tag):
    """'decision_step-1500' -> (stage order, 1500, tag); unknown tags sort last."""
    m = re.search(r"step-(\d+)$", tag)
    step = int(m.group(1)) if m else 10**9
    stage = 0 if tag.startswith(("base_model", "decision")) else 1
    return stage, step, tag


def _read(reply):
    """The reply as a dict when it is a JSON object (a ```json fence is allowed), else None."""
    text = reply.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicates)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _no_duplicates(pairs):
    """json.loads keeps the last of a repeated key; a reply that repeats one is not an answer, so refuse it."""
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("repeated key")
    return dict(pairs)


def _num(x):
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x):
    if x is None:
        return "-"
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}".rstrip("0")


def _cell(text):
    return text.replace("\n", "\\n").replace("|", "\\|")


def truth(run_dir):
    """The persona's true rule from the run's data folder: (dims, weights, screen) or None."""
    rule = persona_rule.first(run_dir)
    if rule is None:
        return None
    weights = dict(zip(rule.dims, rule.weights.astype(float)))
    s = rule.screen
    screen = None if s is None else {"screened_dimension": s.position, "attribute": s.attribute,
                                     "minimum_value": s.value, "minimum_percent": s.percent,
                                     "units": s.units, "direction": s.direction}
    return rule.dims, weights, screen


def _answers(batch, d, dims):
    """Whether a JSON reply holds the keys its question asks for."""
    if _weighted(batch):
        return bool(dims) and all(k in d for k in dims)
    return any(k in d for k in SCREEN_KEYS)


def summarize_step(rows, dims=None):
    """One summary dict per question asked at this step (cut_probe rows aside: see probe_curve)."""
    by_batch = {}
    for r in rows:
        if r.get("level") is not None:
            continue
        by_batch.setdefault(r["batch"], []).append(r)
    if not dims:
        dims = []
        for r in by_batch.get("main", []):
            d = _read(r["reply"])
            if d:
                dims = [k for k in d if k not in SCREEN_KEYS]
                break
    order = WITH_TRIAL + [q + TRIAL_FIRST for q in WITH_TRIAL] + NO_TRIAL
    out = []
    for batch in order + [b for b in by_batch if b not in order]:
        if batch not in by_batch:
            continue
        rs = by_batch[batch]
        # readable: a JSON object holding what the question asks for (every dimension, for a weight question;
        # a screened dimension or a minimum, for a screen question). Laxer than the scorer, which also wants no
        # other keys and values in range, so a near miss is still shown; scorer_parsed says what was scored.
        ok = [d for d in (_read(r["reply"]) for r in rs) if d is not None and _answers(batch, d, dims)]
        s = {"question": batch, "asked": len(rs), "readable": len(ok),
             "scorer_parsed": sum(bool(r.get("parsed")) for r in rs)}
        if _weighted(batch):
            for dim in dims:
                vals = [v for v in (_num(d.get(dim)) for d in ok) if v is not None]
                s[dim] = sum(vals) / len(vals) if vals else None
        named = [v for v in (_num(d.get("screened_dimension")) for d in ok) if v is not None]
        if not named and not ok and len(rs) == 1 and _base(batch).startswith("screen"):
            bare = re.fullmatch(r"\s*(\d+)\s*", rs[0]["reply"])
            if bare:
                named = [float(bare.group(1))]
        if named:
            top, n = Counter(named).most_common(1)[0]
            s["screened_dimension"] = top
            s["screened_dimension_share"] = f"{n}/{len(named)}"
        for key in MINIMA:
            vals = [v for v in (_num(d.get(key)) for d in ok) if v is not None]
            if vals:
                s[key] = sum(vals) / len(vals)
        top, n = Counter(r["reply"].strip() for r in rs).most_common(1)[0]
        if len(rs) == 1:
            s["reply"] = top
        else:
            s["most_common_reply"], s["most_common_reply_count"] = top, n
        out.append(s)
    return out


def probe_curve(rows):
    """The cut_probe answers at one step, per probe question: its points [(level, reply, answer)] by level,
    answer "yes", "no" or None (unreadable), and the stated minimum the curve implies."""
    curves = {}
    for r in rows:
        if r.get("level") is None:
            continue
        name = r["batch"].rsplit("_", 1)[0]
        d = _read(r["reply"])
        value = None
        if d and len(d) == 1:
            v = next(iter(d.values()))
            if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
                value = v.strip().lower()
        curves.setdefault(name, []).append((float(r["level"]), r["reply"].strip(), value))
    return {name: {"points": sorted(points), "stated_minimum": _crossing(sorted(points))}
            for name, points in curves.items()}


def _crossing(points):
    """Where the curve turns from rejected (yes) to accepted (no), when every yes lies below every no."""
    answers = [(lv, a) for lv, _, a in points if a is not None]
    if not answers:
        return "unreadable"
    yes = [lv for lv, a in answers if a == "yes"]
    no = [lv for lv, a in answers if a == "no"]
    if not yes:
        return f"never rejected (from {_fmt(answers[0][0])})"
    if not no:
        return f"always rejected (to {_fmt(answers[-1][0])})"
    if max(yes) < min(no):
        return f"between {_fmt(max(yes))} and {_fmt(min(no))}"
    return f"not monotone ({len(yes)} yes, {len(no)} no)"


def _collect(run_dir):
    steps = []
    for f in Path(run_dir).glob("reports_raw_*.jsonl"):
        tag = f.stem[len("reports_raw_"):]
        with f.open(encoding="utf-8") as h:
            rows = [json.loads(line) for line in h if line.strip()]
        steps.append((_step_of(tag), tag, rows))
    return [(tag, rows) for _, tag, rows in sorted(steps, key=lambda x: x[0])]


def _md_step(tag, summary, dims, probe):
    L = [f"## {tag}\n"]
    trial = [s for s in summary if _shows_trial(s["question"])]
    if trial:
        L.append("**Questions shown a trial (mean over readable replies; `_trial_first` = the trial before the question)**\n")
        head = ["question", "readable", "scorer parsed"] + dims + ["screened dim", "minimum"]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for s in trial:
            cells = [f"`{s['question']}`", f"{s['readable']}/{s['asked']}", f"{s['scorer_parsed']}/{s['asked']}"]
            cells += [_fmt(s.get(d)) if _weighted(s["question"]) else "" for d in dims]
            sd = s.get("screened_dimension")
            cells.append(f"{_fmt(sd)} ({s['screened_dimension_share']})" if sd is not None else "")
            m = next((k for k in MINIMA if k in s), None)
            cells.append("" if m is None else (f"{_fmt(s[m])} %" if m in PERCENT else f"{_fmt(s[m])} (fraction)" if m == "minimum_level" else _fmt(s[m])))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        L.append("Most common reply per question:\n")
        for s in trial:
            top, count = s.get("most_common_reply", s.get("reply", "")), s.get("most_common_reply_count", 1)
            L.append(f"- `{s['question']}` ({count}/{s['asked']}): `{_cell(top)}`")
        L.append("")
    none = [s for s in summary if not _shows_trial(s["question"])]
    if none:
        L.append("**No-trial questions (asked once, reply verbatim)**\n")
        head = ["question", "reply", "scorer parsed"] + dims + ["screened dim", "minimum"]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for s in none:
            reply = s.get("reply", s.get("most_common_reply", ""))
            cells = [f"`{s['question']}`", f"`{_cell(reply)}`", "yes" if s["scorer_parsed"] else "no"]
            cells += [_fmt(s.get(d)) if _weighted(s["question"]) else "" for d in dims]
            cells.append(_fmt(s.get("screened_dimension")) if "screened_dimension" in s else "")
            m = next((k for k in MINIMA if k in s), None)
            cells.append("" if m is None else f"{_fmt(s[m])} %" if m in PERCENT else _fmt(s[m]))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    for name, curve in probe.items():
        pts = curve["points"]
        L.append(f"**Cut curve `{name}` (no trial; Y = would be rejected): stated minimum {curve['stated_minimum']}**\n")
        L.append("| level | " + " | ".join(_fmt(lv) for lv, _, _ in pts) + " |")
        L.append("|" + "---|" * (len(pts) + 1))
        L.append("| rejected | " + " | ".join({"yes": "Y", "no": "n"}.get(a, "?") for _, _, a in pts) + " |")
        odd = [(lv, r) for lv, r, a in pts if a is None]
        if odd:
            L.append("")
            L.append("Unreadable: " + "; ".join(f"{_fmt(lv)}: `{_cell(r[:60])}`" for lv, r in odd))
        L.append("")
    return "\n".join(L)


def update(run_dir):
    """Rebuild step_answers.md and step_answers.csv from every reports_raw file in run_dir."""
    run_dir = Path(run_dir)
    steps = _collect(run_dir)
    if not steps:
        return None
    t = truth(run_dir)
    dims = t[0] if t else []
    md = [f"# Report answers per step: {run_dir.name}\n"]
    if t:
        _, weights, screen = t
        md.append("**True rule (the persona's):**\n")
        cut_name = screen["direction"] if screen else "cut"
        md.append("| " + " | ".join(dims) + f" | screened dim | {cut_name} |")
        md.append("|" + "---|" * (len(dims) + 2))
        sc = f"{screen['screened_dimension']} ({screen['attribute']})" if screen else "none"
        units = f" {screen['units']}" if screen and screen["units"] else ""
        mn = (f"{_fmt(screen['minimum_value'])}{units} ({_fmt(screen['minimum_percent'])} % of range)"
              if screen else "")
        md.append("| " + " | ".join(_fmt(weights[d]) for d in dims) + f" | {sc} | {mn} |\n")
        if screen:
            md.append("`main_threshold_not_met` has no agreed right answer yet.\n")
    if not dims:
        # no rule could be read: take the dimension names from the first readable weight reply
        for _, raw in steps:
            dims = next((s for s in [summarize_step(raw)] if s), [])
            break
        dims = [k for k in (dims[0] if dims else {}) if k not in ("question", "asked", "readable", "scorer_parsed")
                and not k.startswith(("most_common", "reply", "screened", "minimum"))]
    rows = []
    for tag, raw in steps:
        summary = summarize_step(raw, dims)
        probe = probe_curve(raw)
        md.append(_md_step(tag, summary, dims, probe))
        step = _step_of(tag)[1]
        rows += [{"tag": tag, "step": step, **s} for s in summary]
        for name, curve in probe.items():
            rows.append({"tag": tag, "step": step, "question": name, "asked": len(curve["points"]),
                         "readable": sum(a is not None for _, _, a in curve["points"]),
                         "stated_minimum": curve["stated_minimum"],
                         "curve": " ".join(f"{_fmt(lv)}:{a or '?'}" for lv, _, a in curve["points"])})
    (run_dir / "step_answers.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    pd.DataFrame(rows).to_csv(run_dir / "step_answers.csv", index=False, lineterminator="\n")
    return run_dir / "step_answers.md"


def print_step(run_dir, tag):
    """A short console view of one step's answers, printed after its reports."""
    f = Path(run_dir) / f"reports_raw_{tag}.jsonl"
    if not f.exists():
        return
    t = truth(run_dir)
    dims = t[0] if t else []
    with f.open(encoding="utf-8") as h:
        raw = [json.loads(line) for line in h if line.strip()]
    print(f"  answers at {tag}:")
    for s in summarize_step(raw, dims):
        parts = []
        if _weighted(s["question"]) and dims:
            parts.append("weights " + "/".join(_fmt(s.get(d)) for d in dims))
        if "screened_dimension" in s:
            parts.append(f"dim {_fmt(s['screened_dimension'])}")
        for key in MINIMA:
            if key in s:
                parts.append(f"{key} {_fmt(s[key])}")
        if "reply" in s:
            parts.append("reply " + json.dumps(s["reply"]))
        print(f"    {s['question']:<34} scored {s['scorer_parsed']}/{s['asked']}, readable {s['readable']}/{s['asked']}  " + ", ".join(parts))
    for name, curve in probe_curve(raw).items():
        line = "".join({"yes": "Y", "no": "n"}.get(a, "?") for _, _, a in curve["points"])
        lo, hi = curve["points"][0][0], curve["points"][-1][0]
        print(f"    {name:<34} {_fmt(lo)}..{_fmt(hi)} {line}  stated minimum {curve['stated_minimum']}")
