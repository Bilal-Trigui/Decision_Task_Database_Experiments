"""Per-reply flags for the cut questions: is the named dimension right, how far is the stated cut from the
true one, and could the reply be a copy of a shown value or a range end?

    python tools/cutoff_flags.py results/<run_id> [results/<run_id> ...]

evaluate._reports calls update() after every checkpoint. Reads only the run's reports_raw_<tag>.jsonl files
and its data folder (persona_rule), so it works on finished runs too.

Which questions. Every question whose reply names a screened dimension and a cut, in units or in percent:
  units    cutoff_units, cutoff_units_no_trial            key cutoff_value
           screen_cut_units (+ _trial_first, _no_trial)   key minimum_value     (the old wording, for comparison)
  percent  cutoff_percent, cutoff_percent_no_trial        key cutoff_percent
           screen_cut_percent_no_trial                    key minimum_percent   (the old wording)
A question is in units or percent by its answer key, so a new question with one of these keys is flagged too.

Flags, one row per reply (cutoff_flags.csv):
  readable          the reply is a JSON object with the question's two keys and numbers for both
  named_dim, dim_right          the dimension number given, and whether it is the persona's screened one
  stated, true_cut, distance    the stated cut, the persona's (mg, or percent of range), stated - true
  exact, within_1, within_5     |distance| = 0, <= 1, <= 5 (units); for percent within_2 and within_10
  right             dim_right and within 1 (units) or within 2 points (percent)
  range_end         the stated number is an end of the named dimension's range (0 or 70 mg; 0 or 100 %)
  trial shown only (the option values are read back from the prompt saved with the reply):
  kind              weights (both options pass the cut), screen (one fails), neither (both fail); no_trial otherwise
  copies_named_dim  the stated number equals option A's or B's value on the named dimension
  copies_any_dim    ... on any dimension (for a percent reply, this catches an mg value written as a percent)
  cut_shown_near    an option's value on the screened dimension is within 2 (units) of the true cut, so a copy
                    could look right; set on every trial-shown question, in units of that dimension
  right_not_copy    right, and not a copy of an option value on the named dimension
Summary per step, question and kind (cutoff_flags_summary.csv): n asked, n readable, and the share of the
readable replies with each flag, plus the mean stated cut.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import persona_rule
from src.step_answers import _read, _num, _step_of, _fmt

UNITS_KEYS = ("cutoff_value", "minimum_value")
PERCENT_KEYS = ("cutoff_percent", "minimum_percent")
TOL = {"units": (1.0, 5.0), "percent": (2.0, 10.0)}     # (right / first band, second band)
NEAR_SHOWN = 2.0            # an option within 2 units of the true cut on the screened dimension
EPS = 1e-6
FLAGS = ["dim_right", "exact", "within_a", "within_b", "right", "range_end",
         "copies_named_dim", "copies_any_dim", "cut_shown_near", "right_not_copy"]


def _question_spec(name):
    """(keys, show_trial) for a question folder, or None when it is not a folder question."""
    folder = Path(__file__).resolve().parents[1] / "questions" / "paper_minimum_threshold" / name
    try:
        spec = json.loads((folder / "question.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return spec.get("keys") or [], spec.get("show_trial", True)


def scale_of(keys):
    """'units' or 'percent' for a cut question's answer keys, else None."""
    if not keys or "screened_dimension" not in keys or len(keys) != 2:
        return None
    other = [k for k in keys if k != "screened_dimension"][0]
    return "units" if other in UNITS_KEYS else ("percent" if other in PERCENT_KEYS else None)


_BLOCK = re.compile(r"(?:^|\n)([AB]):\n(.*?)(?=\n\n|\Z)", re.S)
_LINE = re.compile(r"^\s*([A-Za-z0-9_]+):\s*(-?[0-9]+(?:\.[0-9]+)?)")


def options_in(prompt, dims):
    """{"A": [values in dims order], "B": [...]} read back from a prompt's trial, or None when it shows none."""
    out = {}
    for label, body in _BLOCK.findall(prompt):
        vals = {}
        for line in body.split("\n"):
            m = _LINE.match(line)
            if m:
                vals[m.group(1)] = float(m.group(2))
        if all(d in vals for d in dims):
            out[label] = [vals[d] for d in dims]
    return out if set(out) == {"A", "B"} else None


def _kind(options, screen):
    if options is None:
        return "no_trial"
    j = screen.position - 1
    a, b = screen.passes(options["A"][j]), screen.passes(options["B"][j])
    return "weights" if a and b else ("neither" if not a and not b else "screen")


def flag(row, rule, keys, scale):
    """One reply's flags as a dict (see the module docstring)."""
    screen, dims = rule.screen, rule.dims
    options = options_in(row.get("question", ""), dims)
    out = {"question": row["batch"], "sample": row.get("sample"), "scale": scale, "kind": _kind(options, screen),
           "reply": row["reply"], "scorer_parsed": bool(row.get("parsed"))}
    true_cut = screen.value if scale == "units" else screen.percent
    out["true_cut"] = true_cut
    if options is not None:
        j = screen.position - 1
        out["A_screened"], out["B_screened"] = options["A"][j], options["B"][j]
        # whether a copy could look right: units of the screened dimension, the same test for a percent question
        out["cut_shown_near"] = any(abs(options[x][j] - screen.value) <= NEAR_SHOWN for x in "AB")
    d = _read(row["reply"])
    value_key = [k for k in keys if k != "screened_dimension"][0]
    named = _num(d.get("screened_dimension")) if d else None
    stated = _num(d.get(value_key)) if d else None
    out["readable"] = d is not None and named is not None and stated is not None
    if not out["readable"]:
        return out
    out["named_dim"], out["stated"] = named, stated
    out["dim_right"] = abs(named - screen.position) < EPS
    dist = stated - true_cut
    a, b = TOL[scale]
    out.update(distance=dist, exact=abs(dist) < EPS, within_a=abs(dist) <= a + EPS, within_b=abs(dist) <= b + EPS)
    out["right"] = out["dim_right"] and out["within_a"]
    valid = named.is_integer() and 1 <= named <= len(dims)
    if valid:
        k = int(named) - 1
        ends = (0.0, 100.0) if scale == "percent" else (
            (rule.lows[k], rule.highs[k]) if rule.lows is not None else ())
        out["range_end"] = any(abs(stated - e) < EPS for e in ends) if ends else None
    if options is not None:
        out["copies_named_dim"] = valid and any(abs(stated - options[x][int(named) - 1]) < EPS for x in "AB")
        out["copies_any_dim"] = any(abs(stated - v) < EPS for x in "AB" for v in options[x])
        out["right_not_copy"] = out["right"] and not out["copies_named_dim"]
    return out


def flags_for_rows(rows, rule):
    """Flag rows for every cut-question reply among one step's reports_raw rows."""
    out = []
    specs = {}
    for r in rows:
        if r.get("level") is not None:
            continue        # cut_probe
        name = r["batch"]
        if name not in specs:
            spec = _question_spec(name)
            specs[name] = (spec[0], scale_of(spec[0])) if spec else (None, None)
        keys, scale = specs[name]
        if scale is None:
            continue
        out.append(flag(r, rule, keys, scale))
    return out


def summarize(flags):
    """Per question and kind (and 'all'): asked, readable, each flag's share of the readable, mean stated."""
    df = pd.DataFrame(flags)
    rows = []
    if df.empty:
        return rows
    for (tag, q), part in df.groupby(["tag", "question"], sort=False):
        kinds = ["all"] + [k for k in ("weights", "screen", "neither", "no_trial") if k in set(part["kind"])]
        for kind in kinds:
            p = part if kind == "all" else part[part["kind"] == kind]
            ok = p[p["readable"] == True]   # noqa: E712 (a column of objects)
            row = {"tag": tag, "step": p["step"].iloc[0], "question": q, "scale": p["scale"].iloc[0], "kind": kind,
                   "asked": len(p), "readable": len(ok),
                   "mean_stated": float(ok["stated"].mean()) if len(ok) else None,
                   "true_cut": p["true_cut"].iloc[0]}
            for f in FLAGS:
                if f in ok and ok[f].notna().any():
                    row[f] = float(ok[f].dropna().astype(bool).mean())
            if "cut_shown_near" in p and p["cut_shown_near"].notna().any():
                row["cut_shown_near_asked"] = float(p["cut_shown_near"].dropna().astype(bool).mean())
            rows.append(row)
    return rows


def _rows_of(path):
    with Path(path).open(encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def update(run_dir):
    """Rebuild cutoff_flags.csv and cutoff_flags_summary.csv from every reports_raw file. Returns the summary."""
    run_dir = Path(run_dir)
    rule = persona_rule.first(run_dir)
    if rule is None or rule.screen is None:
        return pd.DataFrame()
    flags = []
    for f in sorted(run_dir.glob("reports_raw_*.jsonl"), key=lambda p: _step_of(p.stem[len("reports_raw_"):])):
        tag = f.stem[len("reports_raw_"):]
        step = _step_of(tag)[1]
        flags += [{"tag": tag, "step": step, **x} for x in flags_for_rows(_rows_of(f), rule)]
    if not flags:
        return pd.DataFrame()
    pd.DataFrame(flags).to_csv(run_dir / "cutoff_flags.csv", index=False, lineterminator="\n")
    summary = pd.DataFrame(summarize(flags))
    summary.to_csv(run_dir / "cutoff_flags_summary.csv", index=False, lineterminator="\n")
    return summary


SHORT = {"dim_right": "dim", "exact": "exact", "within_a": "w1|2", "within_b": "w5|10", "right": "right",
         "range_end": "end", "copies_named_dim": "copy", "copies_any_dim": "copy_any",
         "right_not_copy": "right&!copy"}


def print_step(run_dir, tag):
    summary = update(run_dir)
    if summary.empty:
        return
    s = summary[summary["tag"] == tag]
    if s.empty:
        return
    print(f"  cut-question flags at {tag} (share of readable replies; w1|2 = within 1 mg or 2 points):")
    def num(x):     # NaN (no readable reply) or None -> None, which _fmt shows as "-"
        return None if x is None or x != x else float(x)

    for _, r in s.iterrows():
        cells = [f"{SHORT[f]} {r[f]:.2f}" for f in SHORT if f in r and num(r[f]) is not None]
        print(f"    {r['question']:<30} {r['kind']:<8} readable {int(r['readable'])}/{int(r['asked'])}, "
              f"mean {_fmt(num(r['mean_stated']))} (true {_fmt(num(r['true_cut']))}): " + ", ".join(cells))
