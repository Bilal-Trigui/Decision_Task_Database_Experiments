"""Faithfulness split by the kind of trial each report was asked about.

    python tools/faithfulness_by_trial_kind.py results/<run_id> [results/<run_id> ...]

evaluate._reports calls run() after every checkpoint, so new runs write the file as they go.

Why. A weight question shown a trial is asked about 50 option pairs drawn as they come. Under a minimum
threshold those pairs are of three kinds, by the screened attribute against the persona's cut:
  weights   both options pass the screen: the weighted sum decides          (about 25% at a 50% cut)
  screen    one passes, one fails: the screen decides                       (about 50%)
  neither   both fail: the answer is C                                      (about 25%)
The results' faithfulness row averages all 50 replies into one report and scores it against the recovered
weights, which the estimator fitted only on the estimation trials the weights decide. So the reported side
pools every kind while the recovered side is weights-only. This splits the reported side.

Reads the saved weight_reports_<tag>_<schema>_<batch>.csv (every readable reply, with the options it was
shown), recovered_<tag>.csv and the data folder's constraint_table.csv. Retrains and re-asks nothing.

For each checkpoint, weight question and kind (and "all", which reproduces the results row's mean report):
  n                 readable replies of that kind
  template_share    share of them with every weight equal (all -100, all +100, ...)
  cos_recovered     cosine of the kind's mean report with the recovered weights (the faithfulness measure)
  cos_recovered_centred   the same with each vector's mean removed (a constant reply scores 0)
  cos_hidden, cos_hidden_centred   against the persona's own weights (hidden_vs_reported)
  cos_constant_recovered   what an all -100 reply would score against the recovered weights: the floor
                    a reply with no attribute-specific content gets for free when the weights lean negative
  ..._4             the same five cosines over the four unscreened weights only: the screened attribute's slot
                    is left out of both vectors (its weight is 0 by construction, and a reply's value there is
                    about the screen, not the weights). cos_recovered_4 is the one to quote. Equal to the
                    five-value columns for a persona with no screen.
  share_all_-100, share_all_0, share_screened_100   reply forms among the kind's readable replies: every
                    weight -100, every weight 0, the screened attribute's weight 100
Writes results/<run_id>/faithfulness_by_trial_kind.csv and prints the main question's rows.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import persona_rule

KINDS = ("weights", "screen", "neither")


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def centred(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return cosine(a - a.mean(), b - b.mean())


def _shows_trial(batch):
    """Whether the question behind a weight_reports file (batch, or '<batch>-<block>') was shown a trial."""
    folder = Path(__file__).resolve().parents[1] / "questions" / "paper_minimum_threshold" / batch.split("-")[0]
    try:
        return json.loads((folder / "question.json").read_text()).get("show_trial", True)
    except FileNotFoundError:
        return True     # a schema without question folders (linear): its one question shows a trial


def kind_of(row, screen):
    """The trial's kind from the screened attribute's two values: weights (both pass), screen (one), neither."""
    if screen is None:
        return "weights"
    j = screen.position
    a, b = screen.passes(row[f"A_attribute_{j}"]), screen.passes(row[f"B_attribute_{j}"])
    return "weights" if a and b else ("neither" if not a and not b else "screen")


COLUMNS = ["tag", "step", "scenario", "question", "kind", "n", "template_share", "cos_recovered",
           "cos_recovered_centred", "cos_hidden", "cos_hidden_centred", "cos_constant_recovered",
           "cos_recovered_4", "cos_recovered_centred_4", "cos_hidden_4", "cos_hidden_centred_4",
           "cos_constant_recovered_4", "share_all_-100", "share_all_0", "share_screened_100",
           "mean_report", "recovered"]


def unscreened(rule):
    """Indices of the attributes the screen does not use (all of them for a persona with no screen)."""
    n = len(rule.dims)
    return [i for i in range(n) if rule.screen is None or i != rule.screen.position - 1]


def reply_forms(reports, rule):
    """Shares of readable weight replies that are all -100, all 0, or put 100 on the screened attribute."""
    reports = np.asarray(reports, float)
    out = {"share_all_-100": float(np.mean(np.all(reports == -100, axis=1))),
           "share_all_0": float(np.mean(np.all(reports == 0, axis=1)))}
    out["share_screened_100"] = (float(np.mean(reports[:, rule.screen.position - 1] == 100))
                                 if rule.screen is not None else float("nan"))
    return out


def run(run_dir):
    """Every checkpoint's split, per persona, written to faithfulness_by_trial_kind.csv. Returns (rows, rules)."""
    run_dir = Path(run_dir)
    by_persona = persona_rule.rules(run_dir)
    rows = []
    for f in sorted(run_dir.glob("weight_reports_*.csv")):
        m = re.match(r"weight_reports_(.+?)_paper_(?:minimum|maximum)_threshold_(.+)\.csv$", f.name) or \
            re.match(r"weight_reports_(.+?)_(?:linear)_(.+)\.csv$", f.name)
        if not m:
            continue
        tag, batch = m.groups()
        if not _shows_trial(batch):
            continue        # asked with no trial: its option columns are not what it was shown
        df = pd.read_csv(f)
        rec_file = run_dir / f"recovered_{tag}.csv"
        if df.empty or "A_attribute_1" not in df or "scenario" not in df or not rec_file.exists():
            continue
        rec_table = pd.read_csv(rec_file).set_index("scenario")
        step = int(re.search(r"step-(\d+)", tag).group(1)) if re.search(r"step-(\d+)", tag) else -1
        for scenario, reports_df in df.groupby("scenario", sort=False):
            rule = by_persona.get(scenario)
            if rule is None or scenario not in rec_table.index:
                continue
            n = len(rule.dims)
            cols = [f"report_attr{i}" for i in range(1, n + 1)]
            if not all(c in reports_df for c in cols) or reports_df[[f"A_attribute_{i}" for i in range(1, n + 1)]].isna().all().all():
                continue    # not a weight question, or its options were not saved
            recovered = rec_table.loc[scenario, [f"b_attr{i}" for i in range(1, n + 1)]].to_numpy(float)
            kinds = [kind_of(r, rule.screen) for _, r in reports_df.iterrows()]
            for kind in ("all",) + KINDS:
                part = reports_df if kind == "all" else reports_df[[k == kind for k in kinds]]
                base = {"tag": tag, "step": step, "scenario": scenario, "question": batch, "kind": kind, "n": len(part)}
                if part.empty:
                    rows.append(base)
                    continue
                reports = part[cols].to_numpy(float)
                mean = reports.mean(axis=0)
                keep = unscreened(rule)
                m4, r4, h4 = mean[keep], recovered[keep], rule.weights[keep]
                rows.append({**base, **reply_forms(reports, rule),
                    "cos_recovered_4": cosine(m4, r4), "cos_recovered_centred_4": centred(m4, r4),
                    "cos_hidden_4": cosine(m4, h4), "cos_hidden_centred_4": centred(m4, h4),
                    "cos_constant_recovered_4": cosine(-np.ones(len(keep)), r4),
                    "template_share": float(np.mean([np.ptp(r) == 0 for r in reports])),
                    "cos_recovered": cosine(mean, recovered), "cos_recovered_centred": centred(mean, recovered),
                    "cos_hidden": cosine(mean, rule.weights), "cos_hidden_centred": centred(mean, rule.weights),
                    "cos_constant_recovered": cosine(-np.ones(n), recovered),
                    "mean_report": " / ".join(f"{v:.0f}" for v in mean),
                    "recovered": " / ".join(f"{v:.0f}" for v in recovered)})
    out = pd.DataFrame(rows, columns=COLUMNS)
    if len(out):
        out = out.sort_values(["question", "scenario", "step", "kind"])
    out.to_csv(run_dir / "faithfulness_by_trial_kind.csv", index=False, lineterminator="\n")
    return out, by_persona



def print_step(run_dir, tag):
    """One line per weight question shown a trial at this step: n and cosine over the four unscreened
    weights with the recovered ones, per kind."""
    out, _ = run(run_dir)
    out = out[out["tag"] == tag]
    if out.empty:
        return
    print(f"  faithfulness by report-trial kind at {tag} (cosine with recovered over the unscreened weights; n replies):")
    for (q, scenario), part in out.groupby(["question", "scenario"], sort=False):
        cells = []
        for kind in ("all",) + KINDS:
            r = part[part["kind"] == kind]
            if r.empty or not r.iloc[0]["n"]:
                cells.append(f"{kind} -")
            else:
                c = r.iloc[0].get("cos_recovered_4")
                cells.append(f"{kind} {c:+.2f} (n {int(r.iloc[0]['n'])})" if c == c else f"{kind} nan (n {int(r.iloc[0]['n'])})")
        label = q if out["scenario"].nunique() == 1 else f"{q} [{scenario}]"
        print(f"    {label:<34} " + ", ".join(cells))

