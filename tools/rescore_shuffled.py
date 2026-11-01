"""The shuffled-persona baseline for a run already finished, from its saved files. Retrains nothing.

    python tools/rescore_shuffled.py results/<run_id> [--draws 1000]

Runs scored before the baseline existed have faithfulness and hidden_vs_reported but nothing to read
them against except chance. This reads the run's settings.json, its data folder, its readable reports
(weight_reports_*.csv) and its recovered rules (recovered_*.csv), and scores, for every checkpoint,
question and block, the real comparison and the same reports given to the wrong persona, exactly as
evaluate.py now does at run time. Introspection folds are pooled into "both", as results.csv pools
them. Writes results/<run_id>/shuffled_baseline.csv and prints it; results.csv is left as it is.

An introspection branch's reports are scored against the rule recovered at the decision step it
branched from (recovered_decision_step-<N>.csv), as the pipeline scores them.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src.config import apply_defaults, resolve, validate  # noqa: E402
from src.evaluate import _compare, _with_shuffled  # noqa: E402

INTROSPECTION = re.compile(r"^introspection-from-(\d+)(?:-([a-z]+))?-fold(\d)_step-(\d+)$")


def _recovered(run_dir, tag, estimator, index):
    path = run_dir / f"recovered_{tag}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [f"b_{c}" for c in estimator.latent_columns()]
    return {index[s]: row[columns].to_numpy(float) for s, row in frame.set_index("scenario").iterrows() if s in index}


def _reports(path, schema, block, index):
    frame = pd.read_csv(path)
    columns = [c for c in frame.columns if c.startswith("report_")]
    out = {}
    for scenario, rows in frame.groupby("scenario"):
        if scenario in index:
            out[index[scenario]] = schema.aggregate(block, [r for r in rows[columns].to_numpy(float)])
    return out


def rescore(run_dir, draws=1000):
    run_dir = Path(run_dir)
    cfg = validate(apply_defaults(json.loads((run_dir / "settings.json").read_text())))
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    rule, estimator = comps.rule, comps.estimator
    n = data.attribute_count
    rule_slices, est_slices = rule.block_slices(n), estimator.block_slices()
    index = {p.short_name: k for k, p in enumerate(data.personas)}
    me, seed = {"shuffle_draws": draws}, cfg["model_hyperparameters"]["seed"]

    # (checkpoint label, schema, question, block) -> {persona: report}, folds pooled
    pooled, recovered_for = {}, {}
    for schema in comps.schemas:
        for batch in schema.batches_for(data.personas[0]):
            for block in batch.blocks:
                key = batch.name if len(batch.blocks) == 1 else f"{batch.name}-{block}"
                suffix = f"_{schema.name}_{key}.csv"
                for path in sorted(run_dir.glob(f"weight_reports_*{suffix}")):
                    tag = path.name[len("weight_reports_"):-len(suffix)]
                    match = INTROSPECTION.match(tag)
                    if match:
                        start, target, _, step = match.groups()
                        label = f"introspection-from-{start}{'' if not target else '-' + target}-both_step-{step}"
                        source = f"decision_step-{start}"
                    elif tag.startswith(("decision_step-", "base_model_step-")):
                        label, source = tag, tag
                    else:
                        print(f"skipping {path.name}: no decision step to score it against")
                        continue
                    pooled.setdefault((label, schema.name, batch.name, block), {}).update(
                        _reports(path, schema, block, index))
                    recovered_for[label] = source

    rows = []
    for (label, schema_name, question, block), rep in pooled.items():
        fits = _recovered(run_dir, recovered_for[label], estimator, index)
        row = {"checkpoint": label, "report_schema": schema_name, "report_batch": question, "block": block,
               "n_personas_reported": len(rep)}
        if fits and block in est_slices:
            rec = {k: v[est_slices[block]] for k, v in fits.items()}
            keyed = [(k, rep[k], rec[k]) for k in rep if k in rec]
            columns, kept, _ = _compare(estimator, block, keyed, "faithfulness")
            row.update(columns)
            row["n_personas_scored"] = len(kept)
            scored = [(k, a, b) for k, a, b in keyed if not np.isnan(estimator.block_distance(block, a, b))]
            row.update(_with_shuffled(estimator, block, scored, columns["faithfulness"], "faithfulness", me, seed,
                                      "faithfulness_gap", "faithfulness_shuffled_p"))
        if block in rule_slices:
            hid = {k: np.asarray(data.latents[k])[rule_slices[block]] for k in rep}
            keyed = [(k, hid[k], rep[k]) for k in rep]
            columns, _, _ = _compare(estimator, block, keyed, "hidden_vs_reported")
            row.update(columns)
            scored = [(k, a, b) for k, a, b in keyed if not np.isnan(estimator.block_distance(block, a, b))]
            row.update(_with_shuffled(estimator, block, scored, columns["hidden_vs_reported"], "hidden_vs_reported", me,
                                      seed, "hidden_vs_reported_gap", "hidden_vs_reported_shuffled_p"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    order = {"base_model_step-0": 0}
    frame["_order"] = [order.get(c, int(re.findall(r"(\d+)", c)[0]) + (0.5 if c.startswith("intro") else 0))
                       for c in frame["checkpoint"]]
    frame = frame.sort_values(["_order", "report_batch", "block"]).drop(columns="_order")
    frame.to_csv(run_dir / "shuffled_baseline.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="results/<run_id>")
    parser.add_argument("--draws", type=int, default=1000, help="reshufflings per comparison (default 1000)")
    args = parser.parse_args()
    frame = rescore(args.run_dir, args.draws)
    show = ["checkpoint", "report_batch", "block", "faithfulness", "shuffled_faithfulness", "faithfulness_shuffled_p",
            "hidden_vs_reported", "shuffled_hidden_vs_reported", "hidden_vs_reported_shuffled_p"]
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        print(frame.reindex(columns=show).to_string(index=False))
    print(f"\nwritten to {Path(args.run_dir) / 'shuffled_baseline.csv'}")


if __name__ == "__main__":
    main()
