"""Cross-persona control for a finished run: score each report against OTHER personas' true weights.

    python tools/cross_persona_control.py results/<run_id> [--batch main] [--block main]
    python tools/cross_persona_control.py results/<persona B run> --against results/<persona A run>

Asks whether a run's faithfulness (e.g. 0.65 at step 1500 of the one-persona run) is specific to the
persona the model was trained on. Retrains nothing; reads settings.json, the data folder and the saved
weight_reports_*.csv. For every checkpoint, question and block it scores the persona's mean report:

  own            against the persona's own hidden weights (reproduces hidden_vs_reported)
  pool           against every other persona's hidden weights in instilled_weights.csv, attribute
                 slot by slot (the empirical null: mean, 95th percentile, p = (1 + #>=own)/(1 + N))
  pool_chance    the same pool fed through src/chance.py's chance_level as its sampler, so the null
                 is computed by the harness exactly as `chance` is (only the latent source differs)
  template       a constant all-equal reply (every weight -100, or +100, whichever scores higher) against
                 the own weights: what a reply carrying no attribute-specific content earns here
  step0          the untrained model's mean report against the own weights: the attribute-name prior
  vs_step0       this checkpoint's report against the step-0 report: how far training moved it
  no_template    own score after dropping all-equal replies before averaging
  template_share share of readable replies that are all-equal
  against        (with --against) the same report scored against the OTHER run's persona's hidden weights,
                 with against_centred, against_no_template(_centred) and own_minus_against beside it. Two runs
                 that train the same persona and prompts on different weights give the two-way test: a report
                 carries training signal when it scores higher against its own weights than the other run's.

A persona-specific report beats the pool. Beating the pool does NOT rule out priors on its own: a prior
or template that happens to fit this persona also beats it. template, step0 and the centred scores say
whether that is what happened. Writes results/<run_id>/cross_persona_control.csv.
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
from src.chance import chance_level  # noqa: E402
from src.config import apply_defaults, resolve, validate  # noqa: E402


class PoolRule:
    """A stand-in rule for chance_level: same block slices, latents drawn from real personas instead."""

    def __init__(self, rule, pool):
        self.rule, self.pool = rule, pool

    def block_slices(self, n):
        return self.rule.block_slices(n)

    def sample_latent(self, n, rng):
        return self.pool[rng.randrange(len(self.pool))]


def _step(tag):
    return 0 if tag.startswith("base_model") else int(re.findall(r"(\d+)", tag)[0])


def _score(estimator, block, a, b):
    return (estimator.block_distance(block, a, b), estimator.block_distance_centred(block, a, b),
            estimator.block_rank_correlation(block, a, b))


def other_weights(other_run, rule, n):
    """scenario -> hidden latent (rule's latent column order) of another run's personas, from its data folder."""
    cfg = json.loads((Path(other_run) / "settings.json").read_text())
    table = pd.read_csv(Path(cfg["model_training"]["data_dir"]) / "instilled_weights.csv").set_index("scenario")
    instances = cfg["model_training"].get("instances") or len(table)
    return {s: table.loc[s, rule.latent_columns(n)].to_numpy(float) for s in table.index[:instances]}


def control(run_dir, only_batch=None, only_block=None, chance_draws=2000, against=None):
    run_dir = Path(run_dir)
    cfg = validate(apply_defaults(json.loads((run_dir / "settings.json").read_text())))
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    rule, estimator = comps.rule, comps.estimator
    n = data.attribute_count
    slices = rule.block_slices(n)
    own_names = {p.short_name for p in data.personas}
    index = {p.short_name: k for k, p in enumerate(data.personas)}

    table = pd.read_csv(Path(cfg["model_training"]["data_dir"]) / "instilled_weights.csv").set_index("scenario")
    pool_full = table.loc[[s for s in table.index if s not in own_names], rule.latent_columns(n)].to_numpy(float)
    seed = cfg["model_hyperparameters"]["seed"]
    other = other_weights(against, rule, n) if against else {}

    rows, step0 = [], {}
    for schema in comps.schemas:
        for batch in schema.batches_for(data.personas[0]):
            if only_batch and batch.name != only_batch:
                continue
            for block in batch.blocks:
                if (only_block and block != only_block) or block not in slices:
                    continue
                key = batch.name if len(batch.blocks) == 1 else f"{batch.name}-{block}"
                suffix = f"_{schema.name}_{key}.csv"
                paths = [p for p in run_dir.glob(f"weight_reports_*{suffix}")
                         if p.name[len("weight_reports_"):].startswith(("decision_step-", "base_model_step-"))]
                for path in sorted(paths, key=lambda p: _step(p.name[len("weight_reports_"):])):
                    tag = path.name[len("weight_reports_"):-len(suffix)]
                    frame = pd.read_csv(path)
                    cols = [c for c in frame.columns if c.startswith("report_")]
                    for scenario, part in frame.groupby("scenario"):
                        if scenario not in index:
                            continue
                        replies = part[cols].to_numpy(float)
                        rep = schema.aggregate(block, list(replies))
                        own = np.asarray(data.latents[index[scenario]], float)[slices[block]]
                        pool = pool_full[:, slices[block]]
                        flat = np.array([np.allclose(r, r[0]) for r in replies])
                        real = _score(estimator, block, rep, own)
                        nulls = np.array([_score(estimator, block, rep, w) for w in pool], float)
                        cos_null = nulls[:, 0][~np.isnan(nulls[:, 0])]
                        harness = chance_level(PoolRule(rule, pool_full), block, {0: own}, n, chance_draws, seed, estimator)
                        template = max(estimator.block_distance(block, np.full_like(own, s), own) for s in (-100.0, 100.0))
                        if _step(tag) == 0:
                            step0[(batch.name, block, scenario)] = rep
                        prior = step0.get((batch.name, block, scenario))
                        kept = replies[~flat]
                        rows.append({
                            "checkpoint_step": _step(tag), "report_batch": batch.name, "block": block, "scenario": scenario,
                            "n_replies": len(replies), "template_share": float(flat.mean()),
                            "own": real[0], "own_centred": real[1], "own_spearman": real[2],
                            "pool_mean": float(np.mean(cos_null)), "pool_p95": float(np.percentile(cos_null, 95)),
                            "pool_p": float((1 + np.sum(cos_null >= real[0] - 1e-9)) / (1 + cos_null.size)),
                            "pool_centred_mean": float(np.nanmean(nulls[:, 1])),
                            "pool_centred_p": float((1 + np.sum(nulls[:, 1] >= real[1] - 1e-9)) / (1 + np.sum(~np.isnan(nulls[:, 1])))),
                            "pool_n": int(cos_null.size),
                            "pool_chance": harness["chance"], "pool_chance_centred": harness["chance_centred"],
                            "template": template,
                            "step0": estimator.block_distance(block, prior, own) if prior is not None else np.nan,
                            "vs_step0": estimator.block_distance(block, rep, prior) if prior is not None else np.nan,
                            "no_template": estimator.block_distance(block, kept.mean(axis=0), own) if len(kept) else np.nan,
                            "no_template_centred": estimator.block_distance_centred(block, kept.mean(axis=0), own) if len(kept) else np.nan,
                        })
                        if against and scenario in other:
                            theirs = other[scenario][slices[block]]
                            row = rows[-1]
                            row["against"] = estimator.block_distance(block, rep, theirs)
                            row["against_centred"] = estimator.block_distance_centred(block, rep, theirs)
                            row["against_no_template"] = estimator.block_distance(block, kept.mean(axis=0), theirs) if len(kept) else np.nan
                            row["against_no_template_centred"] = (estimator.block_distance_centred(block, kept.mean(axis=0), theirs)
                                                                  if len(kept) else np.nan)
                            row["own_minus_against"] = row["own"] - row["against"]
                            row["own_minus_against_no_template"] = row["no_template"] - row["against_no_template"]
    out = pd.DataFrame(rows)
    name = f"cross_persona_control_vs_{Path(against).name}.csv" if against else "cross_persona_control.csv"
    out.to_csv(run_dir / name, index=False, lineterminator="\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--batch", default=None, help="one question only, e.g. main")
    parser.add_argument("--block", default=None, help="one block only, e.g. main")
    parser.add_argument("--chance-draws", type=int, default=2000)
    parser.add_argument("--against", default=None, help="results/<run> of the same persona trained on other weights")
    a = parser.parse_args()
    frame = control(a.run_dir, a.batch, a.block, a.chance_draws, a.against)
    show = ["checkpoint_step", "report_batch", "block", "template_share", "own", "own_centred", "pool_mean", "pool_p95",
            "pool_p", "pool_centred_p", "pool_chance", "template", "step0", "vs_step0", "no_template", "no_template_centred"]
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.float_format", "{:.3f}".format):
        print(frame[show].to_string(index=False))
        if a.against:
            print(f"\nagainst {a.against}")
            print(frame.reindex(columns=["checkpoint_step", "report_batch", "block", "own", "against", "own_minus_against",
                                         "no_template", "against_no_template", "own_minus_against_no_template",
                                         "own_centred", "against_centred"]).to_string(index=False))
    name = f"cross_persona_control_vs_{Path(a.against).name}.csv" if a.against else "cross_persona_control.csv"
    print(f"\nwritten to {Path(a.run_dir) / name}")


if __name__ == "__main__":
    main()
