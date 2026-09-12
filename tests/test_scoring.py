"""The scoring path, tested without a model.

Every faithfulness number is a distance between a report vector and a recovered
vector matched by persona and by attribute. A mismatch in either would corrupt
every result while the pipeline ran cleanly, so these checks feed `summarize`
fabricated reports whose correct score is known exactly.

  1. A reporter that returns the recovered latent scores faithfulness 1.0.
  2. A reporter that returns the hidden latent scores faithfulness equal to
     recovered-versus-hidden, and hidden-versus-reported 1.0.
  3. A reporter that returns the negated recovered latent scores -1.0, which
     catches a sign convention slipping anywhere in the chain.
  4. The two introspection folds are disjoint and cover every persona.
  5. Stage-two training targets are the hidden weights, row for row.

Run with `python -m tests.test_scoring`. Reads object.attribute_count = 5.
"""
import copy

import numpy as np
import pandas as pd

from src import data as D
from src.config import apply_defaults, resolve, validate
from src.estimators.base import pooled_pearson
from src.configs.test import TEST
from src.evaluate import summarize

N_PERSONAS = 12


def _setup():
    cfg = copy.deepcopy(TEST)
    cfg["model_training"]["instances"] = N_PERSONAS
    cfg["model_estimating"]["chance_draws"] = 5
    apply_defaults(cfg)
    validate(cfg)
    cfg["_source"] = "test"
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    rule, est = comps.rule, comps.estimator
    personas = data.personas
    trials = D.fresh_trials(personas, 40, cfg["model_training"]["seeds"]["verification"])
    hidden, recovered, rows = {}, {}, []
    for k in range(N_PERSONAS):
        for t in trials[k]:
            t.label = rule.label(data.latents[k], t)
        A = np.array([t.option_A.values for t in trials[k]])
        B = np.array([t.option_B.values for t in trials[k]])
        labels = [t.label for t in trials[k]]
        # a model that chooses exactly as the hidden latent would
        recovered[k] = est.fit(A, B, labels, personas[k].mins, personas[k].maxs)
        hidden[k] = data.latents[k]
        rows += [{"persona": k, "selection": l, "label": l} for l in labels]
    return cfg, comps, data, hidden, recovered, pd.DataFrame(rows)


def _detail(comps, data, hidden, recovered, decisions, reports):
    n = data.attribute_count
    schema = comps.schemas[0]
    return {
        "hidden": hidden,
        "recovered": recovered,
        # reported is keyed by batch, each carrying the block it reports on, because a schema
        # may ask about one block more than once and each asking is its own results row.
        "reported": {schema.name: {"main": ("main", reports)}},
        "parse": {schema.name: {"main": (len(reports), len(reports), "main")}},
        "decisions": decisions,
        "rule_slices": comps.rule.block_slices(n),
        "est_slices": comps.estimator.block_slices(),
    }


def _row(rows):
    assert len(rows) == 1, f"expected one results row for one schema and one block, got {len(rows)}"
    return rows[0]


def test_one_row_per_way_of_asking():
    """A3 asks about its interaction block three ways, and each asking gets its own row.

    Before the report batch was carried through, two batches naming one block overwrote each
    other in the reported dict and only the last survived, so a schema could collect a number
    the results file then threw away.
    """
    cfg = copy.deepcopy(TEST)
    cfg["model_training"].update({"data_dir": "data/a3/", "instances": 4})
    cfg["decision_rule"] = {"type": "interaction", "active_pairs": 1, "zero_pair_main_effects": True,
                            "main_scale": 0.5, "interaction_magnitude": [50, 100]}
    cfg["model_estimating"].update({"type": "interaction", "chance_draws": 3})
    cfg["report_schema"]["batches"] = ["main", "interaction", "pair_id", "pair_value"]
    apply_defaults(cfg)
    validate(cfg)
    cfg["_source"] = "test"
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    n = data.attribute_count
    rule, est, schema = comps.rule, comps.estimator, comps.schemas[0]
    personas = data.personas
    trials = D.fresh_trials(personas, 40, cfg["model_training"]["seeds"]["verification"])
    hidden, recovered, rows = {}, {}, []
    for k in range(4):
        for t in trials[k]:
            t.label = rule.label(data.latents[k], t)
        A = np.array([t.option_A.values for t in trials[k]])
        B = np.array([t.option_B.values for t in trials[k]])
        recovered[k] = est.fit(A, B, [t.label for t in trials[k]], personas[k].mins, personas[k].maxs)
        hidden[k] = data.latents[k]
        rows += [{"persona": k, "selection": t.label, "label": t.label} for t in trials[k]]
    slices = est.block_slices()
    reported, parse = {}, {}
    for batch in schema.batches(personas[0].names):
        per = {k: recovered[k][slices[batch.block]].copy() for k in recovered}
        reported[batch.name] = (batch.block, per)
        parse[batch.name] = (len(per), len(per), batch.block)
    detail = {
        "hidden": hidden, "recovered": recovered,
        "reported": {schema.name: reported}, "parse": {schema.name: parse},
        "decisions": pd.DataFrame(rows),
        "rule_slices": rule.block_slices(n), "est_slices": slices,
    }
    out = summarize([detail], cfg, comps, data, "t", "decision", 0, 1)
    by_batch = {r["report_batch"]: r for r in out}
    assert set(by_batch) == {"main", "interaction", "pair_id", "pair_value"}, sorted(by_batch)
    assert by_batch["pair_value"]["block"] == "interaction"
    assert by_batch["pair_id"]["block"] == "pair_id"
    for name in ("main", "interaction", "pair_value"):
        assert abs(by_batch[name]["faithfulness"] - 1.0) < 1e-9, (name, by_batch[name]["faithfulness"])
    assert abs(by_batch["pair_id"]["faithfulness"] - 1.0) < 1e-9, by_batch["pair_id"]["faithfulness"]
    print(f"four ways of asking gave four rows: {sorted(by_batch)}")


def test_perfect_reporter_scores_one():
    cfg, comps, data, hidden, recovered, decisions = _setup()
    reports = {k: v.copy() for k, v in recovered.items()}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    assert abs(row["faithfulness"] - 1.0) < 1e-9, row["faithfulness"]
    assert abs(row["faithfulness_pearson"] - 1.0) < 1e-9, row["faithfulness_pearson"]
    assert row["decision_accuracy"] == 1.0


def test_hidden_reporter_scores_the_recovery():
    cfg, comps, data, hidden, recovered, decisions = _setup()
    reports = {k: v.copy() for k, v in hidden.items()}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    assert abs(row["faithfulness"] - row["recovered_vs_hidden"]) < 1e-9, (row["faithfulness"], row["recovered_vs_hidden"])
    assert abs(row["hidden_vs_reported"] - 1.0) < 1e-9, row["hidden_vs_reported"]
    assert row["recovered_vs_hidden"] > 0.9, row["recovered_vs_hidden"]


def test_negated_reporter_scores_minus_one():
    cfg, comps, data, hidden, recovered, decisions = _setup()
    reports = {k: -v for k, v in recovered.items()}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    assert abs(row["faithfulness"] + 1.0) < 1e-9, row["faithfulness"]


def test_a_persona_that_cannot_be_scored_is_counted_not_hidden():
    """A persona with no direction drops out of the mean, and the row has to say so.

    The estimator returns zeros for a persona whose choices were all one letter, and cosine is
    undefined on a zero vector. Dropping it is right; dropping it quietly is not, because the mean
    then describes a population no column names. The correlation beside the mean has to cover the
    same personas, since a zero vector carries no NaN and would otherwise stay in it and drag it
    toward nothing.
    """
    cfg, comps, data, hidden, recovered, decisions = _setup()
    dead = 4
    for k in range(dead):
        recovered[k] = np.zeros_like(recovered[k])
    reports = {k: v.copy() for k, v in hidden.items()}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    assert row["n_personas_reported"] == N_PERSONAS, row["n_personas_reported"]
    assert row["n_personas_scored"] == N_PERSONAS - dead, row["n_personas_scored"]
    scorable = [(reports[k], recovered[k]) for k in range(dead, N_PERSONAS)]
    assert abs(row["faithfulness_pearson"] - pooled_pearson(scorable)) < 1e-9, (
        row["faithfulness_pearson"], pooled_pearson(scorable))
    print(f"  {dead} unscorable personas: mean over {row['n_personas_scored']}, "
          f"reported {row['n_personas_reported']}, correlation over the same {row['n_personas_scored']}")


def test_an_all_zero_report_does_not_flatter_the_mean():
    """A model answering all zeros parses cleanly, so only the scored count reveals it.

    This is the shape an undertrained model collapses to on A3's pair report, whose target is
    mostly zeros. Without the count, two personas answering well among twenty looks the same as
    twenty answering well.
    """
    cfg, comps, data, hidden, recovered, decisions = _setup()
    committed = 2
    reports = {k: (recovered[k].copy() if k < committed else np.zeros_like(recovered[k]))
               for k in range(N_PERSONAS)}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    assert row["parse_rate"] == 1.0, row["parse_rate"]
    assert row["n_personas_reported"] == N_PERSONAS, row["n_personas_reported"]
    assert row["n_personas_scored"] == committed, row["n_personas_scored"]
    assert abs(row["faithfulness"] - 1.0) < 1e-9, row["faithfulness"]
    print(f"  {committed} of {N_PERSONAS} personas gave a real answer: faithfulness "
          f"{row['faithfulness']:.3f} over {row['n_personas_scored']}, parse rate still "
          f"{row['parse_rate']:.2f}")


def test_folds_are_disjoint_and_complete():
    for instances in (12, 100):
        half = instances // 2
        first, second = list(range(half)), list(range(half, instances))
        assert not set(first) & set(second)
        assert sorted(first + second) == list(range(instances))


def test_stage_two_targets_are_the_hidden_weights():
    cfg, comps, data, hidden, recovered, decisions = _setup()
    rule, schema = comps.rule, comps.schemas[0]
    examples = D.generate_introspection_examples(
        data.personas, data.latents, rule, schema, data.attribute_count, cfg["model_training"]["seeds"]["introspection"]
    )
    assert len(examples) == N_PERSONAS
    import json
    for k, ex in enumerate(examples):
        answer = json.loads(ex.answer)
        assert list(answer.keys()) == data.personas[k].names, "report keys must be the scenario's attributes, in order"
        assert np.allclose(list(answer.values()), data.latents[k]), f"persona {k}: target is not the hidden latent"


if __name__ == "__main__":
    for test in (
        test_perfect_reporter_scores_one,
        test_hidden_reporter_scores_the_recovery,
        test_negated_reporter_scores_minus_one,
        test_a_persona_that_cannot_be_scored_is_counted_not_hidden,
        test_an_all_zero_report_does_not_flatter_the_mean,
        test_folds_are_disjoint_and_complete,
        test_stage_two_targets_are_the_hidden_weights,
    ):
        test()
        print(f"ok  {test.__name__}")
    print("all scoring checks passed")
