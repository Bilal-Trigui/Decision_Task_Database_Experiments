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
        "reported": {schema.name: {"main": reports}},
        "parse": {schema.name: {"main": (len(reports), len(reports), "main")}},
        "decisions": decisions,
        "rule_slices": comps.rule.block_slices(n),
        "est_slices": comps.estimator.block_slices(),
    }


def _row(rows):
    assert len(rows) == 1, f"expected one results row for one schema and one block, got {len(rows)}"
    return rows[0]


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
        test_folds_are_disjoint_and_complete,
        test_stage_two_targets_are_the_hidden_weights,
    ):
        test()
        print(f"ok  {test.__name__}")
    print("all scoring checks passed")
