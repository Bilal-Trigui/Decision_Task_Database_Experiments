"""Estimation trials as a sweep: recovering the rule from several trial counts in one run.

model_estimating.estimation_trials_per_persona is one number, or a list. With a list the model
decides once, on the largest count, and the rule is recovered again from the first N decisions
for each N, every count with its own results rows.

  1. The setting: a number or a list; a list is sorted; bad values are refused.
  2. Recovery from the first N trials uses exactly those trials, and writes its own file.
  3. summarize gives every count its own rows, and more trials recover the rule better.
  4. The plots: checkpoints keep to the largest count, and the sweep gets its own figure.

Run with `python -m tests.test_estimation_trials`.
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.config import SettingsError, apply_defaults, estimation_counts, hyperparameter_columns, resolve, validate
from src.configs.test import TEST
from src.evaluate import _recover, summarize
from src.plots import plot_checkpoints, plot_estimation_trials

N_PERSONAS = 12
COUNTS = [10, 40, 160]


def _cfg(trials):
    cfg = copy.deepcopy(TEST)
    cfg["model_training"]["instances"] = N_PERSONAS
    cfg["model_estimating"].update(estimation_trials_per_persona=trials, chance_draws=5)
    cfg["_source"] = "test"
    return validate(apply_defaults(cfg))


def test_the_setting():
    assert estimation_counts(_cfg(50)) == [50]
    assert estimation_counts(_cfg([160, 10, 40])) == COUNTS
    assert hyperparameter_columns(_cfg(50))["estimation_trials_per_persona"] == 50
    assert json.loads(hyperparameter_columns(_cfg([160, 10, 40]))["estimation_trials_per_persona"]) == COUNTS
    for bad in (0, [], [10, 10], [10, 0], [True], 2.5, "50"):
        try:
            _cfg(bad)
        except SettingsError as err:
            assert "estimation_trials_per_persona" in str(err), (bad, str(err))
            continue
        raise AssertionError(f"estimation_trials_per_persona {bad!r} was accepted")
    print("  a number or a list, sorted; seven bad values refused")


def _decisions(cfg, comps, data):
    """Clean choices from the hidden latent on the largest count, laid out as collect lays them out."""
    n, rule = data.attribute_count, comps.rule
    trials = D.fresh_trials(data.personas, estimation_counts(cfg)[-1], cfg["model_training"]["seeds"]["verification"])
    rows = []
    for k in range(N_PERSONAS):
        for t_i, t in enumerate(trials[k]):
            label = rule.label(data.latents[k], t)
            rows.append({"persona": k, "trial": t_i, "selection": label, "label": label,
                         **{f"A_attribute_{i+1}": t.option_A.attributes[i]["value"] for i in range(n)},
                         **{f"B_attribute_{i+1}": t.option_B.attributes[i]["value"] for i in range(n)}})
    return pd.DataFrame(rows)


def test_the_first_n_trials():
    cfg = _cfg(COUNTS)
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    decisions = _decisions(cfg, comps, data)
    personas = list(range(N_PERSONAS))
    with tempfile.TemporaryDirectory() as d:
        few = _recover(comps.estimator, decisions, data, personas, "t", Path(d), trials=10)
        by_hand = _recover(comps.estimator, decisions[decisions["trial"] < 10], data, personas, "u", Path(d))
        full = _recover(comps.estimator, decisions, data, personas, "t", Path(d))
        assert (Path(d) / "recovered_t_trials-10.csv").exists() and (Path(d) / "recovered_t.csv").exists()
    assert all(np.array_equal(few[k], by_hand[k]) for k in personas)
    assert not all(np.array_equal(few[k], full[k]) for k in personas)
    print("  the fit from 10 trials is the fit on the first 10 decisions, saved beside the main one")


def test_every_count_gets_its_rows():
    cfg = _cfg(COUNTS)
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    decisions = _decisions(cfg, comps, data)
    personas = list(range(N_PERSONAS))
    with tempfile.TemporaryDirectory() as d:
        by_count = {N: _recover(comps.estimator, decisions, data, personas, "t", Path(d), trials=N) for N in COUNTS}
        schema = comps.schemas[0]
        reports = {k: data.latents[k] for k in personas}
        detail = {"hidden": {k: data.latents[k] for k in personas}, "recovered": by_count[COUNTS[-1]],
                  "recovered_by_count": by_count, "reported": {schema.name: {"main": ("main", reports)}},
                  "parse": {schema.name: {"main": (N_PERSONAS, N_PERSONAS, "main")}}, "decisions": decisions}
        scores_path = Path(d) / "scores.csv"
        rows = summarize([detail], cfg, comps, data, "r", "decision", 0, 1, scores_path=scores_path)
        scores = pd.read_csv(scores_path)
        frame = pd.DataFrame(rows)
        assert sorted(frame["estimation_trials"]) == COUNTS, list(frame["estimation_trials"])
        assert sorted(scores["estimation_trials"].unique()) == COUNTS
        recovery = frame.set_index("estimation_trials")["recovered_vs_hidden"]
        assert recovery[160] > recovery[10], recovery.to_dict()
        # the plots: checkpoint figures keep to the largest count, the sweep gets its own
        frame["run_id"] = "r"
        assert plot_estimation_trials(frame, "r", d).exists()
        assert plot_checkpoints(frame, "r", d).exists()
        assert plot_estimation_trials(frame[frame["estimation_trials"] == 160], "r", d) is None
    print("  recovered vs hidden by trial count: "
          + ", ".join(f"{N} trials {recovery[N]:.3f}" for N in COUNTS) + "; one row each")


if __name__ == "__main__":
    for test in (test_the_setting, test_the_first_n_trials, test_every_count_gets_its_rows):
        test()
        print(f"ok  {test.__name__}")
    print("all estimation trial checks passed")
