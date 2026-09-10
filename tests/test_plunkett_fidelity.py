"""Fidelity of the ported generators against Plunkett's own files in data/plunkett.

Run with `python -m tests.test_plunkett_fidelity` or pytest. Every check is
byte-for-byte or value-for-value:
  1. scenarios.csv rebuilt from scenario_definitions.json + roles.csv with the roles seed.
  2. instilled_weights.csv and both trial JSONL files rebuilt by the linear rule's make_dataset.
  3. The Experiment 2 introspection-training file rebuilt with the introspection seed.
  4. The data folder holds exactly Plunkett's file names.
  5. The 50 verification trials per persona (seed 5) are structurally valid: scenario order,
     values inside each attribute's range, rounded with his rule.
     They cannot be compared value for value with his recorded selections files: those were
     collected through tqdm's gather, which schedules the trial coroutines through a Python
     set, so the RNG was consumed in a scrambled order. His GPT-4o and GPT-4o-mini selections
     files record different option values for the same seed, which confirms it. The pipeline
     draws them sequentially, so its verification trials are reproducible.
Reads model_training.seeds (Plunkett's values), object.attribute_count = 5.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.reports.linear import LinearSchema
from src.rules.linear import LinearRule

PLUNKETT = Path(__file__).resolve().parents[1] / "data" / "plunkett"
SEEDS = {"roles": 0, "weights": 1, "trials": 2, "verification": 5, "introspection": 6}


def test_scenarios_csv():
    definitions = D.read_definitions(PLUNKETT)
    roles = D.load_roles(PLUNKETT)
    rebuilt = D.assemble_scenarios(definitions, roles, SEEDS["roles"]).to_csv(index=False)
    assert rebuilt == (PLUNKETT / "scenarios.csv").read_text(), "scenarios.csv differs"


def test_weights_and_trials():
    with tempfile.TemporaryDirectory() as tmp:
        LinearRule().make_dataset(PLUNKETT, tmp, seed=SEEDS["weights"], trials_seed=SEEDS["trials"])
        for name in ("instilled_weights.csv", "instill_100_prefs.jsonl", "instill_100_prefs_val.jsonl"):
            assert (Path(tmp) / name).read_bytes() == (PLUNKETT / name).read_bytes(), f"{name} differs"


def test_introspection_training_file():
    rule = LinearRule()
    scenarios = D.load_scenarios(PLUNKETT, 5)[:100]
    latents = D.load_hidden_latents(PLUNKETT, scenarios, rule, 5)
    examples = D.generate_introspection_examples(scenarios, latents, rule, LinearSchema(5), 5, SEEDS["introspection"])
    rebuilt = "\n".join(D.example_to_json(e) for e in examples)
    assert rebuilt == (PLUNKETT / "instilled_weights_100_training.jsonl").read_text(), "introspection file differs"


# The inputs of Plunkett's experiment, which is all this folder keeps. His own model
# outputs (the twenty GPT-4o and GPT-4o-mini result CSVs) were removed: the pipeline
# never reads them, and the comparison for a replication is his published numbers.
# They remain untouched in plunkett-self-interpretability/data if they are wanted back.
PLUNKETT_FILES = sorted(
    ["candidate_scenarios.json", "instill_100_prefs.jsonl", "instill_100_prefs_val.jsonl",
     "instilled_weights.csv", "instilled_weights_100_training.jsonl", "roles.csv", "scenarios.csv"]
)


def test_file_names_are_plunketts():
    present = sorted(p.name for p in PLUNKETT.iterdir() if p.is_file() and not p.name.startswith("."))
    generated = {D.INTROSPECTION_TRAINING_CSV, "manifest.json"}
    present = [name for name in present if name not in generated]
    assert present == PLUNKETT_FILES, f"data/plunkett file names differ from Plunkett's: {set(present) ^ set(PLUNKETT_FILES)}"


def test_verification_trials():
    scenarios = D.load_scenarios(PLUNKETT, 5)[:100]
    trials = D.fresh_trials(scenarios, 50, SEEDS["verification"])
    recorded_path = PLUNKETT / "gpt-4o-2024-08-06_instilled_selections.csv"
    if recorded_path.exists():  # only when a folder still carries his recorded runs
        recorded = pd.read_csv(recorded_path)
        assert len(recorded) == 5000
        assert list(recorded["scenario"]) == [sc.short_name for sc in scenarios for _ in range(50)]
    for sc, persona in zip(scenarios, trials):
        for t in persona:
            for option in (t.option_A, t.option_B):
                for a, attr in zip(option.attributes, sc.attributes):
                    lo, hi = attr["range"]
                    slack = 0.5 * 10 ** (-D.rounding_precision(attr))  # his rounding can step just past the range
                    assert lo - slack <= a["value"] <= hi + slack
                    assert a["value"] == round(a["value"], D.rounding_precision(attr))
    again = D.fresh_trials(scenarios, 50, SEEDS["verification"])
    assert all(
        np.array_equal(np.concatenate([a.option_A.values, a.option_B.values]), np.concatenate([b.option_A.values, b.option_B.values]))
        for pa, pb in zip(trials, again) for a, b in zip(pa, pb)
    ), "verification trials must be reproducible from the seed"


if __name__ == "__main__":
    for test in (test_scenarios_csv, test_weights_and_trials, test_introspection_training_file, test_file_names_are_plunketts, test_verification_trials):
        test()
        print(f"ok  {test.__name__}")
    print("all fidelity checks passed")
