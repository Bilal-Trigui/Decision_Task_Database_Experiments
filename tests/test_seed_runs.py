"""The seed repeats in configs/minimum_threshold_one_persona/seeds/ (scripts/make_seed_runs.py).

Run with `python -m tests.test_seed_runs`. Checks:

  - every persona (A, A-inverse, B, E-positive, E-negative) has seeds 5 and 6 and no others, each one 40 GB file
    that loads and validates
  - each asks exactly the nine questions of base/persona_a/one_persona_no_trial_a.json, with no reply prefix and
    C not shown
  - none scores the base model (baseline_before_training false), and each trains from step 0 (no start_from)
  - against its persona's own settings file, the only fields that differ are the seed, the run_tag, the report
    questions, baseline_before_training and the 40 GB batch sizes, so the data folder, the data seeds and every
    training setting are kept
  - no two files share a run_tag, and none uses seed 4 (the existing runs')
"""
import json
from pathlib import Path

from scripts.make_seed_runs import OUT, PERSONAS, QUESTIONS_FROM
from src import settings_files as SF
from src.config import load

SEEDS = (5, 6)
ALLOWED = {"model_hyperparameters.seed", "model_hyperparameters.run_tag", "model_hyperparameters.micro_batch_size",
           "model_hyperparameters.baseline_before_training",
           "model_hyperparameters.gradient_checkpointing", "compute.eval_batch_size", "compute.decision_eval_batch_size",
           "compute.introspection_eval_batch_size"}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def _files():
    for name, (folder, source, _, _) in PERSONAS.items():
        for seed in SEEDS:
            yield name, source, seed, OUT / folder / f"one_persona_{name}_nine_q_seed{seed}_40gb.json"


def test_files_exist_and_load():
    expected = {p for *_, p in _files()}
    present = set(OUT.glob("*/*.json"))
    assert present == expected, (sorted(p.name for p in present - expected), sorted(p.name for p in expected - present))
    for _, _, seed, path in _files():
        assert path.exists(), path
        cfg = load(str(path))
        assert cfg["model_hyperparameters"]["seed"] == seed
        assert cfg["model_hyperparameters"]["micro_batch_size"] == 5 and not cfg["model_hyperparameters"]["gradient_checkpointing"]
        assert cfg["compute"]["decision_eval_batch_size"] == 96
        assert cfg["model_hyperparameters"]["baseline_before_training"] is False, path.name
        assert cfg["model_hyperparameters"]["start_from"] is None, path.name
    print(f"  {len(PERSONAS) * len(SEEDS)} seed files load, 40 GB batch sizes, no base-model evaluation, trained from step 0")


def test_nine_questions():
    want = json.loads(Path(QUESTIONS_FROM).read_text())["report_schema"]["questions"]
    want_asked = [q for q, v in want.items() if v.get("ask")]
    assert len(want_asked) == 9
    for _, _, _, path in _files():
        rs = json.loads(path.read_text())["report_schema"]
        assert rs["questions"] == want, path.name
        assert rs["reply_prefix"] == "" and rs["show_neither"] is False, path.name
    print(f"  every file asks {want_asked}")


def test_only_the_seed_and_questions_change():
    for _, source, _, path in _files():
        a = _flat(SF.flat(source))
        b = _flat(json.loads(path.read_text()))
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k) and not k.startswith("report_schema.")}
        assert diff <= ALLOWED, (path.name, sorted(diff - ALLOWED))
    print("  against each persona's own file only seed, run_tag and batch sizes differ (and the report questions)")


def test_tags_unique_and_no_seed_4():
    tags = [json.loads(p.read_text())["model_hyperparameters"]["run_tag"] for *_, p in _files()]
    assert len(set(tags)) == len(tags), tags
    assert all(json.loads(p.read_text())["model_hyperparameters"]["seed"] != 4 for *_, p in _files())
    print(f"  {len(tags)} distinct run tags, none on seed 4")


def test_bootstrap_resamples_trials_not_replies():
    """tools/persona_readouts.bootstrap draws whole trials: ten identical replies to each trial (ten steps that
    answered alike) give exactly the interval one reply per trial gives, where drawing replies one by one would
    narrow it. And replies that never vary give a zero-width interval."""
    import numpy as np
    from tools.persona_readouts import bootstrap

    rng = np.random.default_rng(1)
    true, dims = np.array([-83.0, -38.0, -100.0, 31.0]), ["a", "b", "c", "d"]
    rep = rng.choice([-100.0, -50.0, 50.0, 100.0], size=(50, 4))
    keys = np.array([f"t{i}" for i in range(50)])
    once = bootstrap(rep, keys, true, dims, 300, seed=3)
    tenfold = bootstrap(np.repeat(rep, 10, axis=0), np.repeat(keys, 10), true, dims, 300, seed=3)
    for name in ("cos4", "sign_a", "neg_share", "spearman4_each"):
        assert np.isclose(once[f"{name}_lo"], tenfold[f"{name}_lo"]) and np.isclose(once[f"{name}_hi"], tenfold[f"{name}_hi"]), name
    assert once["boot_trials"] == tenfold["boot_trials"] == 50
    flat = bootstrap(np.tile(true, (50, 1)), keys, true, dims, 100)
    assert flat["cos4_lo"] == flat["cos4_hi"] and np.isclose(flat["cos4_lo"], 1.0)
    print("  bootstrap resamples whole trials (tenfold copies, same interval); constant replies, zero width")


if __name__ == "__main__":
    test_files_exist_and_load()
    test_nine_questions()
    test_only_the_seed_and_questions_change()
    test_tags_unique_and_no_seed_4()
    test_bootstrap_resamples_trials_not_replies()
    print("all seed-run checks passed")
