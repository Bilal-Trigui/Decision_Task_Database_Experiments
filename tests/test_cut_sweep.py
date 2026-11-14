"""Branching from another run's checkpoint, and the cut-sweep training data.

Run with `python -m tests.test_cut_sweep`. The last check trains Qwen3-0.6B on the CPU (the TEST
settings), so the model is downloaded on first use. Checks:

  - cut_overturns_only: every screen-decided training trial is one the weights alone would have
    answered the other way, and the shares come out exact, down to weight_decided_share 0
  - every data/minimum_threshold_one_persona/cut_sweep/persona_a/ folder holds exactly the shares its config names
  - start_from, save_optimizer and run_tag are validated
  - a run branched from step 5 of another run trains steps 6 to 10 exactly as the source did
    (same adapter, AdamW state and data order), keeps AdamW's state only at its last checkpoint
    under save_optimizer "last", and carries its run_tag in the run id
"""
import collections
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.config import SettingsError, apply_defaults, load, resolve, validate
from src.configs.test import TEST


def _kinds(rule, latent, trials):
    out = collections.Counter()
    for t in trials:
        key = rule.quota_key(latent, t)
        out[key if key in ("weights", "C") else ("overturns" if rule.weights_choice(latent, t) != t.label else "agrees")] += 1
    return out


def test_cut_overturns_only():
    from src.config import load_component

    cfg = load("configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_w65_c35.json")
    data = D.load_data(cfg, resolve(cfg).rule)
    for share, neither in ((0.65, 0.175), (0.0, 0.5), (0.3, 0.1)):
        params = {**cfg["decision_rule"], "weight_decided_share": share, "neither_share": neither}
        params.pop("type")
        rule = load_component("rules", "paper_minimum_threshold", params)
        train, val = D.generate_decision_trials(data.personas, data.latents, rule, 400, 40, 11)
        for trials, n in ((train[0], 400), (val[0], 40)):
            k = _kinds(rule, data.latents[0], trials)
            assert k["agrees"] == 0, f"a screen trial the weights answer the same way slipped in: {k}"
            assert k["weights"] == round(share * n) and k["C"] == round(neither * n), (share, neither, k)
            assert sum(k.values()) == n
    try:
        params = {**cfg["decision_rule"], "weight_decided_share": None}
        params.pop("type")
        load_component("rules", "paper_minimum_threshold", params)
        raise AssertionError("cut_overturns_only without weight_decided_share was allowed")
    except ValueError:
        pass
    print("  cut_overturns_only: exact shares, no screen trial the weights would answer the same way")


def test_sweep_folders():
    # the 16 GB files: each _24gb twin names the same data folder, so checking it again adds nothing
    configs = sorted(p for p in Path("configs/minimum_threshold_one_persona/cut_sweep/persona_a").glob("cut_sweep_*.json")
                     if not p.stem.endswith("_24gb"))
    assert configs, "no configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_*.json"
    for path in configs:
        cfg = load(str(path))
        rule = resolve(cfg).rule
        data = D.load_data(cfg, rule)
        dr, n = cfg["decision_rule"], len(data.train)
        trials, _ = D.generate_decision_trials(data.personas, data.latents, rule, n, cfg["model_training"]["val_trials_per_persona"],
                                               cfg["model_training"]["seeds"]["trials"])
        assert [t.label for t in trials[0]] == [e.messages[-1]["content"] for e in data.train], f"{path}: the file is not these trials"
        k = _kinds(rule, data.latents[0], trials[0])
        assert k["weights"] == round(dr["weight_decided_share"] * n) and k["C"] == round(dr["neither_share"] * n), (path, k)
        assert k["agrees"] == 0 and k["overturns"] == n - k["weights"] - k["C"], (path, k)
    print(f"  {len(configs)} cut sweep folders hold exactly the shares their configs name")


def _cfg(**hyper):
    cfg = apply_defaults(copy.deepcopy(TEST))
    cfg["model_hyperparameters"].update(hyper)
    return validate(cfg)


def test_branch_settings():
    good = {"run_id": "r", "step": 5}
    _cfg(start_from=good, baseline_before_training=False, save_optimizer="last", run_tag="cut-w65")
    bad = [
        dict(start_from={"run_id": "r"}, baseline_before_training=False),
        dict(start_from={"run_id": "r", "step": 0}, baseline_before_training=False),
        dict(start_from={"run_id": "r", "step": 10}, baseline_before_training=False),    # not below training_steps 10
        dict(start_from=good),                                                         # baseline_before_training on
        dict(start_from=good, baseline_before_training=False, introspection_training=True),
        dict(save_optimizer="some"),
        dict(run_tag="has space"),
    ]
    for hyper in bad:
        try:
            _cfg(**hyper)
            raise AssertionError(f"accepted {hyper}")
        except SettingsError:
            pass
    print("  start_from, save_optimizer and run_tag validated")


def test_branch_is_exact():
    from src.pipeline import run

    source_cfg = _cfg(training_steps=10, checkpoint_every=5)
    with tempfile.TemporaryDirectory() as tmp:
        results, ckpts = Path(tmp) / "results", Path(tmp) / "checkpoints"
        run(copy.deepcopy(source_cfg), results_dir=results, checkpoint_root=ckpts)
        log = pd.read_csv(results / "train_log.csv")
        source = log["run_id"].iloc[0]
        assert (ckpts / source / "decision" / "step-5" / "optimizer.pt").exists()

        branch_cfg = _cfg(training_steps=15, checkpoint_every=5, start_from={"run_id": source, "step": 5},
                          baseline_before_training=False, save_optimizer="last", run_tag="br")
        rows = run(copy.deepcopy(branch_cfg), results_dir=results, checkpoint_root=ckpts)
        branch = rows[-1]["run_id"]
        assert branch.endswith("_br") and branch != source, branch
        log = pd.read_csv(results / "train_log.csv")
        theirs = log[(log["run_id"] == source) & log["step"].between(6, 10)]["loss"].to_numpy()
        ours = log[(log["run_id"] == branch) & log["step"].between(6, 10)]["loss"].to_numpy()
        steps = log[log["run_id"] == branch]["step"].tolist()
        assert steps == list(range(6, 16)), steps
        assert np.allclose(theirs, ours, atol=1e-4), f"branched losses differ from the source's: {theirs} vs {ours}"
        table = pd.read_csv(results / "results.csv")
        assert set(table[(table["run_id"] == branch) & (table["stage"] == "decision")]["checkpoint_step"]) == {10, 15}
        assert not (table[table["run_id"] == branch]["stage"] == "base_model").any()
        assert not (ckpts / branch / "decision" / "step-10" / "optimizer.pt").exists()
        assert (ckpts / branch / "decision" / "step-10" / "adapter_model.safetensors").exists()
        assert (ckpts / branch / "decision" / "step-15" / "optimizer.pt").exists()
        saved = json.loads((results / branch / "settings.json").read_text())
        assert saved["model_hyperparameters"]["start_from"] == {"run_id": source, "step": 5}
        # continuing the branch on: resume from its last checkpoint (AdamW saved there) to a later step
        longer = copy.deepcopy(branch_cfg)
        longer["model_hyperparameters"]["training_steps"] = 20
        run(longer, results_dir=results, checkpoint_root=ckpts, resume={"run_id": branch, "from_step": 15})
        log = pd.read_csv(results / "train_log.csv")
        assert log[log["run_id"] == branch]["step"].tolist() == list(range(6, 21))
        table = pd.read_csv(results / "results.csv")
        assert set(table[(table["run_id"] == branch) & (table["stage"] == "decision")]["checkpoint_step"]) == {10, 15, 20}
        assert (ckpts / branch / "decision" / "step-20" / "optimizer.pt").exists()
        missing = _cfg(training_steps=15, start_from={"run_id": source, "step": 7}, baseline_before_training=False)
        try:
            run(missing, results_dir=results, checkpoint_root=ckpts)
            raise AssertionError("branched from a checkpoint that does not exist")
        except SettingsError:
            pass
    print("  a branch from step 5 trains steps 6-10 exactly as its source did; optimizer kept at the last checkpoint only")


if __name__ == "__main__":
    test_cut_overturns_only()
    test_sweep_folders()
    test_branch_settings()
    test_branch_is_exact()
    print("all cut sweep checks passed")
