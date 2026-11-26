"""The one random-answer control (scripts/make_shuffled_control.py). No model is loaded.

Run with `python -m tests.test_shuffled_control`. Checks:

  - the data folder holds E-negative's prompts in E-negative's order, with its scenarios and weights unchanged
  - the answers are A, B and C in the planned mix, and agree with each of the five personas' rules only at chance
    (the agreement that answers drawn without looking at the trial would have)
  - building it again gives the same bytes (the draw is seeded)
  - the settings file is E-negative's seed-4 settings except for the data folder, run_tag, baseline and the 40 GB
    batch sizes, and it passes the checks the notebooks run before training
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import make_shuffled_control as M
from scripts.make_shuffled_controls import PERSONAS
from src import settings_files as SF

OPT = re.compile(r"\n([AB]):\n(.*?)(?=\n\n|\Z)", re.S)
VAL = re.compile(r"^\s*(\w+):\s*([-\d.]+)", re.M)
ROOT = Path(__file__).resolve().parents[1]


def _messages(path):
    return [json.loads(l)["messages"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _rule_labels(prompts, rule_folder):
    """What the persona whose data folder is `rule_folder` would answer to each prompt."""
    ct = pd.read_csv(rule_folder / "constraint_table.csv")
    ct = ct[ct.scenario == ct.scenario.iloc[0]].sort_values("position")
    w, lo, hi = ct.weight.to_numpy(float), ct.range_min.to_numpy(float), ct.range_max.to_numpy(float)
    act = ct[ct.active.astype(str) == "True"].iloc[0]
    j = int(act.position) - 1
    cut = lo[j] + float(act.cut_percent) / 100 * (hi[j] - lo[j])
    labels = []
    for p in prompts:
        o = {k: np.array([float(v) for _, v in VAL.findall(b)]) for k, b in OPT.findall(p)}
        a, b = o["A"], o["B"]
        pa, pb = a[j] >= cut, b[j] >= cut
        if not (pa or pb):
            labels.append("C")
        elif pa != pb:
            labels.append("A" if pa else "B")
        else:
            labels.append("A" if w @ ((a - lo) / (hi - lo)) > w @ ((b - lo) / (hi - lo)) else "B")
    return labels


def test_data():
    source = ROOT / SF.flat(M.SOURCE_SETTINGS)["model_training"]["data_dir"]
    for name in M.FILES:
        src, out = _messages(source / name), _messages(M.OUT / name)
        assert [m[1]["content"] for m in src] == [m[1]["content"] for m in out], name
        assert [m[0] for m in src] == [m[0] for m in out], name
    for name in ("instilled_weights.csv", "scenarios.csv", "candidate_scenarios.json", "roles.csv", "constraint_table.csv"):
        assert (source / name).read_bytes() == (M.OUT / name).read_bytes(), name
    assert "answers_random" in json.loads((M.OUT / "manifest.json").read_text(encoding="utf-8"))

    train = _messages(M.OUT / "instill_1_prefs.jsonl")
    prompts, answers = [m[1]["content"] for m in train], [m[2]["content"] for m in train]
    shares = pd.Series(answers).value_counts(normalize=True)
    assert set(shares.index) == {"A", "B", "C"}
    for k, p in M.SHARES.items():
        assert abs(shares[k] - p) < 0.02, (k, shares[k], p)
    out = []
    for name, (settings, _, _) in PERSONAS.items():
        labels = _rule_labels(prompts, ROOT / SF.flat(settings)["model_training"]["data_dir"])
        agree = np.mean([x == y for x, y in zip(labels, answers)])
        chance = sum(M.SHARES[k] * v for k, v in pd.Series(labels).value_counts(normalize=True).items())
        assert abs(agree - chance) < 0.02, (name, agree, chance)
        out.append(f"{name} {agree:.3f} (chance {chance:.3f})")
    print("  answers follow no persona's rule: agreement " + ", ".join(out))


def test_rebuild_is_identical():
    source = ROOT / SF.flat(M.SOURCE_SETTINGS)["model_training"]["data_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        again = Path(tmp) / "again"
        shutil.copytree(source, again, ignore=shutil.ignore_patterns("README.md"))
        M.randomise_answers(again)
        for name in (*M.FILES, "manifest.json"):
            assert (again / name).read_bytes() == (M.OUT / name).read_bytes(), name
    print("  building it again gives the same bytes")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and k != "questions":
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def test_settings():
    from src.config import resolve
    from src.experiments import load_experiment
    from src.rules.base import check_manifest

    allowed = {"model_training.data_dir", "model_hyperparameters.run_tag", "model_hyperparameters.baseline_before_training",
               "model_hyperparameters.micro_batch_size", "model_hyperparameters.gradient_checkpointing",
               "compute.eval_batch_size", "compute.decision_eval_batch_size", "compute.introspection_eval_batch_size"}
    assert sorted(M.CARDS) == ["24gb", "40gb"] and M.CARDS["40gb"] == (M.SOURCE_SETTINGS, M.CONFIG)
    for card, (settings, config) in M.CARDS.items():
        base, cfg = _flat(SF.flat(M.SOURCE_SETTINGS)), _flat(json.loads(config.read_text(encoding="utf-8")))
        diff = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
        assert diff <= allowed, (card, sorted(diff - allowed))
        loaded = load_experiment(str(config), None, None, None)
        mh = loaded["model_hyperparameters"]
        assert mh["baseline_before_training"] is False and mh["start_from"] is None and mh["run_tag"] == "shuffled"
        # the card's sizing is the persona file's for that card: gradient checkpointing off on both, eval batches by card,
        # and the batch of 10 in passes of 2 on 24 GB (passes of 5 ran an L4 out of memory, 2026-11-22), of 5 on 40 GB
        sized = SF.flat(settings)
        assert mh["micro_batch_size"] == sized["model_hyperparameters"]["micro_batch_size"] == {"24gb": 2, "40gb": 5}[card]
        assert mh["gradient_checkpointing"] is sized["model_hyperparameters"]["gradient_checkpointing"] is False
        assert loaded["compute"]["eval_batch_size"] == sized["compute"]["eval_batch_size"] == {"24gb": 48, "40gb": 96}[card]
        assert Path(ROOT / loaded["model_training"]["data_dir"]).resolve() == M.OUT.resolve()
        check_manifest(loaded["model_training"]["data_dir"], resolve(loaded).rule, loaded)
    rs = SF.flat(M.A_FORMAT)["report_schema"]
    assert not rs.get("reply_prefix") and not rs.get("show_neither"), "A_FORMAT is not A and B's report format"
    print("  settings: E-negative's seed-4 run, random-answer data, a 40 GB and a 24 GB file, no base-model eval; both pass the preflight")


if __name__ == "__main__":
    test_data()
    test_rebuild_is_identical()
    test_settings()
    print("all shuffled-control checks passed")
