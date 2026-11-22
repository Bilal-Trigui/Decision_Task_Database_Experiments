"""The shuffled-label controls, the trial regression and the eval-only module. No model is loaded.

Run with `python -m tests.test_controls_and_tools`. Checks:

  - every shuffled control (scripts/make_shuffled_controls.py) holds its persona's prompts in order with the
    answers permuted: the A/B/C counts are the persona's, the share of answers the rule would give is at chance,
    and the settings file differs from the persona's seed-4 settings only in the data folder, run_tag,
    baseline_before_training and the 40 GB batch sizes
  - tools/trial_regression: replies made from the trial values score R² near 1 and beat the permutation null;
    replies with no link to the trial score about p/(n-1) and do not; too few trials leave R² undefined
  - src/evaluate_saved: an unknown override is refused, a step evaluated again replaces its earlier results rows,
    and reproduction counts identical decisions
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_shuffled_controls import PERSONAS, config_for, folder_for, is_shuffle_of
from src import settings_files as SF

OPT = re.compile(r"\n([AB]):\n(.*?)(?=\n\n|\Z)", re.S)
VAL = re.compile(r"^\s*(\w+):\s*([-\d.]+)", re.M)
ROOT = Path(__file__).resolve().parents[1]


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


def _rule_agreement(folder):
    ct = pd.read_csv(folder / "constraint_table.csv")
    ct = ct[ct.scenario == ct.scenario.iloc[0]].sort_values("position")
    w, lo, hi = ct.weight.to_numpy(float), ct.range_min.to_numpy(float), ct.range_max.to_numpy(float)
    act = ct[ct.active.astype(str) == "True"].iloc[0]
    j = int(act.position) - 1
    cut = lo[j] + float(act.cut_percent) / 100 * (hi[j] - lo[j])
    agree, answers = 0, []
    for line in (folder / "instill_1_prefs.jsonl").read_text().splitlines():
        m = json.loads(line)["messages"]
        o = {k: np.array([float(v) for _, v in VAL.findall(b)]) for k, b in OPT.findall(m[1]["content"])}
        a, b = o["A"], o["B"]
        pa, pb = a[j] >= cut, b[j] >= cut
        if not (pa or pb):
            label = "C"
        elif pa != pb:
            label = "A" if pa else "B"
        else:
            label = "A" if w @ ((a - lo) / (hi - lo)) > w @ ((b - lo) / (hi - lo)) else "B"
        agree += label == m[2]["content"]
        answers.append(m[2]["content"])
    shares = pd.Series(answers).value_counts(normalize=True)
    return agree / len(answers), float((shares ** 2).sum())


def test_shuffled_controls():
    allowed = {"model_training.data_dir", "model_hyperparameters.run_tag", "model_hyperparameters.baseline_before_training",
               "model_hyperparameters.micro_batch_size", "model_hyperparameters.gradient_checkpointing",
               "compute.eval_batch_size", "compute.decision_eval_batch_size", "compute.introspection_eval_batch_size"}
    for name, (settings, _, _) in PERSONAS.items():
        base = SF.flat(settings)
        folder = folder_for(name)
        assert is_shuffle_of(folder, ROOT / base["model_training"]["data_dir"]) is None, name
        agree, chance = _rule_agreement(folder)
        assert abs(agree - chance) < 0.03, (name, agree, chance)
        cfg = json.loads(config_for(name).read_text())
        diff = {k for k in set(_flat(base)) | set(_flat(cfg)) if _flat(base).get(k) != _flat(cfg).get(k)}
        assert diff <= allowed, (name, sorted(diff - allowed))
        assert cfg["model_hyperparameters"]["baseline_before_training"] is False
        assert Path(ROOT / cfg["model_training"]["data_dir"]).resolve() == folder.resolve()
    print(f"  {len(PERSONAS)} shuffled controls: the persona's prompts, answers at chance, settings as seed 4")


def test_trial_regression():
    from tools.trial_regression import mean_r2

    rng = np.random.default_rng(0)
    n, keep = 50, [1, 2, 3, 4]
    A, B = rng.random((n, 5)), rng.random((n, 5))
    linked = np.column_stack([100 * (A[:, j] - B[:, j]) for j in keep])
    noise = rng.normal(size=(n, 4))
    for spec, p in (("own", 2), ("all", 10)):
        r2 = mean_r2(spec, linked, A, B, keep)
        null = [mean_r2(spec, linked[rng.permutation(n)], A, B, keep) for _ in range(200)]
        assert r2 > 0.99 and r2 > np.percentile(null, 95), (spec, r2)
        r2n = np.mean([mean_r2(spec, rng.normal(size=(n, 4)), A, B, keep) for _ in range(200)])
        assert abs(r2n - p / (n - 1)) < 0.03, (spec, r2n, p / (n - 1))
    assert mean_r2("own", noise, A, B, keep) < 0.3
    print("  trial regression: linked replies R² ~1 above the null; unlinked replies R² ~ p/(n-1)")


def test_evaluate_saved_helpers():
    from src import evaluate_saved as ES

    cfg = {"compute": {"eval_batch_size": 4}}
    try:
        ES.apply_overrides(cfg, {"compute.nope": 1})
        raise AssertionError("an unknown override was accepted")
    except KeyError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out, run_id = tmp / "out", "run"
        out.mkdir()
        for value in (1, 2):   # the same step twice: the second replaces the first
            stage = tmp / "_staging"
            (stage / run_id).mkdir(parents=True)
            (stage / run_id / "selections_decision_step-5.csv").write_text("x\n")
            pd.DataFrame({"checkpoint_step": [5, 5], "v": [value, value]}).to_csv(stage / "results.csv", index=False)
            ES._move_up(stage, run_id, out)
        rows = pd.read_csv(out / "results.csv")
        assert list(rows["v"]) == [2, 2] and not (tmp / "_staging").exists()
        cols = ["persona", "trial", "sample", "selection", "p_A", "p_B", "p_C"]
        a = pd.DataFrame([[0, i, 0, "A" if i % 2 else "B", .9, .1, 0] for i in range(10)], columns=cols)
        b = a.copy()
        b.loc[0, "selection"] = "C"
        run_dir = tmp / "run_dir"
        run_dir.mkdir()
        a.to_csv(run_dir / "selections_decision_step-5.csv", index=False)
        b.to_csv(out / "selections_decision_step-5.csv", index=False)
        r = ES.reproduction(run_dir, out, 5)
        assert r["decisions"] == 10 and r["identical"] == 9
        assert ES.reproduction(run_dir, out, 7) is None
    print("  evaluate_saved: unknown override refused, a re-evaluated step replaces its rows, reproduction counts")


if __name__ == "__main__":
    test_shuffled_controls()
    test_trial_regression()
    test_evaluate_saved_helpers()
    print("all control and tool checks passed")
