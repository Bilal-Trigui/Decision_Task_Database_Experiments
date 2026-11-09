"""Natural-mix validation and the gradient cap.

Run with `python -m tests.test_validation_and_clipping`. No model is needed. Checks:

  - the natural validation trials ignore the training quotas: every cut-sweep mix gets the same trials
  - each trial's kind and label agree with the rule, under the minimum and the maximum threshold
  - no validation trial is a training trial, and the training data is exactly what the folder holds
  - both new settings are validated, and both default to off
  - max_grad_norm caps the gradient, and leaves it alone when null
  - train_log.csv's new columns migrate an old log's rows in place
"""
import csv
import json
import tempfile
from pathlib import Path

import torch

from src import data as D
from src.config import SettingsError, apply_defaults, load, resolve, validate
from src.train import TRAIN_LOG_COLUMNS, _append_log


def _loaded(path, natural=1000):
    cfg = load(path)
    cfg["model_training"]["natural_val_trials_per_persona"] = natural
    return cfg, resolve(cfg).rule


def test_same_trials_whatever_the_training_mix():
    prompts = {}
    for mix in ("w75_c25", "w00_c100"):
        cfg, rule = _loaded(f"configs/cut_sweep_{mix}.json", natural=200)
        prompts[mix] = [e.user for e in D.load_data(cfg, rule).val]
    assert prompts["w75_c25"] == prompts["w00_c100"], "the training mix changed the validation trials"
    print("  w75_c25 and w00_c100 are checked on the same 200 trials")


def test_kinds_and_labels_follow_the_rule():
    for path in ("configs/one_persona_minimum_threshold.json", "configs/one_persona_maximum_threshold.json"):
        cfg, rule = _loaded(path)
        data = D.load_data(cfg, rule)
        counts = {}
        for e in data.val:
            counts[e.decided_by] = counts.get(e.decided_by, 0) + 1
            if e.decided_by == "neither":
                assert e.answer == "C", (e.decided_by, e.answer)
            else:
                assert e.answer in "AB", (e.decided_by, e.answer)
        assert set(counts) == {"weights", "neither", "cut_agrees", "cut_overturns"}, counts
        assert sum(counts.values()) == 1000
        # as they come, with a cut at 50%: about a quarter of the trials each
        assert all(150 < c < 350 for c in counts.values()), counts
        assert not {e.user for e in data.train} & {e.user for e in data.val}, "a validation trial is in training"
        print(f"  {rule.name}: {counts}")


def test_the_maximum_rule_reverses_the_kinds():
    """The same draws under the two rules: a trial both options pass under one, both fail under the other."""
    low, high = _loaded("configs/one_persona_minimum_threshold.json"), _loaded("configs/one_persona_maximum_threshold.json")
    a, b = D.load_data(*low), D.load_data(*high)
    pairs = [(x.decided_by, y.decided_by) for x, y in zip(a.val, b.val) if x.user == y.user]
    assert len(pairs) > 900, len(pairs)
    swapped = sum((x, y) in (("weights", "neither"), ("neither", "weights")) for x, y in pairs)
    same_two = sum(x == y and x in ("weights", "neither") for x, y in pairs)
    assert swapped > 300, swapped
    # only a value exactly at the cut can pass both screens
    assert same_two < 0.05 * len(pairs), same_two
    print(f"  {swapped} of {len(pairs)} trials swap between weights and neither; {same_two} keep their kind")


def test_training_data_is_untouched():
    with_natural = D.load_data(*_loaded("configs/one_persona_minimum_threshold.json"))
    without = D.load_data(*_loaded("configs/one_persona_minimum_threshold.json", natural=None))
    assert [e.messages for e in with_natural.train] == [e.messages for e in without.train]
    assert len(without.val) == 100, "null must keep the folder's validation file"
    print(f"  {len(with_natural.train)} training trials identical; null keeps the folder's {len(without.val)} validation trials")


def test_settings_are_checked_and_default_off():
    filled = apply_defaults({})
    assert filled["model_hyperparameters"]["max_grad_norm"] is None
    assert filled["model_training"]["natural_val_trials_per_persona"] is None
    for block, field, bad in (("model_hyperparameters", "max_grad_norm", 0), ("model_hyperparameters", "max_grad_norm", True),
                              ("model_training", "natural_val_trials_per_persona", 0),
                              ("model_training", "natural_val_trials_per_persona", 1.5)):
        cfg = load("configs/one_persona_minimum_threshold.json")
        cfg[block][field] = bad
        try:
            validate(cfg)
        except SettingsError:
            continue
        raise AssertionError(f"{block}.{field} = {bad!r} was accepted")
    print("  both off by default; bad values refused")


def test_saved_runs_can_still_be_read():
    """A settings.json from before these fields reads them as off, which is what that run ran with."""
    saved = json.loads(Path("results/20261106-152354_paper_minimum_threshold_Qwen3-8B/settings.json").read_text())
    assert "max_grad_norm" not in saved["model_hyperparameters"]
    filled = apply_defaults(saved)
    assert filled["model_hyperparameters"]["max_grad_norm"] is None
    assert filled["model_training"]["seeds"]["validation"] == 8
    print("  an older run's settings read as uncapped, with the folder's validation file")


def test_the_cap():
    def grads(limit):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
        before = torch.nn.utils.clip_grad_norm_([p], limit).item() if limit else p.grad.norm().item()
        return before, p.grad.norm().item()
    assert grads(1.0) == (5.0, 1.0) or abs(grads(1.0)[1] - 1.0) < 1e-5
    assert grads(None) == (5.0, 5.0)
    print("  a gradient of size 5 is logged as 5 and capped to 1; with null it stays 5")


def test_log_migrates():
    old_columns = [c for c in TRAIN_LOG_COLUMNS if not c.startswith("val_acc_") and c != "grad_norm"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_log.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow({c: 1 for c in old_columns})
        _append_log(path, [{c: 2 for c in TRAIN_LOG_COLUMNS}])
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    assert list(rows[0]) == TRAIN_LOG_COLUMNS
    assert rows[0]["loss"] == "1" and rows[0]["grad_norm"] == "" and rows[1]["grad_norm"] == "2"
    print("  an old log keeps its rows, with the new columns blank")


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            check()
    print("all passed")
