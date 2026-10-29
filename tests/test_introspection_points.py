"""Introspection at several decision steps: the setting, the branch, and what it writes.

model_hyperparameters.introspection_at_steps lists the decision steps that introspection training
branches from. At each, both folds train from the decision adapter as it stands and are scored,
then the adapter and every random number generator are put back, so decision training carries on
exactly as it would have without the branch.

  1. The setting: null means the last step only, a list is sorted, and a bad list is refused.
  2. Branch names: each introspection point gets its own checkpoint folder and file tag.
  3. Undoing a branch is exact: training after it matches training that never branched.
  4. train_log.csv written by older code is migrated to the new column rather than misaligned.
  5. The plots place each branch at the step it came from, and still read older results.
  6. The base model, untrained: on by default, refused unless true or false, drawn at step 0.
  7. Stage two never reads the first token: it is measured over every answer token.

Run with `python -m tests.test_introspection_points`.
"""
import copy
import csv
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import SettingsError, introspection_points, load
from src.plots import _checkpoint_x, plot_checkpoints, plot_training
from src.train import (TRAIN_LOG_COLUMNS, _accuracy_counts, _append_log, evaluate_examples, restore, rng_state,
                       set_rng_state, snapshot, stage_label)


def _load_with(**fields):
    cfg = json.loads(Path("configs/default.json").read_text())
    cfg["model_hyperparameters"].update(fields)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "points.json"
        path.write_text(json.dumps(cfg))
        return load(str(path))


def test_the_setting():
    assert introspection_points(load("configs/default.json")) == [1500]
    assert introspection_points(load("configs/atkinson.json")) == []
    assert introspection_points(load("configs/minimum_threshold_8b.json")) == [1500, 2500, 3000]
    assert introspection_points(_load_with(introspection_at_steps=[900, 300, 1500])) == [300, 900, 1500]
    # a range: start, start + every, ..., and end always included
    assert introspection_points(_load_with(introspection_at_steps={"start": 300, "end": 1500, "every": 600})) == [300, 900, 1500]
    assert introspection_points(_load_with(introspection_at_steps={"start": 300, "end": 1400, "every": 600})) == [300, 900, 1400]
    assert introspection_points(_load_with(introspection_at_steps={"start": 600, "end": 1500})) == [600, 1500]
    for bad, says in (([], "non-empty list"), ([0], "outside"), ([1501], "outside"), ([300, 300], "twice"),
                      ([True], "non-empty list"), (300, "non-empty list"), ([300.0], "non-empty list"),
                      ({"start": 0, "end": 900}, "outside"), ({"start": 900, "end": 300}, "start <= end"),
                      ({"start": 300, "end": 900, "every": 0}, "every at least 1"), ({"from": 300}, "as a range takes")):
        try:
            _load_with(introspection_at_steps=bad)
        except SettingsError as err:
            assert says in str(err), (bad, str(err))
            continue
        raise AssertionError(f"introspection_at_steps {bad!r} was accepted")
    try:
        _load_with(introspection_at_steps=[300], introspection_training=False)
    except SettingsError as err:
        assert "introspection_training is false" in str(err), str(err)
    else:
        raise AssertionError("introspection_at_steps was accepted with introspection_training off")
    print("  null is the last step, lists are sorted, ranges expand, eleven bad values and a switched-off stage refused")


def test_branch_names():
    assert stage_label("decision") == "decision"
    assert stage_label("introspection", 1) == "introspection-fold1"
    assert stage_label("introspection", 2, 1500) == "introspection-from-1500-fold2"
    assert stage_label("introspection", "both", 2500) == "introspection-from-2500-foldboth"
    print("  decision, introspection-fold1, introspection-from-1500-fold2")


def _train(model, optimizer, steps, start=0):
    """A few steps that draw from all three generators, the way training does through shuffling and dropout."""
    losses = []
    for step in range(start, start + steps):
        x = torch.randn(4, 3) + random.random() + float(np.random.rand())
        loss = (torch.nn.functional.dropout(model(x), 0.5, training=True) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_undoing_a_branch_is_exact():
    _seed()
    plain = torch.nn.Linear(3, 2)
    branched = copy.deepcopy(plain)
    start = rng_state()

    opt = torch.optim.AdamW(plain.parameters(), lr=0.1)
    expected = _train(plain, opt, 6)

    set_rng_state(start)
    opt = torch.optim.AdamW(branched.parameters(), lr=0.1)
    got = _train(branched, opt, 3)
    generators, state = rng_state(), snapshot(branched)
    # the branch: its own optimizer, as each introspection fold has, moving the weights a long way
    detour = torch.optim.AdamW(branched.parameters(), lr=1.0)
    _train(branched, detour, 5)
    assert not torch.equal(branched.weight, plain.weight)
    restore(branched, state)
    set_rng_state(generators)
    got += _train(branched, opt, 3, start=3)

    assert got == expected, (got, expected)
    assert torch.equal(branched.weight, plain.weight) and torch.equal(branched.bias, plain.bias)
    print("  six steps with a branch at step 3 match six steps without one, loss for loss and weight for weight")


def test_an_old_train_log_is_migrated():
    old_columns = [c for c in TRAIN_LOG_COLUMNS if c != "introspection_from_step"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "train_log.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow({c: (7 if c == "step" else "old") for c in old_columns})
        _append_log(path, [{c: (300 if c == "introspection_from_step" else "new") for c in TRAIN_LOG_COLUMNS}])
        log = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(log.columns) == TRAIN_LOG_COLUMNS, list(log.columns)
        assert log["step"].tolist() == ["7", "new"] and log["introspection_from_step"].tolist() == ["", "300"]
    print("  an old train_log.csv gains the column, and old and new rows line up")


def _results(with_origin=True):
    rows = [{"stage": "base_model", "introspection_fold": "0", "checkpoint_step": 0}]
    for step in (1500, 2500, 3000):
        rows.append({"stage": "decision", "introspection_fold": "0", "checkpoint_step": step})
        rows.append({"stage": "introspection", "introspection_fold": "both", "checkpoint_step": 300,
                     "introspection_from_step": step if with_origin else ""})
    frame = pd.DataFrame(rows).assign(
        run_id="r", report_schema="linear", report_batch="main", block="main",
        recovered_vs_hidden=0.5, recovered_vs_hidden_pearson=0.5, faithfulness=0.4, faithfulness_pearson=0.4,
        chance=0.0, chance_pearson=0.0,
    )
    if not with_origin:
        frame = frame.drop(columns="introspection_from_step")
    return frame


def test_the_plots_place_each_branch():
    x, ticks = _checkpoint_x(_results())
    assert x == [0, 1500, 1500, 2500, 2500, 3000, 3000] and not ticks, (x, ticks)
    x, ticks = _checkpoint_x(_results(with_origin=False))
    assert x == [0, 1500, 3500, 2500, 3500, 3000, 3500] and ticks == {3500: "after\nintrospection"}, (x, ticks)

    log = pd.DataFrame([
        {"run_id": "r", "stage": "decision", "fold": 0, "introspection_from_step": "", "step": s, "loss": 1.0,
         "first_token_accuracy": 0.5, "val_loss": "", "val_first_token_accuracy": ""} for s in range(1, 4)
    ] + [
        {"run_id": "r", "stage": "introspection", "fold": f, "introspection_from_step": o, "step": s, "loss": 0.5,
         "first_token_accuracy": 1.0, "val_loss": "", "val_first_token_accuracy": ""}
        for o in (1, 3) for f in (1, 2) for s in (1, 2)
    ])
    with tempfile.TemporaryDirectory() as d:
        assert plot_checkpoints(_results(), "r", d).exists()
        assert plot_checkpoints(_results(with_origin=False), "r", d).exists()
        assert plot_training(log, "r", d).exists()
        assert plot_training(log.drop(columns="introspection_from_step"), "r", d).exists()
    print("  each branch sits at its decision step, and results without the column still plot")


def test_the_base_model():
    assert load("configs/default.json")["model_hyperparameters"]["baseline_before_training"] is True
    assert _load_with(baseline_before_training=False)["model_hyperparameters"]["baseline_before_training"] is False
    for bad in ("yes", 1, None):
        try:
            _load_with(baseline_before_training=bad)
        except SettingsError as err:
            assert "baseline_before_training" in str(err), str(err)
            continue
        raise AssertionError(f"baseline_before_training {bad!r} was accepted")
    assert stage_label("base_model") == "base_model"
    print("  on by default, off when set false, three bad values refused, tagged base_model_step-0")


class _Echo(torch.nn.Module):
    """A stand-in model that predicts every next token correctly except the first answer token."""

    def forward(self, input_ids, attention_mask, labels):
        logits = torch.full((*input_ids.shape, 10), -5.0)
        logits[:, :-1].scatter_(-1, input_ids[:, 1:, None], 5.0)
        for i in range(labels.shape[0]):
            first = torch.nonzero(labels[i, 1:] != -100).flatten()[0]
            logits[i, first] = 0.0
            logits[i, first, 9] = 5.0          # the first answer token is always predicted wrong
        return type("Out", (), {"loss": torch.tensor(0.5), "logits": logits})()


def test_stage_two_never_reads_the_first_token():
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]])
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits = _Echo()(ids, None, labels).logits
    firsts, correct, valid = _accuracy_counts(logits, labels)
    assert firsts == [False] and (correct, valid) == (3, 4)
    firsts, correct, valid = _accuracy_counts(logits, labels, first_token=False)
    assert firsts == [] and (correct, valid) == (3, 4)
    tok = type("Tok", (), {"pad_token_id": 0})()
    encoded = [([1, 2, 3, 4, 5, 6], [-100, -100, 3, 4, 5, 6])] * 3
    _, first, answer = evaluate_examples(_Echo(), tok, encoded, 2, "cpu", "no", first_token=False)
    assert np.isnan(first) and answer == 0.75, (first, answer)
    _, first, answer = evaluate_examples(_Echo(), tok, encoded, 2, "cpu", "no")
    assert first == 0.0 and answer == 0.75, (first, answer)
    assert "val_answer_token_accuracy" in TRAIN_LOG_COLUMNS
    print("  stage two: first token unread (NaN), answer-token accuracy 0.75 from the other three tokens")


if __name__ == "__main__":
    for test in (test_the_setting, test_branch_names, test_undoing_a_branch_is_exact,
                 test_an_old_train_log_is_migrated, test_the_plots_place_each_branch,
                 test_the_base_model, test_stage_two_never_reads_the_first_token):
        test()
        print(f"ok  {test.__name__}")
    print("all introspection point checks passed")
