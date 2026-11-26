"""flip_ab (ask_only and evaluate_saved; model_estimating.flip_ab in their settings.json): an evaluation's trials shown with their two options in each other's place.

The same pairs an unflipped evaluation shows, A's values under B and B's under A. It tells a model that answers by the
letter from one that answers by the options: the first keeps its letter, the second moves with the option.

  1. A mirrored trial is the same pair, swapped, and the trial it came from is untouched.
  2. The rule's right answer moves with the options (A <-> B, C stays C).
  3. The setting: off unless set, true or false only, set by the two tools, and no fallback (settings files unchanged).
  4. The decision step under the setting, and flip_check: a letter model keeps its letter, an option model follows.
  5. ask_only's balanced trials keep their draws and kinds.

No model is loaded: the answers are stubbed.

    python -m tests.test_flip_ab
"""
import copy
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import ask_only as A
from src import config as C
from src import data as D
from src import evaluate as E
from src import evaluate_saved as ES
from src import persona_rule
from tests import config_path

ROOT = Path(__file__).resolve().parents[1]
SWAP = {"A": "B", "B": "A", "C": "C"}


def _setup(flip=False, trials=200):
    cfg = C.load(str(config_path("shuffled_24gb")))
    cfg["model_estimating"].update(estimation_trials_per_persona=trials, flip_ab=flip)
    C.validate(cfg)
    comps = C.resolve(cfg)
    return cfg, comps, D.load_data(cfg, comps.rule)


def test_a_mirrored_trial_is_the_same_pair_swapped():
    cfg, comps, data = _setup()
    plain = D.fresh_trials(data.personas, 20, 5)[0]
    before = [t.generate_choice(3) for t in plain]
    flipped = D.fresh_trials(data.personas, 20, 5, flip_ab=True)[0]
    for t, m, text in zip(plain, flipped, before):
        assert m.option_A.values.tolist() == t.option_B.values.tolist()
        assert m.option_B.values.tolist() == t.option_A.values.tolist()
        assert (m.option_A.letter, m.option_B.letter) == ("A", "B")
        # the prompt: the question, then B's block under "A:", then A's under "B:", then C
        a_block, b_block = t.option_A.description[2:], t.option_B.description[2:]
        assert m.generate_choice(3) == f"{t.scenario.question}\nA:{b_block}\n\nB:{a_block}\n\n{D.NEITHER_OPTION}"
        assert t.generate_choice(3) == text, "mirroring changed the trial it was given"
        assert D.mirrored(m).generate_choice(3) == text, "mirroring twice is not the trial again"
    assert [t.generate_choice(3) for t in D.fresh_trials(data.personas, 20, 5)[0]] == before
    print("  20 trials: values and prompt swapped, the original untouched, twice is the original")


def test_the_right_answer_moves_with_the_options():
    cfg, comps, data = _setup()
    plain = D.fresh_trials(data.personas, 1000, 5)[0]
    labels = [comps.rule.label(data.latents[0], t) for t in plain]
    moved = [comps.rule.label(data.latents[0], D.mirrored(t)) for t in plain]
    assert set(labels) == {"A", "B", "C"}
    assert moved == [SWAP[l] for l in labels]
    print(f"  1000 trials: every label moved with its option ({labels.count('C')} stayed C)")


def test_the_setting():
    cfg, _, _ = _setup()
    # no fallback: a settings file does not gain a field, so every flat file in configs/ still matches the fallbacks
    assert "flip_ab" not in C.apply_defaults({})["model_estimating"]
    plain = C.load(str(config_path("shuffled_24gb")))
    assert "flip_ab" not in plain["model_estimating"] and E.flip_ab(plain) is False
    for bad in ("yes", 1, None):
        wrong = copy.deepcopy(cfg)
        wrong["model_estimating"]["flip_ab"] = bad
        try:
            C.validate(wrong)
        except C.SettingsError as err:
            assert "flip_ab" in str(err)
            continue
        raise AssertionError(f"flip_ab {bad!r} was accepted")
    # through the two tools, on a saved run's settings: the folder each writes records it
    run = ROOT / "results" / "20261122-214856_paper_minimum_threshold_Qwen3-8B_shuffled"
    if (run / "settings.json").exists():
        assert E.flip_ab(A.run_settings(run)) is False
        assert E.flip_ab(A.run_settings(run, flip_ab=True)) is True
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / run.name).mkdir()
            shutil.copy(run / "settings.json", Path(tmp) / run.name / "settings.json")
            for tool, prepare in (("_evaluate_only", lambda flip: ES.prepare(run.name, "t", results_dir=tmp, flip_ab=flip)),
                                  ("_ask_only", lambda flip: A.prepare(run.name, name="t", results_dir=tmp, flip_ab=flip))):
                for flip in (None, True):
                    _, out_dir, made = prepare(flip)
                    saved = json.loads((out_dir / "settings.json").read_text(encoding="utf-8"))
                    assert E.flip_ab(made) is bool(flip) and saved[tool]["flip_ab"] is bool(flip), (tool, flip)
                    assert saved["model_estimating"].get("flip_ab", False) is bool(flip), (tool, flip)
    print("  off unless set; three bad values refused; set and recorded by ask_only and evaluate_saved; no fallback")


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[1]["content"]


def _values(text):
    """Every attribute's value under A and under B, read off a decision prompt."""
    a, b = text.split("\nA:\n")[1].split("\n\nB:\n")
    read = lambda block: tuple(float(v) for v in re.findall(r": (-?[\d.]+) ", block.split("\n\nC:")[0] + " "))
    return read(a), read(b)


def _answers(model):
    """A stand-in for evaluate.choice_logits: `model` maps a prompt to the letter answered."""
    def choice_logits(_model, tok, texts, batch_size, device, letters=("A", "B")):
        logits = np.zeros((len(texts), len(letters)))
        for i, text in enumerate(texts):
            logits[i, letters.index(model(text))] = 5.0
        return logits, np.zeros(len(texts), dtype=int), [0]
    return choice_logits


def _decide(model, flip, out_dir):
    cfg, comps, data = _setup(flip)
    real = E.choice_logits
    E.choice_logits = _answers(model)
    try:
        return E._decisions(None, _Tok(), cfg, comps.rule, data, "cpu", [0], "decision_step-1", Path(out_dir))
    finally:
        E.choice_logits = real


def test_a_letter_model_keeps_its_letter_and_an_option_model_follows():
    always_a = lambda text: "A"
    # by the options alone: the one whose values come first in order (two options are never equal on all five)
    by_values = lambda text: "A" if _values(text)[0] > _values(text)[1] else "B"
    for name, model, same, followed in (("always A", always_a, 1.0, 0.0), ("by the options' values", by_values, 0.0, 1.0)):
        with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as out:
            before, after = _decide(model, False, run), _decide(model, True, out)
            # the flipped file describes the trials as shown: the run's B values under A, and its own labels
            assert (before["B_attribute_1"].values == after["A_attribute_1"].values).all()
            assert after["label"].tolist() == [SWAP[l] for l in before["label"]]
            check = ES.flip_check(run, out, 1)
            assert check["mirrored"] is True and check["decisions"] == 200
            assert (check["same_letter"], check["followed_option"]) == (same, followed), (name, check)
            # an unflipped evaluation of the same trials is not mistaken for a flipped one
            with tempfile.TemporaryDirectory() as again:
                _decide(model, False, again)
                assert ES.flip_check(run, again, 1)["mirrored"] is False
            assert ES.flip_check(run, out, 2) is None
        print(f"  {name}: same letter {same:.0%}, followed the option {followed:.0%}")


def test_balanced_trials_keep_their_draws_and_kinds():
    cfg, comps, data = _setup()
    folder = Path(cfg["model_training"]["data_dir"])
    folder = folder if folder.is_absolute() else ROOT / folder
    table = pd.read_csv(folder / "constraint_table.csv")
    active = table[table["active"].astype(str).str.lower() == "true"].iloc[0]
    screen = persona_rule.Screen(int(active["position"]), active["attribute"], active["units"], float(active["cut_value"]),
                                 float(active["cut_percent"]), "minimum", float(active["range_min"]), float(active["range_max"]))
    plain = A.balanced_trials(data.personas[0], screen, 10)
    flipped = A.balanced_trials(data.personas[0], screen, 10, flip_ab=True)
    assert [(k, i) for k, _, i in plain] == [(k, i) for k, _, i in flipped]
    for (kind, t, _), (_, m, _) in zip(plain, flipped):
        assert m.option_A.values.tolist() == t.option_B.values.tolist() and m.option_B.values.tolist() == t.option_A.values.tolist()
        assert A.trial_kind(m, screen) == kind
    print("  30 balanced trials: the same draws and kinds, each pair swapped")


if __name__ == "__main__":
    for test in (test_a_mirrored_trial_is_the_same_pair_swapped, test_the_right_answer_moves_with_the_options,
                 test_the_setting, test_a_letter_model_keeps_its_letter_and_an_option_model_follows,
                 test_balanced_trials_keep_their_draws_and_kinds):
        test()
        print(f"ok  {test.__name__}")
    print("all flip_ab checks passed")
