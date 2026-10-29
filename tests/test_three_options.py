"""Three options: A, B, or C ("neither") when both options fail the screen.

decision_rule.option_count is 2 (Plunkett's task) by default and 3 on the two threshold experiments.

  1. The rule: C exactly when both options fail the screen; with 2 options those trials go back to
     the weighted sum, and every other trial is labelled the same either way.
  2. neither_share: every persona's training and validation trials hold exactly that share of C,
     while the two-option draw is Plunkett's, untouched.
  3. The prompt: Plunkett's instruction verbatim for 2 options; for 3, C in the instruction and
     listed after the options.
  4. The checks: settings and data folder must agree on the option count, from the manifest and
     from the stored examples themselves; a rule that cannot reject both options refuses 3, and
     a share of C without a C is refused.
  5. The estimator, an if statement: from three-option choices with lapses mixed in, it finds the
     screened attribute and its level (25, 50 or 75) for every persona; a weighted-sum estimator
     leaves the C trials out rather than reading them as B.

Run with `python -m tests.test_three_options`.
"""
import copy
import json
import random
import tempfile
from pathlib import Path

import numpy as np

from src import data as D
from src.config import SettingsError, apply_defaults, load_component, validate
from src.configs.test import TEST
from src.rules.base import check_manifest

N = 5


def _personas(count=20):
    return D.load_scenarios("data/plunkett", N)[:count]


def _latents(rule, count=20, seed=1):
    rng = random.Random(seed)
    return [rule.sample_latent(N, rng) for _ in range(count)]


def test_c_exactly_when_both_fail():
    three = load_component("rules", "paper_minimum_threshold", {})
    two = load_component("rules", "paper_minimum_threshold", {"option_count": 2})
    assert three.letters == ("A", "B", "C") and two.letters == ("A", "B")
    assert two.params["neither_share"] is None, "a default share of C must drop when there is no C"
    personas, latents = _personas(), _latents(three)
    counted = {"C": 0, "other": 0}
    for k, sc in enumerate(personas):
        for t in D.fresh_trials([sc], 100, k)[0]:
            both_fail = not three.passes(latents[k], t.option_A.values, sc.mins, sc.maxs) and \
                not three.passes(latents[k], t.option_B.values, sc.mins, sc.maxs)
            label3, label2 = three.label(latents[k], t), two.label(latents[k], t)
            assert (label3 == "C") == both_fail, (k, label3)
            assert label2 in ("A", "B")
            if not both_fail:
                assert label3 == label2, (k, label3, label2)
            counted["C" if both_fail else "other"] += 1
    print(f"  C on {counted['C']} of {sum(counted.values())} trials, all and only where both options fail")


def test_the_share_of_c():
    rule = load_component("rules", "paper_minimum_threshold", {})
    personas, latents = _personas(), _latents(rule)
    train, val = D.generate_decision_trials(personas, latents, rule, 50, 10, 2)
    for trials, total in ((train, 50), (val, 10)):
        for persona in trials:
            assert len(persona) == total
            assert sum(t.label == "C" for t in persona) == round(0.2 * total), [t.label for t in persona]
    # Plunkett's two-option draw is unchanged: the same stream, trial for trial
    linear = load_component("rules", "linear", {})
    a, _ = D.generate_decision_trials(personas[:3], _latents(linear, 3), linear, 5, 2, 2)
    rng = random.Random(2)
    by_hand = [D.Trial(sc, rng) for sc in personas[:1] for _ in range(7)][:5]
    assert all(np.array_equal(x.option_A.values, y.option_A.values) for x, y in zip(a[0], by_hand))
    print("  10 of 50 training and 2 of 10 validation trials answered C, for every persona; two options untouched")


def test_the_prompt():
    t = D.fresh_trials(_personas(1), 1, 0)[0][0]
    assert D.selection_prompt(t, 2) == D.SELECTION_PROMPT_BASE + t.generate_choice()
    three = D.selection_prompt(t, 3)
    assert '"C" if neither option is acceptable' in three and three.endswith("\n\nC:\nNeither option")
    assert D.decision_example(t, 3).user == three
    print("  two options: Plunkett's prompt verbatim; three: C in the instruction and after the options")


def _cfg(rule):
    cfg = copy.deepcopy(TEST)
    cfg["decision_rule"] = rule
    return validate(apply_defaults(cfg))


def _refused(fn, *words):
    try:
        fn()
    except (SettingsError, ValueError) as err:
        assert all(w in str(err) for w in words), (words, str(err))
        return str(err)
    raise AssertionError(f"accepted, expected a refusal naming {words}")


def test_settings_and_data_must_agree():
    two = load_component("rules", "paper_minimum_threshold", {"option_count": 2})
    # the manifest: data/paper_minimum_threshold is built with three options
    _refused(lambda: check_manifest("data/paper_minimum_threshold", two, None), "option_count is 2", "built with 3")
    # an old manifest, written before option_count existed, reads as two options
    linear = load_component("rules", "linear", {})
    check_manifest("data/plunkett", linear, None)
    _refused(lambda: check_manifest("data/plunkett", load_component("rules", "minimum_threshold", {}), None), "option_count")
    # the stored examples themselves, for a folder with no manifest
    with tempfile.TemporaryDirectory() as d:
        for name in ("candidate_scenarios.json", "scenarios.csv", "roles.csv", "instilled_weights.csv",
                     "instill_100_prefs.jsonl", "instill_100_prefs_val.jsonl"):
            Path(d, name).write_bytes(Path("data/paper_minimum_threshold", name).read_bytes())
        cfg = _cfg({"type": "paper_minimum_threshold", "option_count": 2})
        cfg["model_training"]["data_dir"] = d
        _refused(lambda: D.load_data(cfg, two), "three-option question", "option_count is 2")
    # settings: 2 or 3 only; 3 needs a rule that can reject both; a share needs a C
    _refused(lambda: _cfg({"type": "linear", "option_count": 4}), "option_count must be 2")
    _refused(lambda: load_component("rules", "linear", {"option_count": 3}), "can only label 2")
    _refused(lambda: load_component("rules", "interaction", {"option_count": 3}), "can only label 2")
    _refused(lambda: load_component("rules", "minimum_threshold", {"option_count": 2, "neither_share": 0.2}), "needs option_count 3")
    _refused(lambda: load_component("rules", "minimum_threshold", {"neither_share": 1.0}), "between 0 and 1")
    print("  manifest, stored examples and settings each refuse a mismatched option count")


def test_the_estimator_is_an_if_statement():
    rule = load_component("rules", "paper_minimum_threshold", {})
    est = load_component("estimators", "paper_minimum_threshold", {"attribute_count": N, "option_count": 3})
    personas, latents = _personas(), _latents(rule)
    rng = np.random.default_rng(0)
    found = 0
    for k, sc in enumerate(personas):
        trials = D.fresh_trials([sc], 300, 5 + k)[0]
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        # a model that follows the rule but lapses on one choice in twenty
        choices = [rule.label(latents[k], t) if rng.random() > 0.05 else rule.letters[rng.integers(3)] for t in trials]
        fit = est.fit(A, B, choices, sc.mins, sc.maxs)
        hidden, fitted = latents[k][N:], fit[N:]
        found += int(np.array_equal(hidden, fitted))
    assert found == len(personas), f"attribute and level right for {found} of {len(personas)}"
    # a weighted-sum estimator reads C as no preference, not as B
    linear = load_component("estimators", "linear", {"attribute_count": N})
    A, B = np.eye(N)[:2] * 10, np.eye(N)[2:4] * 10
    mins, maxs = np.zeros(N), np.full(N, 10.0)
    assert np.array_equal(linear.fit(np.vstack([A, A]), np.vstack([B, B]), ["A", "B", "C", "C"], mins, maxs),
                          linear.fit(A, B, ["A", "B"], mins, maxs))
    print(f"  screened attribute and level (25, 50 or 75) found for {found}/{len(personas)} personas with 5% lapses; "
          "C left out of the linear fit")


if __name__ == "__main__":
    for test in (test_c_exactly_when_both_fail, test_the_share_of_c, test_the_prompt,
                 test_settings_and_data_must_agree, test_the_estimator_is_an_if_statement):
        test()
        print(f"ok  {test.__name__}")
    print("all three option checks passed")
