"""The paper maximum threshold experiment: the rule, estimator, questions, data folder and settings.

  1. Above the cut fails: both above is C, one above is the other, neither above is the weighted sum.
  2. The rule is the minimum rule's exact mirror: reflect the screened attribute, and the maximum
     rule at cut c gives the answer the minimum rule gives at cut 100 - c, on every trial.
  3. The estimator finds the screened attribute and level from the maximum rule's choices, lapses
     included, and the minimum estimator reading the same choices does not.
  4. Every question asks about a maximum, the wording of main aside, and a training answer parses back.
  5. data/one_persona_maximum_threshold/: every stored label is the maximum rule's, the quotas are
     exact, and it is the minimum folder's persona, weights and cut.
  6. The settings file resolves to the maximum rule, estimator and schema, and differs from the
     minimum settings only in those names, the data folder, the training length and the checkpoint spacing.
  7. The whole collection and scoring path with a scripted perfect model scores 1 on every question.

Run with `python -m tests.test_maximum_threshold`.
"""
import copy
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from src import data as D
from src import evaluate as E
from src.config import load, load_component, read_settings, resolve
from src.estimators.paper_maximum_threshold import PaperMaximumThresholdEstimator
from src.reports.paper_maximum_threshold import PaperMaximumThresholdSchema
from src.rules.base import check_manifest
from src.rules.paper_maximum_threshold import PaperMaximumThresholdRule
from src.rules.paper_minimum_threshold import PaperMinimumThresholdRule

N = 5
SETTINGS = "configs/one_persona_maximum_threshold.json"
DATA = Path("data/one_persona_maximum_threshold")


def _persona():
    return D.load_scenarios(str(DATA), N)[0]


def _trial(scenario, screened, a, b, others=(0.5, 0.5, 0.5, 0.5, 0.5)):
    """A trial whose options sit at fractions of range: others everywhere, a and b on the screened attribute."""
    t = D.fresh_trials([scenario], 1, 0)[0][0]
    lows, highs = np.array(scenario.mins), np.array(scenario.maxs)
    for option, value in ((t.option_A, a), (t.option_B, b)):
        fractions = np.array(others, float)
        fractions[screened] = value
        _set_values(option, lows + fractions * (highs - lows))
    return t


def _set_values(option, values):
    for attribute, value in zip(option.attributes, values):
        attribute["value"] = float(value)


def test_above_the_cut_fails():
    sc, rule = _persona(), PaperMaximumThresholdRule()
    latent = np.array([0, 50, 0, 0, 0, 50, 0, 0, 0, 0], float)     # screens attribute 1 at 50, weight on attribute 2
    assert rule.label(latent, _trial(sc, 0, 0.8, 0.9)) == "C"
    assert rule.label(latent, _trial(sc, 0, 0.2, 0.9)) == "A"
    assert rule.label(latent, _trial(sc, 0, 0.9, 0.2)) == "B"
    assert rule.label(latent, _trial(sc, 0, 0.5, 0.9)) == "A", "a value exactly at the cut passes"
    both_pass = _trial(sc, 0, 0.2, 0.3)
    both_pass.option_A.attributes[1]["value"] = float(sc.maxs[1])       # the weight now decides: A is higher on attribute 2
    assert rule.label(latent, both_pass) == "A"
    both_pass.option_A.attributes[1]["value"] = float(sc.mins[1])
    assert rule.label(latent, both_pass) == "B"
    # and an unscreened attribute is never constrained, however high
    free = _trial(sc, 0, 0.2, 0.3, others=(1, 1, 1, 1, 1))
    assert rule.passes(latent, free.option_A.values, sc.mins, sc.maxs)
    print("  C when both above, the other when one above, weights when neither, the cut itself passes")


def test_the_rule_mirrors_the_minimum_rule():
    sc = _persona()
    high, low = PaperMaximumThresholdRule(), PaperMinimumThresholdRule()
    for level in (25, 50, 75):
        latent = np.array([0, -83, -38, -100, 31, level, 0, 0, 0, 0], float)
        mirror = latent.copy()
        mirror[N] = 100 - level
        counts = Counter()
        for t in D.fresh_trials([sc], 3000, 11)[0]:
            reflected = copy.deepcopy(t)
            for option in (reflected.option_A, reflected.option_B):
                option.attributes[0]["value"] = float(sc.mins[0] + sc.maxs[0] - option.attributes[0]["value"])
            answer = low.label(latent, t)
            assert high.label(mirror, reflected) == answer, (level, t.option_A.values, t.option_B.values)
            counts[answer] += 1
        assert counts["C"] > 0 and counts["A"] > 0 and counts["B"] > 0, counts
    print("  maximum at 100 - c on reflected values answers as minimum at c, on 9000 trials")


def test_the_estimator_finds_a_maximum():
    rule = PaperMaximumThresholdRule()
    maximum = load_component("estimators", "paper_maximum_threshold", {"attribute_count": N, "option_count": 3})
    minimum = load_component("estimators", "paper_minimum_threshold", {"attribute_count": N, "option_count": 3})
    assert isinstance(maximum, PaperMaximumThresholdEstimator)
    personas = D.load_scenarios(str(DATA), N)[:20]
    py, rng = random.Random(1), np.random.default_rng(0)
    found = fooled = 0
    for k, sc in enumerate(personas):
        latent = rule.sample_latent(N, py)
        trials = D.fresh_trials([sc], 300, 5 + k)[0]
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        choices = [rule.label(latent, t) if rng.random() > 0.05 else rule.letters[rng.integers(3)] for t in trials]
        found += int(np.array_equal(latent[N:], maximum.fit(A, B, choices, sc.mins, sc.maxs)[N:]))
        fooled += int(np.array_equal(latent[N:], minimum.fit(A, B, choices, sc.mins, sc.maxs)[N:]))
    assert found == len(personas), f"attribute and level right for {found} of {len(personas)}"
    assert fooled < len(personas) / 2, f"the minimum estimator also found {fooled} of {len(personas)}"
    print(f"  maximum estimator right for {found}/{len(personas)} with 5% lapses; minimum estimator {fooled}/{len(personas)}")


def test_the_questions_ask_about_a_maximum():
    sc = _persona()
    root = Path("questions/paper_maximum_threshold")
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    # every maximum question mirrors a minimum one; these minimum questions (2026-11-12) have no maximum
    # version yet: the trial-first copies, the plain no-trial weights, and the cut probe (whose training
    # answer is written for a minimum)
    minimum_only = {"cut_probe", "main_no_trial"} | {f"{q}_trial_first" for q in
                                                    ("main", "screen_index", "screen_cut_units", "screen_cut",
                                                     "main_report_units")}
    # the reworded cut questions (2026-11-12) say "under the cutoff", written for a minimum
    minimum_only |= {"cutoff_units", "cutoff_percent", "cutoff_units_no_trial", "cutoff_percent_no_trial"}
    minimum = sorted(p.name for p in Path("questions/paper_minimum_threshold").iterdir() if p.is_dir())
    assert folders == sorted(set(minimum) - minimum_only), sorted(set(minimum) ^ set(folders))
    for name in folders:
        text = (root / name / "prompt.txt").read_text(encoding="utf-8") + (root / name / "question.json").read_text(encoding="utf-8")
        # "at or below that maximum" is a maximum's own wording; a bare "below" is a minimum's ("fell below it")
        assert "minimum" not in text.lower() and "below" not in text.lower().replace("at or below", ""), name
        assert name == "main" or "maximum" in text.lower(), name
    schema = PaperMaximumThresholdSchema(N, batches=folders)
    assert schema.root == root.resolve() or schema.root.samefile(root)
    latent = np.array([0, -83, -38, -100, 31, 50, 0, 0, 0, 0], float)
    batches = {b.name: b for b in schema.batches_for(sc)}
    units = batches["screen_cut_units"]
    answer = schema.format_answer(units, latent[N:])
    assert json.loads(answer) == {"screened_dimension": 1, "maximum_value": 35}, answer
    assert np.allclose(schema.parse(answer, units), [50.0])
    assert schema.parse('{"screened_dimension": 1, "minimum_value": 35}', units) is None, "the old key is not read"
    both = batches["main_report_units"]
    parsed = schema.split(both, schema.parse(schema.format_answer(both, latent), both))
    assert np.allclose(parsed["main"], latent[:N]) and np.allclose(parsed["screen_cut"], [50.0])
    print(f"  {len(folders)} question folders ask about a maximum; training answers parse back")


def test_the_data_folder():
    cfg = load(SETTINGS)
    comps = resolve(cfg)
    check_manifest(cfg["model_training"]["data_dir"], comps.rule, cfg)
    data = D.load_data(cfg, comps.rule)
    assert len(data.personas) == 1
    latent = data.latents[0]
    assert latent.tolist() == [0, -83, -38, -100, 31, 50, 0, 0, 0, 0], latent
    for name in ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv"):
        assert (DATA / name).read_bytes() == (Path("data/one_persona_minimum_threshold") / name).read_bytes(), name
    sc = data.personas[0]
    low = PaperMinimumThresholdRule(**comps.rule.params)
    # the stored answers are the maximum rule's, recomputed from each prompt's own numbers
    check = _stored_trials(sc)
    assert len(check) == 5000, len(check)
    counts = Counter()
    differ = 0
    for values_a, values_b, answer in check:
        t = _trial_from(sc, values_a, values_b)
        assert comps.rule.label(latent, t) == answer, (values_a, values_b, answer)
        a, b = comps.rule.passes(latent, values_a, sc.mins, sc.maxs), comps.rule.passes(latent, values_b, sc.mins, sc.maxs)
        counts["C" if not (a or b) else "weights" if a and b else "cut"] += 1
        differ += int(low.label(latent, t) != answer)
    assert counts == {"weights": 3750, "C": 625, "cut": 625}, counts
    assert differ > 2500, f"only {differ} of 5000 answers differ from the minimum rule's"
    print(f"  5000 stored answers are the maximum rule's: {dict(counts)}; {differ} differ from the minimum rule's answer")


def _stored_trials(scenario):
    """(option A's values, option B's values, answer) for each training trial, read from the prompt text."""
    rows = []
    path = DATA / "instill_1_prefs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        messages = json.loads(line)["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        answer = next(m["content"] for m in messages if m["role"] == "assistant").strip()
        rows.append((*_option_values(user, scenario), answer))
    return rows


def _option_values(prompt, scenario):
    """The two options' attribute values from a decision prompt, in attribute order."""
    a_part = prompt.split("A:\n", 1)[1].split("B:\n", 1)[0]
    b_part = prompt.split("B:\n", 1)[1].split("C:\n", 1)[0]

    def read(part):
        values = []
        for attribute, line in zip(scenario.attributes, [l for l in part.splitlines() if l.strip()]):
            assert attribute["name"] in line, (attribute["name"], line)
            values.append(float(line.split(":", 1)[1].strip().split()[0].replace(",", "")))
        return values
    return read(a_part), read(b_part)


def _trial_from(scenario, values_a, values_b):
    t = D.fresh_trials([scenario], 1, 0)[0][0]
    _set_values(t.option_A, values_a)
    _set_values(t.option_B, values_b)
    return t


def test_the_settings():
    high, _ = read_settings(SETTINGS)
    low, _ = read_settings("configs/one_persona_minimum_threshold.json")
    differences = []

    def walk(a, b, path=()):
        for key in sorted(set(a) | set(b)):
            if key in ("_comment", "extends"):
                continue
            x, y = a.get(key), b.get(key)
            if isinstance(x, dict) and isinstance(y, dict):
                walk(x, y, path + (key,))
            elif x != y:
                differences.append(".".join(path + (key,)))
    walk(high, low)
    assert sorted(differences) == ["decision_rule.type", "model_estimating.type", "model_hyperparameters.checkpoint_every",
                                   "model_hyperparameters.training_steps", "model_training.data_dir"], differences
    assert high["model_hyperparameters"]["training_steps"] == 2000 and high["model_hyperparameters"]["checkpoint_every"] == 50
    assert "extends" not in high
    mh = high["model_hyperparameters"]
    assert not mh.get("start_from") and not mh["introspection_training"], "trained from scratch, decisions only"
    comps = resolve(load(SETTINGS))
    assert comps.rule.name == comps.estimator.name == comps.schemas[0].name == "paper_maximum_threshold"
    print("  resolves to the maximum rule, estimator and schema; five settings differ from the minimum file")


def test_the_scoring_path():
    cfg = load(SETTINGS)
    cfg["model_estimating"].update({"estimation_trials_per_persona": 300, "chance_draws": 5})
    cfg["report_schema"]["samples_per_report"] = 2
    comps = resolve(cfg)
    rule, schema = comps.rule, comps.schemas[0]
    data = D.load_data(cfg, rule)
    sc, latent = data.personas[0], data.latents[0]
    trials = D.fresh_trials(data.personas, 300, cfg["model_training"]["seeds"]["verification"])[0]
    labels = [rule.label(latent, t) for t in trials]
    slices = rule.block_slices(N)

    def choice_logits(model, tok, texts, batch_size, device, letters=("A", "B")):
        assert len(texts) == len(labels)
        one_hot = np.array([[float(lab == letter) for letter in letters] for lab in labels])
        return one_hot, [0] * len(labels), tuple(range(len(letters)))

    replies = []
    for batch in schema.batches_for(sc):
        values = latent if batch.parts else latent[slices[batch.block]]
        replies.append([schema.format_answer(batch, values)] * 2)
    calls = iter(replies)

    saved = (E.choice_logits, E.generate_texts, E.chat_prompt)
    E.choice_logits, E.generate_texts = choice_logits, lambda *a, **k: next(calls)
    E.chat_prompt = lambda tok, content: content
    try:
        with tempfile.TemporaryDirectory() as out:
            detail = E.collect(None, None, cfg, comps, data, "cpu", [0], "t", out)
            rows = E.summarize([detail], cfg, comps, data, "t", "decision", 0, 1)
    finally:
        E.choice_logits, E.generate_texts, E.chat_prompt = saved
    asked = {r["report_batch"] for r in rows if r["report_batch"]}
    assert asked == {"main", "screen_index", "screen_cut_units", "main_report_units"}, asked
    for row in rows:
        if not row["report_batch"]:
            continue
        name = (row["report_batch"], row["block"])
        assert row["decision_accuracy"] == 1.0 and row["parse_rate"] == 1.0, name
        assert abs(row["hidden_vs_reported"] - 1.0) < 1e-9, (name, row["hidden_vs_reported"])
        if row["block"] in ("screen_index", "screen_cut"):
            assert row["recovered_vs_hidden"] == 1.0 and row["faithfulness"] == 1.0, (name, row["faithfulness"])
    print(f"  a perfect reporter scores 1 on all {len(rows)} rows; the fit finds caffeine_content at 50")


if __name__ == "__main__":
    for test in (test_above_the_cut_fails, test_the_rule_mirrors_the_minimum_rule, test_the_estimator_finds_a_maximum,
                 test_the_questions_ask_about_a_maximum, test_the_data_folder, test_the_settings, test_the_scoring_path):
        test()
        print(f"ok  {test.__name__}")
    print("all maximum threshold checks passed")
