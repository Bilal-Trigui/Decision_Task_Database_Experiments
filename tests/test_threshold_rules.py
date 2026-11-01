"""The threshold template and the two experiments built on it.

  1. threshold.py is a template: it cannot be named in a settings file or run as a command.
  2. minimum_threshold.py still draws exactly the latent data/minimum_threshold/ holds.
  3. The template's two cut draws: every integer in cut_range, or only cut_levels.
  4. paper_minimum_threshold.py: Plunkett's weights on both sides of 0 with the largest magnitude
     100, none nearer 0 than 20, the screened weight 0, and one cut at 25, 50 or 75.
  5. The paper schema's four questions: each training answer parses back to what it was built from,
     and a reply outside its range is unreadable, including 0 for the screened attribute.
  6. The paper estimator's screen distances: a perfect report scores 1 on every question, the
     wrong attribute scores 0, so does any report against an all-zeros fit (choices all one
     letter), and the screened weight is never read off the fit.
  7. The paper estimator finds the screened attribute and its level from clean simulated choices.
  8. The fit always names exactly one screen, even from choices with none in them; the old
     penalty's settings are refused, and so is a rule with more than one cut.
  9. screen_fit: True only for the persona's own attribute, and for the cut only at its own level.
  10. min_weight_magnitude: 20 on both threshold rules, null elsewhere, held on every rule that takes
      it, and read as null from a manifest written before it existed.

Run with `python -m tests.test_threshold_rules`.
"""
import hashlib
import random

import numpy as np
import pandas as pd

from src import data as D
from src.config import SettingsError, load_component
from src.estimators.paper_minimum_threshold import PaperMinimumThresholdEstimator
from src.reports.paper_minimum_threshold import PaperMinimumThresholdSchema
from src.rules.build import roll_weights
from src.rules.minimum_threshold import MinimumThresholdRule
from src.rules.paper_minimum_threshold import PaperMinimumThresholdRule
from src.rules.threshold import ThresholdRule
from tests.generate_trials import generate_trials

N = 5


def test_the_template_cannot_run():
    try:
        load_component("rules", "threshold", {})
    except SettingsError as err:
        assert "template" in str(err), str(err)
    else:
        raise AssertionError("the threshold template was built from settings")
    print("  decision_rule.type 'threshold' refused, with the experiments to use instead")


# sha256 of data/a4/instilled_weights.csv as committed in 294d0d2, from that folder's manifest
A4_WEIGHTS = "b30aa7a158cfe592"


def test_minimum_threshold_draws_what_the_a4_folder_holds():
    scenarios = D.load_scenarios("data/minimum_threshold", N)
    _, frame = roll_weights(MinimumThresholdRule(), scenarios, N, seed=1)
    held = pd.read_csv("data/minimum_threshold/instilled_weights.csv")
    assert frame.equals(held), "minimum_threshold no longer draws the shipped folder's latent"
    # with no minimum weight size, the draw is the A4 one, byte for byte
    _, a4 = roll_weights(MinimumThresholdRule(min_weight_magnitude=None), scenarios, N, seed=1)
    assert hashlib.sha256(a4.to_csv(index=False).encode()).hexdigest()[:16] == A4_WEIGHTS
    print(f"  all {len(held)} shipped latents redrawn identically; min_weight_magnitude null redraws A4's file exactly")


def _draw(rule, count=400, seed=0):
    rng = random.Random(seed)
    return np.array([rule.sample_latent(N, rng) for _ in range(count)])


def test_the_two_cut_draws():
    continuous = _draw(ThresholdRule(cut_range=[30, 70]))
    levels = set(continuous[:, N:].max(axis=1))
    assert min(levels) >= 30 and max(levels) <= 70 and len(levels) > 20, sorted(levels)
    discrete = _draw(ThresholdRule(cut_range=[20, 80], cut_levels=[20, 40, 80]))
    assert set(discrete[:, N:].max(axis=1)) == {20, 40, 80}
    try:
        ThresholdRule(cut_range=[30, 70], cut_levels=[25, 50])
    except ValueError as err:
        assert "outside cut_range" in str(err), str(err)
    else:
        raise AssertionError("a cut level outside cut_range was accepted")
    print(f"  continuous draws {len(levels)} levels in 30 to 70; discrete draws only 20, 40 and 80")


def test_the_paper_draw():
    latents = _draw(PaperMinimumThresholdRule())
    weights, cuts = latents[:, :N], latents[:, N:]
    assert (cuts > 0).sum(axis=1).tolist() == [1] * len(latents)
    assert set(cuts.max(axis=1)) == {25, 50, 75}
    rows = np.arange(len(latents))
    screened = weights[rows, cuts.argmax(axis=1)]
    assert np.all(screened == 0.0)
    others = weights[np.arange(N)[None, :] != cuts.argmax(axis=1)[:, None]]
    assert others.min() >= -100 and others.max() <= 100 and (others < 0).any() and (others > 0).any()
    assert np.abs(others).min() >= 20, np.abs(others).min()
    assert np.all(np.abs(weights).max(axis=1) <= 100)
    assert np.mean(np.abs(weights).max(axis=1) == 100) > 0.7     # 100 unless the screened attribute held it
    narrow = _draw(PaperMinimumThresholdRule(weight_range=[0.5, 2.0], min_weight_magnitude=None))[:, :N]
    assert narrow[narrow != 0].min() >= 0.5
    try:
        PaperMinimumThresholdRule(weight_range=[0.5, 2.0])
    except ValueError as err:
        assert "min_weight_magnitude" in str(err), str(err)
    else:
        raise AssertionError("a minimum of 20 was accepted with weights of at most 2")
    print(f"  400 draws: one cut each at 25, 50 or 75, screened weight 0, the others 20 to 100 from 0, "
          f"{(others < 0).mean():.0%} negative; a weight_range still overrides")


def test_the_minimum_weight_magnitude():
    from src.rules.base import check_manifest
    from src.rules.interaction import InteractionRule
    from src.rules.linear import LinearRule, generate_weights

    assert MinimumThresholdRule().params["min_weight_magnitude"] == 20
    assert PaperMinimumThresholdRule().params["min_weight_magnitude"] == 20
    assert ThresholdRule().params["min_weight_magnitude"] is None
    assert LinearRule().params["min_weight_magnitude"] is None
    # null is Plunkett's draw, one call, unchanged
    a, b = random.Random(5), random.Random(5)
    assert all(LinearRule().sample_latent(N, a).tolist() == generate_weights(N, b) for _ in range(50))
    floored = _draw(LinearRule(min_weight_magnitude=30))
    assert np.abs(floored).min() >= 30 and (floored < 0).any()
    interaction = InteractionRule(min_weight_magnitude=20)
    rng = random.Random(1)
    for _ in range(200):
        main = interaction.sample_latent(N, rng)[:N]
        assert np.all((main == 0) | (np.abs(main) >= 20)), main
    for bad in (-1, 100, True, "20"):
        try:
            LinearRule(min_weight_magnitude=bad)
        except ValueError:
            continue
        raise AssertionError(f"min_weight_magnitude {bad!r} was accepted")
    # data/plunkett's manifest predates the setting and still matches a linear rule with no minimum
    check_manifest("data/plunkett", LinearRule(), {})
    try:
        check_manifest("data/plunkett", LinearRule(min_weight_magnitude=20), {})
    except D.DataFormatError:
        pass
    else:
        raise AssertionError("a folder built with no minimum passed for a rule with one")
    print("  20 on both threshold rules, null elsewhere; null is Plunkett's draw; 30 holds on linear and 20 on "
          "interaction; four bad values refused; old manifests read as null")


def test_the_four_questions_round_trip():
    rule, schema = PaperMinimumThresholdRule(), PaperMinimumThresholdSchema(N)
    names = [f"dim_{i}" for i in range(1, N + 1)]
    latent = np.array([-100, 95, 72, 0, -1, 0, 0, 0, 50, 0], float)
    slices = rule.block_slices(N)
    batches = {b.name: b for b in schema.batches(names)}
    assert list(batches) == ["main", "screen_index", "screen_cut", "screen_weight"]
    expected = {"main": '{"dim_1": -100, "dim_2": 95, "dim_3": 72, "dim_4": 0, "dim_5": -1}',
                "screen_index": '{"screened_dimension": 4}', "screen_cut": '{"minimum_level": 0.5}',
                "screen_weight": '{"weight": 0}'}
    for name, batch in batches.items():
        answer = schema.format_answer(batch, latent[slices[batch.block]])
        assert answer == expected[name], (name, answer)
        assert schema.parse(answer, batch) is not None, name
    for name, reply in (("screen_index", '{"screened_dimension": 6}'), ("screen_index", '{"screened_dimension": 2.5}'),
                        ("screen_index", '{"screened_dimension": 0}'), ("screen_cut", '{"minimum_level": 50}'),
                        ("screen_weight", '{"weight": 150}'), ("screen_weight", '{"strength": 1}')):
        assert schema.parse(reply, batches[name]) is None, (name, reply)
    assert "or 0" not in batches["screen_index"].prompt_base
    try:
        schema.format_answer(batches["screen_index"], np.zeros(N))
    except ValueError:
        pass
    else:
        raise AssertionError("a latent with no screen was given a screen_index answer")
    print("  four training answers parse back; an index of 0, 6 or 2.5, a level of 50, a weight of 150 "
          "and a wrong key are unreadable")


def test_main_report_is_the_four_in_one():
    """main_report: one reply that splits into exactly what the four single questions read, trained on too."""
    rule = PaperMinimumThresholdRule()
    schema = PaperMinimumThresholdSchema(N, batches=["main", "screen_index", "screen_cut", "screen_weight", "main_report"])
    names = [f"dim_{i}" for i in range(1, N + 1)]
    latent = np.array([-100, 95, 72, 0, -1, 0, 0, 0, 50, 0], float)
    slices = rule.block_slices(N)
    batches = {b.name: b for b in schema.batches(names)}
    combined = batches["main_report"]
    assert combined.trained and combined.blocks == ("main", "screen_index", "screen_cut", "screen_weight")
    assert all(batches[q].trained and batches[q].blocks == (q,) for q in ("main", "screen_index", "screen_cut", "screen_weight"))
    answer = schema.format_answer(combined, latent)
    assert answer == ('{"dim_1": -100, "dim_2": 95, "dim_3": 72, "dim_4": 0, "dim_5": -1, '
                      '"screened_dimension": 4, "minimum_level": 0.5, "screen_weight": 0}'), answer
    parts = schema.split(combined, schema.parse(answer, combined))
    for q in ("main", "screen_index", "screen_cut", "screen_weight"):
        single = schema.parse(schema.format_answer(batches[q], latent[slices[q]]), batches[q])
        assert np.array_equal(parts[q], single), (q, parts[q], single)
    # one unreadable part makes the whole reply unreadable, and so does a missing or extra key
    for reply in (answer.replace('"screened_dimension": 4', '"screened_dimension": 6'),
                  answer.replace('"minimum_level": 0.5', '"minimum_level": 50'),
                  answer.replace('"screen_weight": 0', '"screen_weight": 150'),
                  answer.replace(', "screen_weight": 0', ''),
                  answer.replace('"screen_weight": 0', '"weight": 0')):
        assert schema.parse(reply, combined) is None, reply
    assert "8 key-value pairs" in combined.prompt_base and "1 dim_1, 2 dim_2" in combined.prompt_base
    print("  main_report's one reply splits into the four single answers exactly; any unreadable part, "
          "missing key or wrong key makes it unreadable; it is not trained on")


def test_single_levels_take_the_most_common_reply():
    schema = PaperMinimumThresholdSchema(N)
    r = lambda *v: [np.array([x], float) for x in v]
    assert schema.aggregate("screen_weight", r(50, -50, 50, 0))[0] == 50
    assert schema.aggregate("screen_weight", r(-50, 50))[0] == -50              # a tie keeps the earliest
    assert schema.aggregate("screen_cut", r(25, 75, 75))[0] == 75
    assert np.array_equal(schema.aggregate("main", [np.array([1., 2.]), np.array([3., 4.])]), [2., 3.])
    votes = [np.eye(N)[i] for i in (3, 3, 1)]
    assert np.argmax(schema.aggregate("screen_index", votes)) == 3            # votes still averaged
    # ten random weights averaged sit near the hidden 0; their most common one does not
    rng = np.random.default_rng(0)
    replies = r(*rng.integers(-100, 101, 10))
    assert abs(np.mean(replies)) < abs(schema.aggregate("screen_weight", replies)[0])
    print("  screen_cut and screen_weight take the most common reply, the earliest on a tie; "
          "main is averaged and screen_index voted as before")


def test_report_prompts_never_list_c():
    rule = PaperMinimumThresholdRule(option_count=3)
    schema = PaperMinimumThresholdSchema(N, batches=["main", "screen_index", "screen_cut", "screen_weight", "main_report"])
    folder = "data/paper_minimum_threshold"
    scenarios = D.load_scenarios(folder, N)[:4]
    latents = D.load_hidden_latents(folder, scenarios, rule, N)[:4]
    examples = D.generate_introspection_examples(scenarios, latents, rule, schema, N, 6)
    assert {e.batch for e in examples} == {"main", "screen_index", "screen_cut", "screen_weight", "main_report"}
    assert len(examples) == 4 * 5
    assert not any(e.user.endswith(D.NEITHER_OPTION) or "Neither option" in e.user for e in examples)
    # main_report's training answer is built from the full latent: it parses, and its four parts
    # are exactly the four single questions' answers for the same persona
    for sc in scenarios:
        batches = {b.name: b for b in schema.batches(sc.names)}
        mine = {e.batch: e.messages[-1]["content"] for e in examples if e.scenario == sc.short_name}
        parts = schema.split(batches["main_report"], schema.parse(mine["main_report"], batches["main_report"]))
        for q in ("main", "screen_index", "screen_cut", "screen_weight"):
            single = schema.parse(mine[q], batches[q])
            assert np.array_equal(parts[q], single), (sc.short_name, q, parts[q], single)
    trial = D.Trial(scenarios[0], random.Random(0))
    assert D.selection_prompt(trial, 3).endswith(D.NEITHER_OPTION), "decision prompts keep C"
    print("  introspection examples show A and B only, main_report's answer is the four in one, decision prompts still list C")


def test_the_screen_distances():
    est, schema = PaperMinimumThresholdEstimator(N), PaperMinimumThresholdSchema(N)
    rule = PaperMinimumThresholdRule()
    names = [f"dim_{i}" for i in range(1, N + 1)]
    latent = np.array([-100, 95, 72, 0, -1, 0, 0, 0, 75, 0], float)
    slices, batches = rule.block_slices(N), {b.name: b for b in schema.batches(names)}
    for name in ("screen_index", "screen_cut", "screen_weight"):
        report = schema.parse(schema.format_answer(batches[name], latent[slices[name]]), batches[name])
        assert est.block_distance(name, report, latent[slices[name]]) == 1.0, name
        assert est.block_distance(name, latent[slices[name]], report) == 1.0, name
    wrong = schema.parse('{"screened_dimension": 2}', batches["screen_index"])
    right = schema.parse('{"screened_dimension": 4}', batches["screen_index"])
    assert est.block_distance("screen_index", wrong, latent[slices["screen_index"]]) == 0.0
    assert est.block_distance("screen_index", right, np.zeros(N)) == 0.0     # an all-zeros fit
    off = schema.parse('{"minimum_level": 0.25}', batches["screen_cut"])
    assert abs(est.block_distance("screen_cut", off, latent[slices["screen_cut"]]) - 0.5) < 1e-12
    assert "screen_weight" not in est.block_slices(), "the fit must not claim to recover one weight's size"
    off_weight = schema.parse('{"weight": 50}', batches["screen_weight"])
    assert abs(est.block_distance("screen_weight", off_weight, latent) - 0.5) < 1e-12
    print("  perfect reports score 1 from either side; wrong attribute 0, and 0 against an all-zeros fit")


def test_the_estimator_finds_the_screen():
    rule, est = PaperMinimumThresholdRule(), PaperMinimumThresholdEstimator(N)
    scenarios = D.load_scenarios("data/paper_minimum_threshold", N)[:30]
    rng = random.Random(3)
    index_hits = level_hits = 0
    for sc in scenarios:
        latent = rule.sample_latent(N, rng)
        trials = generate_trials(rule, latent, sc, 200, rng)
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        recovered = est.fit(A, B, [t.label for t in trials], sc.mins, sc.maxs)
        index_hits += est.block_distance("screen_index", recovered[N:], latent[N:]) == 1.0
        level_hits += est.block_distance("screen_cut", recovered[N:], latent[N:]) == 1.0
    assert index_hits >= 24 and level_hits >= 20, (index_hits, level_hits)
    print(f"  clean choices, 200 trials: screened attribute found {index_hits}/30, its level exactly {level_hits}/30")


def test_the_fit_always_names_one_screen():
    """No "no cut" answer: linear choices, with no screen in them, still come back with exactly one cut."""
    from src.estimators.minimum_threshold import MinimumThresholdEstimator, build
    from src.rules.linear import LinearRule

    rule = LinearRule()
    scenarios = D.load_scenarios("data/minimum_threshold", N)[:5]
    rng = random.Random(4)
    for est in (MinimumThresholdEstimator(N), PaperMinimumThresholdEstimator(N)):
        for sc in scenarios:
            trials = generate_trials(rule, rule.sample_latent(N, rng), sc, 100, rng)
            A = np.array([t.option_A.values for t in trials])
            B = np.array([t.option_B.values for t in trials])
            cuts = est.fit(A, B, [t.label for t in trials], sc.mins, sc.maxs)[N:]
            assert np.count_nonzero(cuts) == 1, (est.name, cuts)
            assert cuts.max() in range(est.grid_step, 100, est.grid_step), (est.name, cuts)
    for old in ("min_gain", "sweeps"):
        try:
            build({"attribute_count": N, old: 1})
        except ValueError as err:
            assert old in str(err), str(err)
        else:
            raise AssertionError(f"model_estimating.{old} was accepted and ignored")
    cfg = {"object": {"attribute_count": N}, "compute": {"backend": "local"},
           "decision_rule": {"type": "minimum_threshold", "active_cuts": 2},
           "model_estimating": {"type": "minimum_threshold"}, "report_schema": {"type": "auto"}}
    from src.config import resolve
    try:
        resolve(cfg)
    except SettingsError as err:
        assert "active_cuts" in str(err), str(err)
    else:
        raise AssertionError("a single-cut fit was paired with a two-cut rule")
    print("  linear choices still get exactly one cut; min_gain, sweeps and active_cuts 2 refused")


def test_the_cut_in_units():
    """screen_cut_units and main_report_units: the minimum in the attribute's own units, scored on its range."""
    rule = PaperMinimumThresholdRule()
    schema = PaperMinimumThresholdSchema(N, batches=["main", "screen_index", "screen_cut_units", "main_report_units"])
    names = [f"dim_{i}" for i in range(1, N + 1)]
    mins, maxs = [0, 0.1, 1, 300, 0], [70, 10, 10, 1200, 100]
    latent = np.array([-100, 95, 72, 0, -1, 0, 0, 0, 50, 0], float)    # dim_4, 300 to 1200, screened at 50
    slices = rule.block_slices(N)
    batches = {b.name: b for b in schema.batches(names, mins, maxs)}
    assert "screen_weight" not in batches and "screen_weight" not in batches["main_report_units"].prompt_base
    single = batches["screen_cut_units"]
    assert single.block == "screen_cut" and single.trained
    answer = schema.format_answer(single, latent[slices["screen_cut"]])
    assert answer == '{"screened_dimension": 4, "minimum_value": 750}', answer
    assert np.allclose(schema.parse(answer, single), [50.0])
    combined = batches["main_report_units"]
    assert combined.blocks == ("main", "screen_index", "screen_cut")
    answer = schema.format_answer(combined, latent)
    assert answer == ('{"dim_1": -100, "dim_2": 95, "dim_3": 72, "dim_4": 0, "dim_5": -1, '
                      '"screened_dimension": 4, "minimum_value": 750}'), answer
    parts = schema.split(combined, schema.parse(answer, combined))
    assert np.array_equal(parts["screen_index"], [0, 0, 0, 1, 0]) and np.allclose(parts["screen_cut"], [50.0])
    # the value is read on the range of the attribute the reply names, right or wrong
    assert np.allclose(schema.parse('{"screened_dimension": 1, "minimum_value": 17.5}', single), [25.0])
    for reply in ('{"screened_dimension": 4, "minimum_value": 1300}', '{"screened_dimension": 4, "minimum_value": 0.5}',
                  '{"screened_dimension": 6, "minimum_value": 750}', '{"minimum_level": 0.5}',
                  '{"screened_dimension": 4, "minimum_value": 750, "weight": 0}'):
        assert schema.parse(reply, single) is None, reply
    try:
        schema.format_answer(schema.batches(names)[2], latent[slices["screen_cut"]])
    except ValueError:
        pass
    else:
        raise AssertionError("a units question was answered without the attribute ranges")
    print("  a minimum of 750 on 300 to 1200 reads as 50; out of range, a bad index, a wrong or extra key unreadable")


def test_screen_fit():
    from src.evaluate import screen_fit

    own = [0, 0, 0, 50, 0]
    same = screen_fit([0, 0, 0, 50, 0], own)
    assert same["estimator_attribute_matches_absolute"] and same["estimator_cut_matches_absolute"], same
    level = screen_fit([0, 0, 0, 75, 0], own)
    assert level["estimator_attribute_matches_absolute"] and not level["estimator_cut_matches_absolute"], level
    assert level["cut_error"] == 25, level
    other = screen_fit([0, 50, 0, 0, 0], own)
    assert not other["estimator_attribute_matches_absolute"] and not other["estimator_cut_matches_absolute"], other
    empty = screen_fit(np.zeros(N), own)
    assert empty["estimator_attribute"] is None and not empty["estimator_attribute_matches_absolute"], empty
    print("  own attribute and level True; wrong level, wrong attribute and an all-zeros fit False")


if __name__ == "__main__":
    for test in (test_the_template_cannot_run, test_minimum_threshold_draws_what_the_a4_folder_holds,
                 test_the_two_cut_draws, test_the_paper_draw, test_the_four_questions_round_trip,
                 test_main_report_is_the_four_in_one, test_single_levels_take_the_most_common_reply,
                 test_report_prompts_never_list_c,
                 test_the_screen_distances, test_the_estimator_finds_the_screen,
                 test_the_fit_always_names_one_screen, test_the_cut_in_units, test_screen_fit,
                 test_the_minimum_weight_magnitude):
        test()
        print(f"ok  {test.__name__}")
    print("all threshold rule checks passed")
