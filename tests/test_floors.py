"""The bars faithfulness is read against, beside chance.

  1. chance_p95: the 95th percentile of the chance draws' means, never below the mean.
  2. floor_shuffled: every report scored against another persona's recovered rule. The pairing
     never leaves a persona with itself, and one seed gives one result.
  3. const_best: the best single answer given to every persona, for screen_index and screen_cut.
  4. What the floors tell apart: a report that is the recovered rule beats all of them; a report
     that says the same thing for every persona scores exactly floor_shuffled, even when it beats
     chance, which is the case floor_shuffled exists for.
  5. The new columns reach the results row and faithfulness_view.

Run with `python -m tests.test_floors`.
"""
import random

import numpy as np

from src.chance import CHANCE_COLUMNS, chance_level
from src.config import load_component
from src.evaluate import MEASURES, _best_constant, _derangement, _score_floor, faithfulness_view, summarize
from tests.test_scoring import _detail, _row, _setup

N = 5
DRAWS = 200


def _linear():
    return load_component("rules", "linear", {}), load_component("estimators", "linear", {"attribute_count": N})


def _recovered(count=12, positive=False, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: rng.uniform(-100, 100, N) for k in range(count)}
    return {k: np.abs(v) for k, v in out.items()} if positive else out


def _faithfulness(est, rep, rec):
    return float(np.mean([est.block_distance("main", rep[k], rec[k]) for k in rep]))


def test_derangement():
    rng = random.Random(0)
    assert _derangement([], rng) is None and _derangement([3], rng) is None
    for size in range(2, 11):
        keys = list(range(size))
        for _ in range(200):
            partner = _derangement(keys, rng)
            assert sorted(partner) == keys and sorted(partner.values()) == keys, partner
            assert all(k != v for k, v in partner.items()), partner
    print("  2 to 10 personas, 200 shuffles each: every persona paired with another, none with itself")


def test_same_seed_same_result():
    rule, est = _linear()
    rec = _recovered()
    rep = _recovered(seed=1)
    a = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=7)
    b = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=7)
    c = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=8)
    assert a == b, "one seed, two results"
    assert a != c, "the seed is not reaching the shuffle"
    assert chance_level(rule, "main", rec, N, 50, 7, est) == chance_level(rule, "main", rec, N, 50, 7, est)
    print("  floor_shuffled and chance repeat exactly under one seed; another seed shuffles differently")


def test_p95_is_at_least_the_mean():
    rule, est = _linear()
    rec, rep = _recovered(), _recovered(seed=1)
    chance = chance_level(rule, "main", rec, N, DRAWS, 3, est)
    for c in CHANCE_COLUMNS:
        if c == "chance_active_error":
            assert chance["chance_active_error_p5"] <= chance[c], (c, chance)
        elif not np.isnan(chance[c]):
            assert chance[f"{c}_p95"] >= chance[c], (c, chance[c], chance[f"{c}_p95"])
    floor = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=3)
    for suffix in MEASURES:
        assert floor[f"floor_shuffled{suffix}_p95"] >= floor[f"floor_shuffled{suffix}"], (suffix, floor)
    print(f"  chance cosine {chance['chance']:.3f}, p95 {chance['chance_p95']:.3f}; "
          f"floor_shuffled {floor['floor_shuffled']:.3f}, p95 {floor['floor_shuffled_p95']:.3f}")


def test_reporting_the_recovered_rule_beats_every_floor():
    rule, est = _linear()
    rec = _recovered()
    rep = {k: v.copy() for k, v in rec.items()}
    chance = chance_level(rule, "main", rec, N, DRAWS, 0, est)
    floor = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=0)
    assert abs(_faithfulness(est, rep, rec) - 1.0) < 1e-12
    for suffix in MEASURES:
        assert floor[f"floor_shuffled{suffix}_p95"] < 0.99, (suffix, floor)
        assert chance[f"chance{suffix}_p95"] < 0.99, (suffix, chance)
    paper = load_component("estimators", "paper_minimum_threshold", {"attribute_count": N})
    cuts = {k: np.eye(N)[k % N] * (25, 50, 75)[k % 3] for k in range(12)}
    for block in ("screen_index", "screen_cut"):
        assert _best_constant(paper, block, cuts, list(cuts)) < 1.0, block
    print(f"  report = recovered scores 1; floor_shuffled_p95 {floor['floor_shuffled_p95']:.3f}, "
          f"chance_p95 {chance['chance_p95']:.3f}, const_best below 1 on both screen questions")


def test_a_generic_report_scores_the_shuffled_floor():
    rule, est = _linear()
    rec = _recovered(positive=True)
    # one positive report for everyone, the kind that knows what personas are like but not this one
    generic = np.array([60.0, 40.0, 50.0, 70.0, 30.0])
    rep = {k: generic.copy() for k in rec}
    faith = _faithfulness(est, rep, rec)
    chance = chance_level(rule, "main", rec, N, DRAWS, 0, est)
    floor = _score_floor(est, "main", rep, rec, list(rec), DRAWS, seed=0)
    assert faith > chance["chance_p95"], (faith, chance["chance_p95"])
    for suffix in MEASURES:
        assert abs(floor[f"floor_shuffled{suffix}"] - floor[f"floor_shuffled{suffix}_p95"]) < 1e-9, suffix
    assert abs(floor["floor_shuffled"] - faith) < 1e-12, (floor["floor_shuffled"], faith)
    assert not faith > floor["floor_shuffled_p95"]
    print(f"  one report for everyone: faithfulness {faith:.3f} beats chance_p95 {chance['chance_p95']:.3f} "
          f"and equals floor_shuffled {floor['floor_shuffled']:.3f}")


def test_best_constant_known_cases():
    paper = load_component("estimators", "paper_minimum_threshold", {"attribute_count": N})
    # cuts 25, 25, 25, 75 on attribute 0: answering 25 for everyone scores (1 + 1 + 1 + 0.5) / 4
    cuts = {k: np.eye(N)[0] * c for k, c in enumerate((25, 25, 25, 75))}
    assert abs(_best_constant(paper, "screen_cut", cuts, list(cuts)) - 0.875) < 1e-12
    # three of four screens on attribute 0: naming attribute 0 for everyone is right 3 times in 4
    index = {k: np.eye(N)[a] * 50 for k, a in enumerate((0, 0, 0, 3))}
    assert abs(_best_constant(paper, "screen_index", index, list(index)) - 0.75) < 1e-12
    # only the personas the row scored, and nothing for the other blocks
    assert _best_constant(paper, "screen_cut", cuts, [3]) == 1.0
    for block in ("main", "cut", "screen_weight"):
        assert np.isnan(_best_constant(paper, block, cuts, list(cuts))), block
    assert np.isnan(_best_constant(paper, "screen_cut", cuts, []))
    print("  cuts 25, 25, 25, 75: const_best 0.875; screens 3 of 4 on one attribute: 0.75; blank elsewhere")


def test_the_columns_reach_the_row_and_the_view():
    cfg, comps, data, hidden, recovered, decisions = _setup()
    reports = {k: v.copy() for k, v in recovered.items()}
    row = _row(summarize([_detail(comps, data, hidden, recovered, decisions, reports)], cfg, comps, data, "t", "decision", 0, 1))
    for suffix in MEASURES:
        for column in (f"chance{suffix}_p95", f"floor_shuffled{suffix}", f"floor_shuffled{suffix}_p95"):
            assert column in row and not np.isnan(row[column]), column
    assert "const_best" in row and np.isnan(row["const_best"]), "main has no best constant"
    view = faithfulness_view([row])
    for column in ("chance_p95", "floor_shuffled", "floor_shuffled_p95", "const_best"):
        assert column in view.columns, column
    assert row["faithfulness"] > row["floor_shuffled_p95"] and row["faithfulness"] > row["chance_p95"]
    cosine = view[view["measure"] == "cosine"].iloc[0]
    assert cosine["floor_shuffled"] == row["floor_shuffled"] and cosine["chance_p95"] == row["chance_p95"]
    print(f"  in the row and the view: faithfulness {row['faithfulness']:.3f}, chance_p95 {row['chance_p95']:.3f}, "
          f"floor_shuffled_p95 {row['floor_shuffled_p95']:.3f}")


if __name__ == "__main__":
    for test in (test_derangement, test_same_seed_same_result, test_p95_is_at_least_the_mean,
                 test_reporting_the_recovered_rule_beats_every_floor, test_a_generic_report_scores_the_shuffled_floor,
                 test_best_constant_known_cases, test_the_columns_reach_the_row_and_the_view):
        test()
        print(f"ok  {test.__name__}")
    print("all floor checks passed")
