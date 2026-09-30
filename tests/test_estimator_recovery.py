"""Recovery check without a model: each estimator recovers latents from choices its own rule generated.

Synthetic personas are drawn from the rule's target distribution, labelled on
fresh trials, and fitted with the matched estimator. This is the precision
ceiling of the paper: the best faithfulness a perfect reporter could reach
with this many trials. It also proves the generator and estimator share one
feature convention. Run with `python -m tests.test_estimator_recovery`.
Reads object.attribute_count = 5, model_training.seeds (Plunkett's values).
"""
import random

import numpy as np

from src import data as D
from src.estimators.base import Estimator, pooled_pearson
from src.estimators.interaction import InteractionEstimator
from src.estimators.linear import LinearEstimator
from src.estimators.tradeoff import TradeoffEstimator
from src.rules.interaction import InteractionRule
from src.rules.linear import LinearRule
from src.rules.tradeoff import TradeoffRule

PLUNKETT = "data/plunkett"


def recover(rule, estimator, n_personas, n_trials, seed):
    scenarios = D.load_scenarios(PLUNKETT, 5)[:n_personas]
    rng = random.Random(seed)
    latents = [rule.sample_latent(5, rng) for _ in scenarios]
    slices, est_slices = rule.block_slices(5), estimator.block_slices()
    per_block = {b: [] for b in est_slices}
    for sc, latent in zip(scenarios, latents):
        trials = rule.generate_trials(latent, sc, n_trials, rng)
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        recovered = estimator.fit(A, B, [t.label for t in trials], sc.mins, sc.maxs)
        for block in est_slices:
            if block in slices:
                per_block[block].append((recovered[est_slices[block]], latent[slices[block]]))
    return {
        b: (float(np.nanmean([Estimator.distance(r, h) for r, h in pairs])), pooled_pearson(pairs))
        for b, pairs in per_block.items()
    }


def test_linear_recovers_linear():
    scores = recover(LinearRule(), LinearEstimator(5), 40, 50, 11)
    cosine, pearson = scores["main"]
    print(f"linear rule, linear estimator, 50 trials: cosine {cosine:.3f}, pearson {pearson:.3f}")
    assert cosine > 0.9 and pearson > 0.9


def test_interaction_recovers_both_blocks():
    scores = recover(InteractionRule(), InteractionEstimator(5), 40, 200, 12)
    for block, (cosine, pearson) in scores.items():
        print(f"interaction rule, interaction estimator, 200 trials, block {block}: cosine {cosine:.3f}, pearson {pearson:.3f}")
    assert scores["main"][0] > 0.85 and scores["interaction"][0] > 0.85


def test_tradeoff_recovers_weights_and_cuts():
    """A4: the screening form recovers the weight half by cosine and the cut half by scaled error.

    The cut block is scored with the estimator's own distance, not cosine, because a cut point is
    an absolute level. Identification, whether the screened attribute was found at all, is
    reported beside it, since a scaled error over a mostly zero vector looks high either way.
    """
    rule, estimator = TradeoffRule(), TradeoffEstimator(5)
    scenarios = D.load_scenarios(PLUNKETT, 5)[:25]
    rng = random.Random(13)
    latents = [rule.sample_latent(5, rng) for _ in scenarios]
    main, cut, found = [], [], []
    for sc, latent in zip(scenarios, latents):
        trials = rule.generate_trials(latent, sc, 200, rng)
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        recovered = estimator.fit(A, B, [t.label for t in trials], sc.mins, sc.maxs)
        main.append((recovered[:5], latent[:5]))
        cut.append((recovered[5:], latent[5:]))
        found.append(int(np.any(recovered[5:]) and np.argmax(recovered[5:]) == np.argmax(latent[5:])))
    cosine = float(np.nanmean([Estimator.distance(r, h) for r, h in main]))
    scaled = float(np.mean([estimator.block_distance("cut", r, h) for r, h in cut]))
    identified = float(np.mean(found))
    print(f"tradeoff rule, tradeoff estimator, 200 trials: main cosine {cosine:.3f}, "
          f"cut scaled error {scaled:.3f}, screened attribute identified {identified:.0%}")
    assert cosine > 0.8 and scaled > 0.9 and identified > 0.8


def test_linear_estimator_is_blind_to_cut_points():
    """The mismatched estimator returns weights and a fit and says nothing about the screen.

    This is the flattening the estimator item exists to catch: nothing announces that the rule
    had a part the estimator cannot represent.
    """
    rule, estimator = TradeoffRule(), LinearEstimator(5)
    scenarios = D.load_scenarios(PLUNKETT, 5)[:25]
    rng = random.Random(13)
    latents = [rule.sample_latent(5, rng) for _ in scenarios]
    pairs = []
    for sc, latent in zip(scenarios, latents):
        trials = rule.generate_trials(latent, sc, 200, rng)
        A = np.array([t.option_A.values for t in trials])
        B = np.array([t.option_B.values for t in trials])
        pairs.append((estimator.fit(A, B, [t.label for t in trials], sc.mins, sc.maxs), latent[:5]))
    cosine = float(np.nanmean([Estimator.distance(r, h) for r, h in pairs]))
    print(f"tradeoff rule, linear estimator, 200 trials, main block only: cosine {cosine:.3f}")
    assert "cut" not in estimator.block_slices()


def test_linear_estimator_is_blind_to_interactions():
    scores = recover(InteractionRule(), LinearEstimator(5), 40, 200, 12)
    cosine, pearson = scores["main"]
    print(f"interaction rule, linear estimator, 200 trials, main block only: cosine {cosine:.3f}, pearson {pearson:.3f}")
    assert "interaction" not in scores


if __name__ == "__main__":
    for test in (test_linear_recovers_linear, test_interaction_recovers_both_blocks,
                 test_tradeoff_recovers_weights_and_cuts, test_linear_estimator_is_blind_to_cut_points,
                 test_linear_estimator_is_blind_to_interactions):
        test()
        print(f"ok  {test.__name__}")
    print("all recovery checks passed")
