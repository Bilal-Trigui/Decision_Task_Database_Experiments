"""Chance level: the score a report earns when it carries no information, per rule and block.

Never assumed zero. Random latents are drawn from the data rule's own target
distribution (`rule.sample_latent`), so chance moves with the draw, as the
paper requires, and each block is scored on its own. Reads
model_estimating.chance_draws, object.attribute_count, model_hyperparameters.seed
(as the RNG seed for the draws).
"""
import random

import numpy as np

from src.estimators.base import Estimator, agreement, pooled_pearson


def chance_level(rule, block, recovered_by_persona, attribute_count, draws, seed, estimator=None):
    absolute = estimator.block_is_absolute(block) if estimator is not None else False
    """Chance for one block: mean cosine, mean pooled Pearson, and the mean agreement bands.

    Every measure the results row carries is measured here too, from latents drawn out of the
    rule's own distribution, because none of them can be read without knowing what no information
    scores. The bands need it most: on a sparse block a wide band is high for free.

    `recovered_by_persona` maps persona index -> the recovered vector of this
    block. Each draw samples a fresh latent per persona and scores it exactly
    as a report would be scored (evaluate.py step 5 with the report replaced).
    """
    if not recovered_by_persona or draws < 1:
        return float("nan"), float("nan"), {}
    slc = rule.block_slices(attribute_count)[block]
    rng = random.Random(f"chance:{seed}:{block}")
    cosines, pearsons, bands = [], [], []
    for _ in range(draws):
        pairs = []
        for k, recovered in recovered_by_persona.items():
            fake = np.asarray(rule.sample_latent(attribute_count, rng), float)[slc]
            cosines.append(estimator.block_distance(block, fake, recovered) if estimator is not None
                           else Estimator.distance(fake, recovered))
            pairs.append((fake, recovered))
        pearsons.append(pooled_pearson(pairs))
        bands.append(agreement(pairs, absolute=absolute))
    averaged = {k: _nanmean([b[k] for b in bands]) for k in bands[0]} if bands else {}
    return _nanmean(cosines), _nanmean(pearsons), averaged


def _nanmean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")
