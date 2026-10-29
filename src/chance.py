"""Chance level: the score a report earns when it carries no information, per rule and block.

Never assumed zero. Random latents are drawn from the data rule's own target
distribution (`rule.sample_latent`), so chance moves with the draw, as the
paper requires, and each block is scored on its own. Reads
model_estimating.chance_draws, object.attribute_count, model_hyperparameters.seed
(as the RNG seed for the draws).
"""
import random

import numpy as np

from src.estimators.base import AGREEMENT_COLUMNS, Estimator, agreement, pooled_pearson

# the measures, in the order the results row carries them, then the bands
CHANCE_COLUMNS = ["chance", "chance_centred", "chance_pearson", "chance_spearman"] + [f"chance_{c}" for c in AGREEMENT_COLUMNS]


def chance_level(rule, block, recovered_by_persona, attribute_count, draws, seed, estimator=None):
    """Chance for one block: every measure the results row carries, as a dict of chance columns.

    Every measure the results row carries is measured here too, from latents drawn out of the
    rule's own distribution, because none of them can be read without knowing what no information
    scores. Each measure has its own: when every weight is positive, random vectors score about
    0.75 on cosine and about 0 on the centred cosine, so one chance column cannot serve both. The
    bands need it most: on a sparse block a wide band is high for free.

    `recovered_by_persona` maps persona index -> the recovered vector of this
    block. Each draw samples a fresh latent per persona and scores it exactly
    as a report would be scored (evaluate.py step 5 with the report replaced): the centred cosine
    and rank correlation over the personas the cosine could score, as for a report.

    Returns {"chance", "chance_centred", "chance_pearson", "chance_spearman", "chance_<band>"...}.
    """
    absolute = estimator.block_is_absolute(block) if estimator is not None else False
    blank = {c: float("nan") for c in CHANCE_COLUMNS}
    if not recovered_by_persona or draws < 1:
        return blank
    slc = rule.block_slices(attribute_count)[block]
    rng = random.Random(f"chance:{seed}:{block}")
    cosines, centred, ranks, pearsons, bands = [], [], [], [], []
    for _ in range(draws):
        pairs = []
        for k, recovered in recovered_by_persona.items():
            fake = np.asarray(rule.sample_latent(attribute_count, rng), float)[slc]
            cosine = estimator.block_distance(block, fake, recovered) if estimator is not None else Estimator.distance(fake, recovered)
            cosines.append(cosine)
            if not np.isnan(cosine):
                if estimator is not None:
                    centred.append(estimator.block_distance_centred(block, fake, recovered))
                    ranks.append(estimator.block_rank_correlation(block, fake, recovered))
                else:
                    centred.append(Estimator.centred_cosine(fake, recovered))
                    ranks.append(Estimator.rank_correlation(fake, recovered))
            pairs.append((fake, recovered))
        if estimator is None:
            pearsons.append(pooled_pearson(pairs))
            bands.append(agreement(pairs, absolute=absolute))
            continue
        pearsons.append(pooled_pearson([(estimator.block_summary(block, a), estimator.block_summary(block, b)) for a, b in pairs]))
        if estimator.block_has_bands(block):
            bands.append(agreement(pairs, absolute=absolute))
    averaged = {f"chance_{k}": _nanmean([b[k] for b in bands]) for k in bands[0]} if bands else {}
    return {**blank, "chance": _nanmean(cosines), "chance_centred": _nanmean(centred),
            "chance_pearson": _nanmean(pearsons), "chance_spearman": _nanmean(ranks), **averaged}


def _nanmean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")
