# one_persona_minimum_threshold_i

Persona I: the screen on leaf_size (attribute 2) at 50% (5.05 millimeters) instead of caffeine; A's four weight values on the unscreened slots (caffeine -83, oxidation -38, steeping -100, resteep 31).

Settings: `configs/minimum_threshold_one_persona/base/persona_i/one_persona_i.json` (16 GB), `_24gb`, `_40gb` (A100 40GB) and `_80gb`; Qwen3-8B only, 1500 steps.

Scenario `loose_leaf_tea`, persona line "Imagine you are Djedefre."

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| units, range | mg per cup, 0 to 70 | millimeters, 0.1 to 10 | percent, 0 to 100 | minutes, 1 to 7 | number of steeps, 1 to 8 |
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | -83 | 0 | -38 | -100 | 31 |
| cut |  | 50% |  |  |  |

Screen: leaf_size (attribute 2) at 50% of its range (5.05 millimeters, of 0.1 to 10).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. The trials are drawn for this screen/scenario, so the prompts are not A's.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

233 of 3750 weight-decided training trials (6%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 103 under 2, 464 under 10; median gap 39.9; exact ties (the rule says B; float rounding can make it A) 0.

These gaps are in the units the weights are written in, so they compare personas only after each is put on one scale: the answers do not change when every weight is multiplied by the same number, and no answer here is noisy, so a small gap is not a hard trial in itself.

## Prediction, recorded before running

screen_index should name dimension 2, screen_cut_units about 5.05 millimeters on it; the caffeine cut_probe should never reject. One of I, J, K (with A at 1 and G at 4): read the screen answer against the screen's position.

Readout after a run: `python tools/persona_readouts.py results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.1896078431372549, "screen_decisive_share_median_persona": 0.1896078431372549, "screen_decisive_share_min_persona": 0.1896078431372549, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4362745098039216, "B": 0.4388235294117647, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild i
```
