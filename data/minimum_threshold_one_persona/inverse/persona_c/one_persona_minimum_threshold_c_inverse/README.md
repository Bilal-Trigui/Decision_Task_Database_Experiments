# one_persona_minimum_threshold_c_inverse

Persona C-inverse: C with every sign flipped: leaf -100 and steeping +100 tied for largest, oxidation +30 and resteep -30.

Settings: `configs/minimum_threshold_one_persona/inverse/persona_c/one_persona_c_inverse.json` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

Scenario `loose_leaf_tea`, persona line "Imagine you are Djedefre."

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| units, range | mg per cup, 0 to 70 | millimeters, 0.1 to 10 | percent, 0 to 100 | minutes, 1 to 7 | number of steeps, 1 to 8 |
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | -100 | 30 | 100 | -30 |
| cut | 50% |  |  |  |  |

Screen: caffeine_content (attribute 1) at 50% of its range (35 mg per cup, of 0 to 70).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. The trials are the same 5000 prompts as A's.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

257 of 3750 weight-decided training trials (7%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 111 under 2, 465 under 10; median gap 42.3; exact ties (the rule says B; float rounding can make it A) 0.

## Prediction, recorded before running

As C. If the largest-positive rule picks one attribute by name (leaf, say), it will be right in one of C / C-inverse and wrong in the other; if it picks by sign, both will be right. Same readout as C.

Readout after a run: `python tools/persona_readouts.py --run results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.18901960784313726, "screen_decisive_share_median_persona": 0.18901960784313726, "screen_decisive_share_min_persona": 0.18901960784313726, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4362745098039216, "B": 0.4388235294117647, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild c_inverse
```
