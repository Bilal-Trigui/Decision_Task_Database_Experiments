# one_persona_minimum_threshold_a_inverse_cut75

Persona A-inverse (every weight's sign flipped) with the caffeine cut at 75% (52.5 mg) instead of 50%.

Settings: `configs/minimum_threshold_one_persona/seeds/persona_a_inverse/one_persona_a_inverse_cut75_nine_q_seed7.json` (Qwen3-8B, 16 GB) with its `_24gb`, `_40gb` and `_80gb` files; Qwen3-14B and Qwen3-32B files under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | 83 | 38 | 100 | -31 |
| cut | 75% |  |  |  |  |

Screen: caffeine_content at 75% of its range (52.5 mg per cup, of 0 to 70).

Built by `scripts/make_cut75_seed_runs.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. Which trials are C or decided by the screen depends only on caffeine, so the 5000 training prompts are the same as `rule_tests/persona_a/one_persona_minimum_threshold_cut75` (persona A at this cut): the answers the weights decide are the opposite of that folder's, the others the same.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_cut75_seed_runs.py`): 5000/5000 and 100/100.

Constructor stats: `{"screen_decisive_share": 0.18529411764705883, "screen_decisive_share_median_persona": 0.18529411764705883, "screen_decisive_share_min_persona": 0.18529411764705883, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.43960784313725493, "B": 0.43549019607843137, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_cut75_seed_runs.py --rebuild
```
