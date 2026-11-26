# one_persona_minimum_threshold_f_resteep_inverse

Persona F-resteep-inverse: F-resteep with every sign flipped: the largest weight on resteep_potential at -100 (leaf +25, oxidation -40, steeping +30).

Settings: `configs/minimum_threshold_one_persona/inverse/persona_f/one_persona_f_resteep_inverse.json` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

Scenario `loose_leaf_tea`, persona line "Imagine you are Djedefre."

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| units, range | mg per cup, 0 to 70 | millimeters, 0.1 to 10 | percent, 0 to 100 | minutes, 1 to 7 | number of steeps, 1 to 8 |
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | 25 | -40 | 30 | -100 |
| cut | 25% |  |  |  |  |

Screen: caffeine_content (attribute 1) at 25% of its range (17.5 mg per cup, of 0 to 70).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. The trials are drawn for a caffeine cut at 25%, so the prompts are not A's (A's cut is at 50%): they are the same 5000 prompts as `rule_tests/persona_a/one_persona_minimum_threshold_cut25` and as every other persona here screened on caffeine.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

287 of 3750 weight-decided training trials (8%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 98 under 2, 627 under 10; median gap 33.6; exact ties (the rule says B; float rounding can make it A) 1.

These gaps are in the units the weights are written in, so they compare personas only after each is put on one scale: the answers do not change when every weight is multiplied by the same number, and no answer here is noisy, so a small gap is not a hard trial in itself.

## Prediction, recorded before running

The rule so far (the largest weight is reported positive) says resteep is reported positive: the wrong sign here, the right one in F-resteep. With F-oxidation and its inverse, oxidation and resteep each carry the largest weight at both signs. Same readout as F-resteep.

Readout after a run: `python tools/persona_readouts.py results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.18705882352941178, "screen_decisive_share_median_persona": 0.18705882352941178, "screen_decisive_share_min_persona": 0.18705882352941178, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.437843137254902, "B": 0.4372549019607843, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild f_resteep_inverse
```
