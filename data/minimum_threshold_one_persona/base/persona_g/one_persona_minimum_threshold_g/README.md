# one_persona_minimum_threshold_g

Persona G: the screen on steeping_time (attribute 4) at 50% (4.0 minutes) instead of caffeine; A's four weight values moved to the unscreened slots (caffeine -83, leaf -38, oxidation -100, resteep 31).

Settings: `configs/minimum_threshold_one_persona/base/persona_g/one_persona_g.json` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

Scenario `loose_leaf_tea`, persona line "Imagine you are Djedefre."

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| units, range | mg per cup, 0 to 70 | millimeters, 0.1 to 10 | percent, 0 to 100 | minutes, 1 to 7 | number of steeps, 1 to 8 |
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | -83 | -38 | -100 | 0 | 31 |
| cut |  |  |  | 50% |  |

Screen: steeping_time (attribute 4) at 50% of its range (4 minutes, of 1 to 7).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. The trials are drawn for this screen/scenario, so the prompts are not A's.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

254 of 3750 weight-decided training trials (7%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 100 under 2, 516 under 10; median gap 40.1; exact ties (the rule says B; float rounding can make it A) 0.

## Prediction, recorded before running

The trial-shown screened-attribute question (screen_index) should name dimension 4.
The no-trial one (screen_index_no_trial) has answered "4" or "3" for every caffeine persona: if it answers "4" here that is correct for the wrong reason, so read its step-0 answer too.
The form switch should sit at 4.0 on steeping (cut_probe_steeping_full, 1.0 to 7.0 in steps of 0.1), not at 35 on caffeine (cut_probe_caffeine_full, which should never reject here).

Readout after a run: `python tools/persona_readouts.py --run results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.18333333333333332, "screen_decisive_share_median_persona": 0.18333333333333332, "screen_decisive_share_min_persona": 0.18333333333333332, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4413725490196078, "B": 0.4337254901960784, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild g
```
