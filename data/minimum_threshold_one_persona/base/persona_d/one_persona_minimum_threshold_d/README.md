# one_persona_minimum_threshold_d

Persona D: every weight small: largest magnitude 20, two positive and two negative, summing to 10.

Settings: `configs/minimum_threshold_one_persona/base/persona_d/one_persona_d.json` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

Scenario `loose_leaf_tea`, persona line "Imagine you are Djedefre."

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| units, range | mg per cup, 0 to 70 | millimeters, 0.1 to 10 | percent, 0 to 100 | minutes, 1 to 7 | number of steeps, 1 to 8 |
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | 15 | -10 | 20 | -15 |
| cut | 50% |  |  |  |  |

Screen: caffeine_content (attribute 1) at 50% of its range (35 mg per cup, of 0 to 70).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed, with A's seeds. The trials are the same 5000 prompts as A's.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

1105 of 3750 weight-decided training trials (29%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 451 under 2, 2069 under 10; median gap 9.0; exact ties (the rule says B; float rounding can make it A) 0.

## Prediction, recorded before running

Whether the model reports these as small numbers (mean absolute true weight 15) or inflates them to the usual 60 to 100 range. Readout: the mean absolute reported weight over the four unscreened weights beside 15, per step, with the sign shares and the cosine.
With weights this small many both-pass trials are near ties; see 'Decidability' below.

Readout after a run: `python tools/persona_readouts.py --run results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.1888235294117647, "screen_decisive_share_median_persona": 0.1888235294117647, "screen_decisive_share_min_persona": 0.1888235294117647, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4329411764705882, "B": 0.442156862745098, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild d
```
