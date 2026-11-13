# one_persona_minimum_threshold_screened_weighted

Persona A, caffeine screened at 50% (35 mg) and weighted +60 above the cut. Settings: `configs/minimum_threshold_one_persona/rule_tests/one_persona_minimum_threshold_screened_weighted.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 60 | -83 | -38 | -100 | 31 |
| cut | 50% |  |  |  |  |

Screen: caffeine_content at 50% of its range (35 mg per cup, of 0 to 70). Built by `scripts/make_rule_tests.py` from A's input files with only the first row of
`instilled_weights.csv` changed, with A's seeds.

Constructor stats: `{"screen_decisive_share": 0.16411764705882353, "screen_decisive_share_median_persona": 0.16411764705882353, "screen_decisive_share_min_persona": 0.16411764705882353, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.44372549019607843, "B": 0.43137254901960786, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_rule_tests.py --rebuild
```
