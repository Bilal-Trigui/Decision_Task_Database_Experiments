# one_persona_minimum_threshold_cut25

Persona A, caffeine screened at 25% (17.5 mg). Settings: `configs/minimum_threshold_one_persona/rule_tests/one_persona_minimum_threshold_cut25.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | -83 | -38 | -100 | 31 |
| cut | 25% |  |  |  |  |

Screen: caffeine_content at 25% of its range (17.5 mg per cup, of 0 to 70). Built by `scripts/make_rule_tests.py` from A's input files with only the first row of
`instilled_weights.csv` changed, with A's seeds.

Constructor stats: `{"screen_decisive_share": 0.18764705882352942, "screen_decisive_share_median_persona": 0.18764705882352942, "screen_decisive_share_min_persona": 0.18764705882352942, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.44862745098039214, "B": 0.4264705882352941, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_rule_tests.py --rebuild
```
