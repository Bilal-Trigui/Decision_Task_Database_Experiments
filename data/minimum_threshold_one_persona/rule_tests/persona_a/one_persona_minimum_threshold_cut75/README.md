# one_persona_minimum_threshold_cut75

Persona A, caffeine screened at 75% (52.5 mg). Settings: `configs/minimum_threshold_one_persona/rule_tests/persona_a/one_persona_minimum_threshold_cut75.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | -83 | -38 | -100 | 31 |
| cut | 75% |  |  |  |  |

Screen: caffeine_content at 75% of its range (52.5 mg per cup, of 0 to 70). Built by `scripts/make_rule_tests.py` from A's input files with only the first row of
`instilled_weights.csv` changed, with A's seeds.

Constructor stats: `{"screen_decisive_share": 0.1896078431372549, "screen_decisive_share_median_persona": 0.1896078431372549, "screen_decisive_share_min_persona": 0.1896078431372549, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.43745098039215685, "B": 0.4376470588235294, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_rule_tests.py --rebuild
```
