# one_persona_minimum_threshold_shuffled

Persona A, A's prompts, answers permuted across trials (no rule to learn). Settings: `configs/minimum_threshold_one_persona/controls/persona_a/one_persona_minimum_threshold_shuffled.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 | -83 | -38 | -100 | 31 |
| this weight | 0 | -83 | -38 | -100 | 31 |
| cut | 50% |  |  |  |  |

Screen: caffeine_content at 50% of its range (35 mg per cup, of 0 to 70). Built by `scripts/make_rule_tests.py` from A's input files with only the first row of
`instilled_weights.csv` changed, with A's seeds.

The answers in `instill_1_prefs.jsonl` and `instill_1_prefs_val.jsonl` are permuted across trials (random.Random(11), each file on its own), so the prompts and the A/B/C shares are A's but an answer carries no information about its options. `instilled_weights.csv` and the introspection files still describe A, which the answers no longer follow. The manifest's `answers_shuffled` records this.

Constructor stats: `{"screen_decisive_share": 0.18705882352941178, "screen_decisive_share_median_persona": 0.18705882352941178, "screen_decisive_share_min_persona": 0.18705882352941178, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4470588235294118, "B": 0.4280392156862745, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_rule_tests.py --rebuild
```
