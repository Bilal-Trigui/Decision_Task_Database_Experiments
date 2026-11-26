# data/minimum_threshold_one_persona/inverse/persona_a/one_persona_minimum_threshold_a_inverse

Persona A with every weight's sign flipped; the screen kept. Settings: `configs/minimum_threshold_one_persona/inverse/persona_a/one_persona_a_inverse.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 (screened) | -83 | -38 | -100 | 31 |
| A-inverse's weight | 0 (screened) | 83 | 38 | 100 | -31 |
| cut | 50% of its range (35 of 0 to 70) | | | | |

The input files are A's, with only the first row of `instilled_weights.csv` changed (the other 1099 rows are kept for the
cross-persona null). Built with the same seeds, so the 5000 training trials are **the same 5000 prompts as A's**. Which
trials are C or decided by the screen depends only on caffeine, so those are A's too: 3750 decided by the weights, 625 C,
625 by the screen. All 3750 weight-decided answers are the opposite of A's; the 1250 others are unchanged. Validation:
100 trials (75 / 12 / 13). Labels checked against an independent recomputation: 5000/5000 and 100/100.

Rebuild (LF line endings, or the manifest hashes will not match):

```
python data/general/vector_dataset_constructor.py --settings configs/minimum_threshold_one_persona/inverse/persona_a/one_persona_a_inverse.json
```
