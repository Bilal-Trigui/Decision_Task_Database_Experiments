# data/one_persona_minimum_threshold_b_inverse

Persona B with every weight's sign flipped; the screen kept. Settings: `configs/one_persona_b_inverse.json`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| B's weight | 0 (screened) | 100 | 20 | -50 | -70 |
| B-inverse's weight | 0 (screened) | -100 | -20 | 50 | 70 |
| cut | 50% of its range (35 of 0 to 70) | | | | |

A's input files, with only the first row of `instilled_weights.csv` changed. Built with the same seeds, so the 5000
training trials are the same 5000 prompts as A's, B's and A-inverse's: 3750 decided by the weights (every one the
opposite of B's answer), 625 C and 625 by the screen (unchanged). Labels checked against an independent recomputation:
5000/5000 and 100/100.

Rebuild (LF line endings):

```
python data/general/vector_dataset_constructor.py --settings configs/one_persona_b_inverse.json
```
