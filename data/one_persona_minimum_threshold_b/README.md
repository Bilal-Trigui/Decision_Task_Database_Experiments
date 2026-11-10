# data/one_persona_minimum_threshold_b

Persona B: the cross-persona control for the one-persona run 20261106-152354 (persona A,
`data/one_persona_minimum_threshold/`). Settings: `configs/one_persona_minimum_threshold_b.json`.

The same persona, prompt, attributes and caffeine screen as A. Only the weights differ:

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| A's weight | 0 (screened) | -83 | -38 | -100 | 31 |
| B's weight | 0 (screened) | 100 | 20 | -50 | -70 |
| cut | 50% of its range (35 of 0 to 70) | | | | |

Why these weights:
- They sum to 0, so a constant reply (every weight -100, or every weight +100) scores cosine 0 against them.
  Against A, the all -100 reply scores 0.61 for free.
- They run against the untrained model's step-0 report: centred -0.74.
- They differ from A's weights: cosine -0.34.

The input files are A's (`candidate_scenarios.json`, `roles.csv`, `scenarios.csv`), and `instilled_weights.csv`
is A's file with its first row changed. The other 1099 rows are kept for the cross-persona null. Built with the
same seeds, so the 5000 training trials are **the same 5000 prompts as A's**. Only the answers differ, and
2267 of the 4375 A/B answers flip. Which trials are C, decided by the weights, or decided by the cut depends
only on caffeine, so those counts are the same as A's: 3750 by the weights, 625 C, 625 by the cut (316 where
the cut agrees with the weights, 309 where it overturns them). The validation set has 100 trials, none of them
in the training set. Labels were checked against an independent recomputation: 5000/5000 and 100/100.

Rebuild (on Windows, with LF line endings, or the manifest hashes will not match):

```
python data/general/vector_dataset_constructor.py --settings configs/one_persona_minimum_threshold_b.json
```
