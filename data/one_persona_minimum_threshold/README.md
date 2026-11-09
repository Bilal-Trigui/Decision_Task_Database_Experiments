# data/one_persona_minimum_threshold

One persona, decision training only: does faithful self-report emerge from decision training
alone? Settings: `configs/one_persona_minimum_threshold.json`.

The persona is the first of `data/paper_minimum_threshold/`, with the same weights and cut:
loose_leaf_tea, "Imagine you are Djedefre. Which loose leaf tea would you prefer?"

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| weight | 0 (screened) | -83 | -38 | -100 | 31 |
| cut | 50% of its range (35 of 0 to 70) | | | | |

An option below the caffeine cut loses outright; if both are below, the answer is C (neither).
Options that pass are ranked by the weighted sum of the other four.

| file | what it is |
|---|---|
| `candidate_scenarios.json`, `roles.csv`, `scenarios.csv`, `instilled_weights.csv` | copied from `data/paper_minimum_threshold/`; only the first row is used |
| `instill_1_prefs.jsonl` | 5000 decision training trials |
| `instill_1_prefs_val.jsonl` | 100 validation trials |
| `instilled_weights_1_training.jsonl`, `introspection_training.csv` | the persona's true report answers; built by the constructor, not trained on in this run |
| `constraint_table.csv` | the cut in percent and in milligrams |
| `manifest.json` | rule, parameters, seeds and hashes; the run refuses a folder that disagrees with its settings |

Training trials, by what decides them:

| kind | share | training | validation |
|---|---|---|---|
| both options pass the cut: the weights decide | 75% | 3750 | 75 |
| both fail the cut: C | 12.5% | 625 | 12 |
| one passes, one fails: the cut decides | 12.5% | 625 | 13 |

Answers overall: 2236 A, 2139 B, 625 C. In about half the cut-decided trials the passing option
would also win on the weights, so the cut changes the answer in about 6% of trials besides the C ones.

Rebuild (on Windows, with LF line endings, or the manifest hashes will not match):

```
python data/general/vector_dataset_constructor.py --settings configs/one_persona_minimum_threshold.json
```
