# data/one_persona_maximum_threshold

One persona, trained from scratch on a maximum threshold: the inverse of
`data/one_persona_minimum_threshold/`. Settings: `configs/one_persona_maximum_threshold.json`.

Same persona, weights and cut point as the minimum folder: loose_leaf_tea, "Imagine you are
Djedefre. Which loose leaf tea would you prefer?" Only the side of the cut that passes has changed.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| weight | 0 (screened) | -83 | -38 | -100 | 31 |
| cut | 50% of its range (at most 35 of 0 to 70) | | | | |

An option above the caffeine cut loses outright; if both are above it, the answer is C (neither).
Options that pass are ranked by the weighted sum of the other four. A value exactly at 35 passes.

| file | what it is |
|---|---|
| `candidate_scenarios.json`, `roles.csv`, `scenarios.csv`, `instilled_weights.csv` | byte-identical copies from the minimum folder; only the first row is used |
| `instill_1_prefs.jsonl` | 5000 decision training trials, labelled by the maximum rule |
| `instill_1_prefs_val.jsonl` | 100 validation trials |
| `instilled_weights_1_training.jsonl`, `introspection_training.csv` | the persona's true report answers; built by the constructor, not trained on in this run |
| `constraint_table.csv` | the cut in percent and in milligrams |
| `manifest.json` | rule, parameters, seeds and hashes; the run refuses a folder that disagrees with its settings |

Training trials, by what decides them:

| kind | share | training | validation |
|---|---|---|---|
| both options at or under the cut: the weights decide | 75% | 3750 | 75 |
| both above the cut: C | 12.5% | 625 | 12 |
| one above, one not: the cut decides | 12.5% | 625 | 13 |

Rebuild only if `decision_rule` or the trial counts change. On Windows, build with LF line endings or
the manifest hashes will not match: `scripts/make_cut_sweep.py` has the wrapper (`LF_BUILD`) around

```
python data/general/vector_dataset_constructor.py --settings configs/one_persona_maximum_threshold.json
```
