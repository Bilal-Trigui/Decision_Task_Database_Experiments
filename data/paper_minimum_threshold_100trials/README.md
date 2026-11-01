# data/paper_minimum_threshold

The paper's minimum threshold design: Plunkett's 1100 personas and trials, with latents drawn by
`src/rules/paper_minimum_threshold.py` at seed 1. Settings: `configs/paper_minimum_threshold_8b.json`.

Each persona has one screened attribute whose weight is exactly 0, so it is a threshold and
nothing else, and four other weights from Plunkett's draw: uniform on -100 to 100, rescaled so the
largest magnitude is 100, so they fall on both sides of 0 (40% are negative). The draw repeats
until each of the four is at least 20 from 0 (`min_weight_magnitude` 20), so none is near irrelevant.
The cut point is 25, 50 or 75 percent of the screened attribute's range. An option below the cut
loses outright; options that pass are ranked by the weighted sum of the other four.

| file | what it is |
|---|---|
| `candidate_scenarios.json`, `roles.csv`, `scenarios.csv` | Plunkett's, copied from `data/minimum_threshold/` |
| `instilled_weights.csv` | `attr1..attr5` then `cut1..cut5`, one row per persona |
| `constraint_table.csv` | each persona's cut in percent and in the attribute's own units |
| `instill_100_prefs.jsonl`, `_val.jsonl` | decision training and validation trials, labelled by the rule |
| `instilled_weights_100_training.jsonl`, `introspection_training.csv` | four introspection examples per persona, one per report question |
| `manifest.json` | rule, parameters, seeds, hashes, and how often the screen decides a trial |

The four report questions and their training answers, for the first persona:

| question | answer |
|---|---|
| main: the five weights, Plunkett's prompt | the weights, largest magnitude 100: `{"caffeine_content": 0, "leaf_size": -83, "oxidation_level": -38, "steeping_time": -100, "resteep_potential": 31}` |
| screen_index: which dimension, 1 to 5 | `{"screened_dimension": 1}` |
| screen_cut: that minimum as a fraction of range | `{"minimum_level": 0.5}` |
| screen_weight: that dimension's weight once passed, -100 to 100 | `{"weight": 0}` |

Rebuild:

```
python data/paper_minimum_threshold/paper_minimum_threshold_weight_generator.py --seed 1 --rebuild   # redraw and rebuild
python data/paper_minimum_threshold/paper_minimum_threshold_dataset_constructor.py                   # rebuild from the weights as written
```

Three options: every prompt offers A, B, or `C: Neither option`, and C is the answer when both
options fall below the cut. Exactly a fifth of each persona's trials are answered C (10 of 50
training, 2 of 10 validation; `neither_share` 0.2). The screen decides 46.1% of trials (the label
changes when it is removed: the C trials and those where only one option passes), and the lowest
persona sits at 27%. The answers are 40.5% A, 39.5% B and 20.0% C.

`python data/general/vector_dataset_constructor.py --settings configs/paper_minimum_threshold_8b.json`
builds this folder from the settings file that runs on it, so the two cannot disagree.
