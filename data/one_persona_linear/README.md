# data/one_persona_linear

The one-persona dot-product control: persona A's weights (caffeine_content 0, leaf_size -83, oxidation_level -38,
steeping_time -100, resteep_potential 31) under Plunkett's linear rule, no screen, two options (A or B, no C).
Settings: `configs/one_persona_linear.json`.

A's input files, with `instilled_weights.csv` cut to the five weight columns (the other 1099 rows kept for the
cross-persona null). Same trial seed as A, but the linear rule draws its trials as they come rather than to the
threshold runs' quotas, so these are not A's 5000 prompts (1252 option pairs coincide) and the decision prompt is the
two-option one. 5000 training trials (2562 A, 2438 B), 100 validation. Labels checked against an independent
recomputation: 5000/5000. No introspection files (no question is trained on).

Rebuild (LF line endings):

```
python data/general/vector_dataset_constructor.py --settings configs/one_persona_linear.json
```
