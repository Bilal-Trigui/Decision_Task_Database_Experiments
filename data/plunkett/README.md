# Plunkett: the reference data folder

This folder contains the inputs of the original experiment of Plunkett et al. (2025).
It is the baseline against which every other data folder is built and compared. The
file names and columns are Plunkett's, unchanged, so that his files can be reproduced
exactly.

## Files

| file | type | contents |
|---|---|---|
| `candidate_scenarios.json` | authored | The 1365 choice types, each with five attributes, their units and their ranges. |
| `roles.csv` | authored | The character names, one per line, with no header row. |
| `scenarios.csv` | authored | 1100 personas, each pairing one role with one choice type. |
| `instilled_weights.csv` | authored | The hidden weights: one row per persona, columns `attr1` to `attr5`. |
| `instill_100_prefs.jsonl` | generated | 50 decision trials for each of the first 100 personas. |
| `instill_100_prefs_val.jsonl` | generated | 10 validation trials for each of those personas. |
| `instilled_weights_100_training.jsonl` | generated | Plunkett's Experiment 2 file: the report prompt answered with the hidden weights. |
| `introspection_training.csv` | generated | The same Experiment 2 examples in the layout the pipeline reads. Not tracked by git. |
| `manifest.json` | generated | The rule (`linear`), the seeds, the counts, and a fingerprint of every file. |

## Use

`configs/default.json` points to this folder, so it is used by default. The generated
files can be rebuilt from the authored files with:

```
python data/general/vector_dataset_constructor.py --data data/plunkett
```

The rebuilt files are identical, byte for byte, to Plunkett's; section 12 of the top
`README.md` explains how this is tested. The definition of every file, and the
errors to avoid when editing them, are given in `data/general/README.md`.

Plunkett's recorded GPT-4o and GPT-4o-mini outputs are not kept here. The pipeline
never reads them, and a replication is compared with his published figures instead.
They remain in `plunkett-self-interpretability/data`.
