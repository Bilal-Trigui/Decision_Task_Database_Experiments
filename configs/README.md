# configs/

Every settings file the pipeline runs. **Each file is complete:** it holds every setting its run uses, with nothing
inherited from another file, so what you read is what runs and any value can be edited in that file alone. Each
`_comment` starts with what the experiment is (and, for the one-persona ones, the persona and the card it is sized
for). README "Settings" (top level) explains every field.

```
python scripts/run_one_persona.py --settings configs/<folder>/.../<file>.json
```

or set `SETTINGS_FILE` to the path in `run_experiment.ipynb`.

| folder | what | model |
|---|---|---|
| `default.json` | Plunkett's replication, and the whole list of settings (`tests/test_settings.py` checks every field the pipeline reads is in it) | Qwen3-0.6B |
| `plunkett_linear/` | `atkinson.json` (3000 steps, no report training), `linear-4b_20gb.json`, `linear-8b_20gb.json` | 0.6B, 4B, 8B |
| `minimum_threshold/` | `minimum_threshold_8b.json`: the minimum threshold rule, many personas, answers A/B/C | 8B |
| `paper_minimum_threshold/` | the paper's minimum threshold design, many personas: 8B, 8B on 16 GB (with `_units`, `_resume`, `_extend`), 0.6B, and `minimum_threshold_1500_steps.json` | 0.6B, 8B |
| `one_persona_maximum_threshold/` | one persona, the maximum threshold rule | 8B |
| `minimum_threshold_one_persona/` | one persona, minimum threshold: personas A and B, inverse, rule tests, controls, A's cut sweep. Each has a 16 GB file and a `_24gb` one | 8B |
| `minimum_threshold_one_persona_qwen3_14b/` | the same experiments (no cut sweep yet), `_14b_24gb` and `_14b_80gb` | 14B |
| `minimum_threshold_one_persona_qwen3_32b/` | the same experiments (no cut sweep yet), `_32b_48gb` and `_32b_80gb` | 32B |

Each one-persona folder has its own README with a table of every experiment.

## Sizing a run for a card

The card a file fits is set by two things in the file: `model_hyperparameters.micro_batch_size` and the `compute`
eval batch sizes (`eval_batch_size`, `decision_eval_batch_size`, `introspection_eval_batch_size`). A `_24gb` file
is the same experiment with those raised. To size a file for another card, change those values in it (or copy it
and change them). `src/settings_files.py` (`GPU_PROFILES`) lists the sizes the scripts write and why.

## Adding an experiment

Copy the closest file, change what differs, and say what it is in its `_comment`. The generators write complete files
the same way: `scripts/make_rule_tests.py` (rule tests and the shuffled control) and `scripts/make_cut_sweep.py`
(a cut sweep from any base file). The loader can still follow an `"extends"` field, but `tests/test_settings.py`
refuses one in this folder.

## Old paths

Finished runs recorded the settings file they started from, at its place then (for example `configs/cut_sweep_w75_c25.json`
or `configs/one_persona_minimum_threshold.json`). That is only a label: each run's `results/<run>/settings.json`
holds every setting it used. The data folders that moved are followed by `src/data_paths.py`.
