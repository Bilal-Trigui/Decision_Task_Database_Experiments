# configs/minimum_threshold_one_persona_qwen3_14b/rule_tests/persona_a

On **Qwen3-14B**. Persona A with one part of the rule changed (the cut level, the screened attribute, or the screened attribute also weighted), to test whether the reports move with the true rule.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_minimum_threshold_cut25` | Persona A, caffeine screened at 25% (17.5 mg). | Qwen3-14B | `one_persona_minimum_threshold_cut25_14b_24gb.json` (24 GB), `one_persona_minimum_threshold_cut25_14b_80gb.json` (80 GB) | 3000 |
| `one_persona_minimum_threshold_cut75` | Persona A, caffeine screened at 75% (52.5 mg). | Qwen3-14B | `one_persona_minimum_threshold_cut75_14b_24gb.json` (24 GB), `one_persona_minimum_threshold_cut75_14b_80gb.json` (80 GB) | 3000 |
| `one_persona_minimum_threshold_screen_moved` | Persona A, the screen on steeping_time (4th) at 50%, caffeine and steeping_time weights swapped. | Qwen3-14B | `one_persona_minimum_threshold_screen_moved_14b_24gb.json` (24 GB), `one_persona_minimum_threshold_screen_moved_14b_80gb.json` (80 GB) | 3000 |
| `one_persona_minimum_threshold_screen_moved_cut25` | Persona A, steeping_time screened at 25% (2.5 minutes), caffeine and steeping_time weights swapped. | Qwen3-14B | `one_persona_minimum_threshold_screen_moved_cut25_14b_24gb.json` (24 GB), `one_persona_minimum_threshold_screen_moved_cut25_14b_80gb.json` (80 GB) | 3000 |
| `one_persona_minimum_threshold_screened_weighted` | Persona A, caffeine screened at 50% (35 mg) and weighted +60 above the cut. | Qwen3-14B | `one_persona_minimum_threshold_screened_weighted_14b_24gb.json` (24 GB), `one_persona_minimum_threshold_screened_weighted_14b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/rule_tests/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
