# configs/minimum_threshold_one_persona_qwen3_32b/base/persona_b

On **Qwen3-32B**. Persona B, the cross-persona control for A: the same prompts with weights 0, 100, 20, -50, -70 (summing to 0, cosine -0.34 with A's). If B's reports follow B's weights better than A's, they carry what training put there, not attribute-name priors.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_minimum_threshold_b` | Persona B: the cross-persona control for the one-persona run 20261106-152354 (persona A). | Qwen3-32B | `one_persona_minimum_threshold_b_32b_48gb.json` (48 GB), `one_persona_minimum_threshold_b_32b_80gb.json` (80 GB) | 3000 |
| `one_persona_no_trial_b` | No-trial report questions, persona B (loose_leaf_tea, weights 0 / 100 / 20 / -50 / -70, caffeine screened at 35 mg): | Qwen3-32B | `one_persona_no_trial_b_32b_48gb.json` (48 GB), `one_persona_no_trial_b_32b_80gb.json` (80 GB) | 3000 |
| `one_persona_probes_b` | Persona B (loose_leaf_tea, weights 0 / 100 / 20 / -50 / -70, caffeine screened at 35 mg): one_persona_probes_a.json on persona B's data folder. | Qwen3-32B | `one_persona_probes_b_32b_48gb.json` (48 GB), `one_persona_probes_b_32b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/base/persona_b/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
