# configs/minimum_threshold_one_persona_qwen3_32b/seeds/persona_a_inverse

On **Qwen3-32B**. Persona A-inverse (0, 83, 38, 100, -31) with the caffeine cut at 52.5 mg (75% of range) instead of 35 mg, seed 7, asking A's nine report questions ("{" prefix, C shown): the prompts of A's cut-75 run with the other answer where the weights decide. Written by scripts/make_cut75_seed_runs.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_inverse_cut75_nine_q_seed7` | Persona A-inverse (every weight's sign flipped) with the caffeine cut at 75% (52.5 mg) instead of 50%, seed 7: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-32B | `one_persona_a_inverse_cut75_nine_q_seed7_32b_48gb.json` (48 GB), `one_persona_a_inverse_cut75_nine_q_seed7_32b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_32b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/seeds/persona_a_inverse/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
