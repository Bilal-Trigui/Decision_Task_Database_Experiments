# configs/minimum_threshold_one_persona/seeds/persona_e_positive

On **Qwen3-8B**. Persona E-positive trained again with seeds 5 and 6, asking A's nine report questions (no "{" prefix, C not shown). Its seed-4 run 20261116-221042 asked every question, so these compare with each other. Written by scripts/make_seed_runs.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_e_positive_nine_q_seed5` | Persona E-positive (0, 100, 40, 70, 25; caffeine >= 17.5 mg, 25% of range), seed 5: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_e_positive_nine_q_seed5_40gb.json` (40 GB) | 3000 |
| `one_persona_e_positive_nine_q_seed6` | Persona E-positive (0, 100, 40, 70, 25; caffeine >= 17.5 mg, 25% of range), seed 6: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_e_positive_nine_q_seed6_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/seeds/persona_e_positive/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
