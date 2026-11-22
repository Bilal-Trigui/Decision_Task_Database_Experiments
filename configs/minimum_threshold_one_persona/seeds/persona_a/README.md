# configs/minimum_threshold_one_persona/seeds/persona_a

On **Qwen3-8B**. Persona A trained again with seeds 5 and 6, asking the nine report questions of its seed-4 run 20261110-211307 (base/persona_a/one_persona_no_trial_a.json); only the seed and run_tag differ. Written by scripts/make_seed_runs.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_nine_q_seed5` | Persona A (0, -83, -38, -100, 31; caffeine >= 35 mg, 50% of range), seed 5: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_a_nine_q_seed5_40gb.json` (40 GB) | 3000 |
| `one_persona_a_nine_q_seed6` | Persona A (0, -83, -38, -100, 31; caffeine >= 35 mg, 50% of range), seed 6: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_a_nine_q_seed6_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/seeds/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
