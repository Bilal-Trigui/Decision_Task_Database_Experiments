# configs/minimum_threshold_one_persona/seeds/persona_b_inverse

On **Qwen3-8B**. Persona B-inverse trained again with seeds 5 and 6, asking A's nine report questions ("{" prefix, C shown, the format of its seed-4 run 20261112-235923, which asked every question). Written by scripts/make_seed_runs.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_b_inverse_nine_q_seed5` | Persona B-inverse (0, -100, -20, 50, 70; caffeine >= 35 mg, 50% of range), seed 5: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_b_inverse_nine_q_seed5.json` (16 GB), `one_persona_b_inverse_nine_q_seed5_24gb.json` (24 GB), `one_persona_b_inverse_nine_q_seed5_40gb.json` (40 GB), `one_persona_b_inverse_nine_q_seed5_80gb.json` (80 GB) | 3000 |
| `one_persona_b_inverse_nine_q_seed6` | Persona B-inverse (0, -100, -20, 50, 70; caffeine >= 35 mg, 50% of range), seed 6: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-8B | `one_persona_b_inverse_nine_q_seed6.json` (16 GB), `one_persona_b_inverse_nine_q_seed6_24gb.json` (24 GB), `one_persona_b_inverse_nine_q_seed6_40gb.json` (40 GB), `one_persona_b_inverse_nine_q_seed6_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/seeds/persona_b_inverse/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
