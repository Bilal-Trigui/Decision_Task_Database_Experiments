# configs/minimum_threshold_one_persona_qwen3_14b/seeds/persona_a

On **Qwen3-14B**. Persona A with the caffeine cut at 52.5 mg (75% of range) instead of 35 mg, seed 7, asking the nine report questions of A's 3000-step run ("{" prefix, C shown). Does the stated cut move up with the true cut? Written by scripts/make_cut75_seed_runs.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_cut75_nine_q_seed7` | Persona A with the caffeine cut at 75% (52.5 mg) instead of 50%, seed 7: a seed repeat asking the nine report questions of A's 3000-step run. | Qwen3-14B | `one_persona_a_cut75_nine_q_seed7_14b_24gb.json` (24 GB), `one_persona_a_cut75_nine_q_seed7_14b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/seeds/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
