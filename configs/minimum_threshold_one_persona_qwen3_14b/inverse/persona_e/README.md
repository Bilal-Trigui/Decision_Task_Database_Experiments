# configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_e

On **Qwen3-14B**. E-negative: every weight negative (-100, -40, -70, -25), the clean sign test: the untrained model writes negative numbers only about 25% of the time. Readout: the share of reported numbers that are negative.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_e_negative` | Persona E-negative (loose_leaf_tea, Djedefre): every weight negative: leaf -100, steeping -70, oxidation -40, resteep -25. | Qwen3-14B | `one_persona_e_negative_14b_24gb.json` (24 GB), `one_persona_e_negative_14b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_e/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
