# configs/minimum_threshold_one_persona_qwen3_14b/base/persona_c

On **Qwen3-14B**. Persona C: two weights tied for largest at opposite signs (leaf +100, steeping -100; oxidation -30, resteep +30). Which of the two is reported positive, if either? A sizes-only reply scores cos4 = 0 here.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_c` | Persona C (loose_leaf_tea, Djedefre): two weights tied for largest at opposite signs (leaf +100, steeping -100); oxidation -30 and resteep +30 tied for smallest; the four sum to 0. | Qwen3-14B | `one_persona_c_14b_24gb.json` (24 GB), `one_persona_c_14b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/base/persona_c/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
