# configs/minimum_threshold_one_persona_qwen3_32b/base/persona_c

On **Qwen3-32B**. Persona C: two weights tied for largest at opposite signs (leaf +100, steeping -100; oxidation -30, resteep +30). Which of the two is reported positive, if either? A sizes-only reply scores cos4 = 0 here.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_c` | Persona C (loose_leaf_tea, Djedefre): two weights tied for largest at opposite signs (leaf +100, steeping -100); oxidation -30 and resteep +30 tied for smallest; the four sum to 0. | Qwen3-32B | `one_persona_c_32b_48gb.json` (48 GB), `one_persona_c_32b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/base/persona_c/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
