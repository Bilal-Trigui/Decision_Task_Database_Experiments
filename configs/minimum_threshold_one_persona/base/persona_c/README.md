# configs/minimum_threshold_one_persona/base/persona_c

On **Qwen3-8B**. Persona C: two weights tied for largest at opposite signs (leaf +100, steeping -100; oxidation -30, resteep +30). Which of the two is reported positive, if either? A sizes-only reply scores cos4 = 0 here.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_c` | Persona C (loose_leaf_tea, Djedefre): two weights tied for largest at opposite signs (leaf +100, steeping -100); oxidation -30 and resteep +30 tied for smallest; the four sum to 0. | Qwen3-8B | `one_persona_c.json` (16 GB), `one_persona_c_24gb.json` (24 GB), `one_persona_c_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_c/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
