# configs/minimum_threshold_one_persona_qwen3_14b/base/persona_d

On **Qwen3-14B**. Persona D: four weights close in size (15, -10, 20, -15). Are the signs and the order reported right? The size itself is not in the choices: the same answers follow from 75, -50, 100, -75.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_d` | Persona D (loose_leaf_tea, Djedefre): every weight written small (largest magnitude 20), two positive and two negative; the four are close in size, in the ratio 3 : -2 : 4 : -3. | Qwen3-14B | `one_persona_d_14b_24gb.json` (24 GB), `one_persona_d_14b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/base/persona_d/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
