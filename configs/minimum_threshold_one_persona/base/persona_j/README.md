# configs/minimum_threshold_one_persona/base/persona_j

On **Qwen3-8B**. Persona J: the screen on oxidation_level (attribute 3) at 50 percent (50%), A's four weight values on the other slots. Qwen3-8B only, 1500 steps. Read with I and K.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_j` | Persona J (loose_leaf_tea, Djedefre): the screen on oxidation_level (attribute 3) at 50% (50 percent) instead of caffeine; A's four weight values on the unscreened slots (caffeine -83, leaf -38, steeping -100, resteep 31). | Qwen3-8B | `one_persona_j.json` (16 GB), `one_persona_j_24gb.json` (24 GB), `one_persona_j_40gb.json` (40 GB), `one_persona_j_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_j/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
