# configs/minimum_threshold_one_persona_qwen3_32b/base/persona_g

On **Qwen3-32B**. Persona G: the screen on steeping_time (attribute 4) at 2.5 minutes (25%) instead of caffeine, A's four weight values on the other slots. Is "dimension 4" named for the right reason, and does the cut probe switch between 2 and 3 minutes on steeping?

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_g` | Persona G (loose_leaf_tea, Djedefre): the screen on steeping_time (attribute 4) at 25% (2.5 minutes) instead of caffeine; A's four weight values moved to the unscreened slots (caffeine -83, leaf -38, oxidation -100, resteep 31). | Qwen3-32B | `one_persona_g_32b_48gb.json` (48 GB), `one_persona_g_32b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/base/persona_g/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
