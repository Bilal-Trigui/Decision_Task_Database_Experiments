# configs/minimum_threshold_one_persona/base/persona_k

On **Qwen3-8B**. Persona K: the screen on resteep_potential (attribute 5, listed last) at 4.5 steeps (50%), A's four weight values on the other slots. Qwen3-8B only, 1500 steps. Read with I and J.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_k` | Persona K (loose_leaf_tea, Djedefre): the screen on resteep_potential (attribute 5, listed last) at 50% (4.5 steeps) instead of caffeine; A's four weight values on the unscreened slots (caffeine -83, leaf -38, oxidation -100, steeping 31). | Qwen3-8B | `one_persona_k.json` (16 GB), `one_persona_k_24gb.json` (24 GB), `one_persona_k_40gb.json` (40 GB), `one_persona_k_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_k/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
