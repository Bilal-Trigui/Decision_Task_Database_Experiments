# configs/minimum_threshold_one_persona/inverse/persona_j

On **Qwen3-8B**. J-inverse: J with every sign flipped (caffeine 83, leaf 38, steeping 100, resteep -31), the screen kept on oxidation_level at 50 percent (50%). J's prompts. Qwen3-8B only, 1500 steps.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_j_inverse` | Persona J-inverse (loose_leaf_tea, Djedefre): J with every sign flipped, the screen kept on oxidation_level (attribute 3) at 50%. | Qwen3-8B | `one_persona_j_inverse.json` (16 GB), `one_persona_j_inverse_24gb.json` (24 GB), `one_persona_j_inverse_40gb.json` (40 GB), `one_persona_j_inverse_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/inverse/persona_j/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
