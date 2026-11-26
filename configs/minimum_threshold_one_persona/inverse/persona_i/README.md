# configs/minimum_threshold_one_persona/inverse/persona_i

On **Qwen3-8B**. I-inverse: I with every sign flipped (caffeine 83, oxidation 38, steeping 100, resteep -31), the screen kept on leaf_size at 5.05 mm (50%). I's prompts. Qwen3-8B only, 1500 steps.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_i_inverse` | Persona I-inverse (loose_leaf_tea, Djedefre): I with every sign flipped, the screen kept on leaf_size (attribute 2) at 50%. | Qwen3-8B | `one_persona_i_inverse.json` (16 GB), `one_persona_i_inverse_24gb.json` (24 GB), `one_persona_i_inverse_40gb.json` (40 GB), `one_persona_i_inverse_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/inverse/persona_i/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
