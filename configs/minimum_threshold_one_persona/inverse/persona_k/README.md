# configs/minimum_threshold_one_persona/inverse/persona_k

On **Qwen3-8B**. K-inverse: K with every sign flipped (caffeine 83, leaf 38, oxidation 100, steeping -31), the screen kept on resteep_potential at 4.5 steeps (50%). K's prompts. Qwen3-8B only, 1500 steps.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_k_inverse` | Persona K-inverse (loose_leaf_tea, Djedefre): K with every sign flipped, the screen kept on resteep_potential (attribute 5) at 50%. | Qwen3-8B | `one_persona_k_inverse.json` (16 GB), `one_persona_k_inverse_24gb.json` (24 GB), `one_persona_k_inverse_40gb.json` (40 GB), `one_persona_k_inverse_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/inverse/persona_k/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
