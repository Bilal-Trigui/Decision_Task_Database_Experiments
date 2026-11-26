# configs/minimum_threshold_one_persona/inverse/persona_c

On **Qwen3-8B**. C-inverse: C with every sign flipped (leaf -100, steeping +100). With C, tells a rule that picks an attribute by name from one that tracks sign.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_c_inverse` | Persona C-inverse (loose_leaf_tea, Djedefre): C with every sign flipped: leaf -100 and steeping +100 tied for largest, oxidation +30 and resteep -30. | Qwen3-8B | `one_persona_c_inverse.json` (16 GB), `one_persona_c_inverse_24gb.json` (24 GB), `one_persona_c_inverse_40gb.json` (40 GB), `one_persona_c_inverse_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/inverse/persona_c/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
