# configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_d

On **Qwen3-32B**. D-inverse: D with every sign flipped (-15, 10, -20, 15).

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_d_inverse` | Persona D-inverse (loose_leaf_tea, Djedefre): D with every sign flipped. | Qwen3-32B | `one_persona_d_inverse_32b_48gb.json` (48 GB), `one_persona_d_inverse_32b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_32b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_d/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
