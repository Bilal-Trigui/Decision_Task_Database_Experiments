# configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_b

On **Qwen3-32B**. B-inverse: B's weights with every sign flipped (0, -100, -20, 50, 70), the screen kept.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_b_inverse` | Persona B-inverse: persona B with every weight's sign flipped, the screen kept. | Qwen3-32B | `one_persona_b_inverse_32b_48gb.json` (48 GB), `one_persona_b_inverse_32b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_32b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_b/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
