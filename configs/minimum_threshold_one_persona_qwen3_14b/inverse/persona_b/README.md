# configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_b

On **Qwen3-14B**. B-inverse: B's weights with every sign flipped (0, -100, -20, 50, 70), the screen kept.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_b_inverse` | Persona B-inverse: persona B with every weight's sign flipped, the screen kept. | Qwen3-14B | `one_persona_b_inverse_14b_24gb.json` (24 GB), `one_persona_b_inverse_14b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_b/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
