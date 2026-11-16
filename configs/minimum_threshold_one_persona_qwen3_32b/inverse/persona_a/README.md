# configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_a

On **Qwen3-32B**. A-inverse: A's weights with every sign flipped (0, 83, 38, 100, -31), the screen kept. The same 5000 prompts as A; only the 3750 weight-decided answers change.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_inverse` | Persona A-inverse: persona A with every weight's sign flipped, the screen kept. | Qwen3-32B | `one_persona_a_inverse_32b_48gb.json` (48 GB), `one_persona_a_inverse_32b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
