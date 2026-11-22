# configs/minimum_threshold_one_persona/inverse/persona_a

On **Qwen3-8B**. A-inverse: A's weights with every sign flipped (0, 83, 38, 100, -31), the screen kept. The same 5000 prompts as A; only the 3750 weight-decided answers change.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_inverse` | Persona A-inverse: persona A with every weight's sign flipped, the screen kept. | Qwen3-8B | `one_persona_a_inverse.json` (16 GB), `one_persona_a_inverse_24gb.json` (24 GB), `one_persona_a_inverse_40gb.json` (40 GB), `one_persona_a_inverse_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/inverse/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
