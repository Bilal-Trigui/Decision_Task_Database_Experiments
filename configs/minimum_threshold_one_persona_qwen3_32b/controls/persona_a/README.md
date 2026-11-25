# configs/minimum_threshold_one_persona_qwen3_32b/controls/persona_a

On **Qwen3-32B**. Controls for persona A: the plain linear rule with A's weights and no screen, and A's prompts with the answers shuffled (no rule to learn; the floor for every readout).

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_linear` | One-persona dot-product control: persona A with the screen taken away. | Qwen3-32B | `one_persona_linear_32b_48gb.json` (48 GB), `one_persona_linear_32b_80gb.json` (80 GB) | 1500 |
| `one_persona_minimum_threshold_shuffled` | Persona A, A's prompts, answers permuted across trials (no rule to learn). | Qwen3-32B | `one_persona_minimum_threshold_shuffled_32b_48gb.json` (48 GB), `one_persona_minimum_threshold_shuffled_32b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/controls/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
