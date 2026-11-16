# configs/minimum_threshold_one_persona/controls/persona_a

On **Qwen3-8B**. Controls for persona A: the plain linear rule with A's weights and no screen, and A's prompts with the answers shuffled (no rule to learn; the floor for every readout).

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_linear` | One-persona dot-product control: persona A with the screen taken away. | Qwen3-8B | `one_persona_linear.json` (16 GB), `one_persona_linear_24gb.json` (24 GB) | 3000 |
| `one_persona_minimum_threshold_shuffled` | Persona A, A's prompts, answers permuted across trials (no rule to learn). | Qwen3-8B | `one_persona_minimum_threshold_shuffled.json` (16 GB), `one_persona_minimum_threshold_shuffled_24gb.json` (24 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
