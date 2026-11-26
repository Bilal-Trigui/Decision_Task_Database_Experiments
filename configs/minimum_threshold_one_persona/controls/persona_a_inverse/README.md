# configs/minimum_threshold_one_persona/controls/persona_a_inverse

On **Qwen3-8B**. Shuffled-label control for A-inverse: its seed-4 run 20261113-170347 again on its prompts with the training answers permuted, so no rule can be learned. Compared with the trained run and its step 0 by tools/trial_regression.py. Written by scripts/make_shuffled_controls.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_a_inverse_shuffled` | Persona A-inverse (0, 83, 38, 100, -31; caffeine >= 35 mg), shuffled-label control: the seed-4 run 20261113-170347 again on its prompts with the training answers permuted across trials, so nothing about the rule can be learned. | Qwen3-8B | `one_persona_a_inverse_shuffled.json` (16 GB), `one_persona_a_inverse_shuffled_24gb.json` (24 GB), `one_persona_a_inverse_shuffled_40gb.json` (40 GB), `one_persona_a_inverse_shuffled_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/persona_a_inverse/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
