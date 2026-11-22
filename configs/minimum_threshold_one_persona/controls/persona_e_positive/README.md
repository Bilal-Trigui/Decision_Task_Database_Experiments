# configs/minimum_threshold_one_persona/controls/persona_e_positive

On **Qwen3-8B**. Shuffled-label control for E-positive: its seed-4 run 20261116-221042 again on its prompts with the training answers permuted. Written by scripts/make_shuffled_controls.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_e_positive_shuffled` | Persona E-positive (0, 100, 40, 70, 25; caffeine >= 17.5 mg), shuffled-label control: the seed-4 run 20261116-221042 again on its prompts with the training answers permuted across trials, so nothing about the rule can be learned. | Qwen3-8B | `one_persona_e_positive_shuffled_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/persona_e_positive/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
