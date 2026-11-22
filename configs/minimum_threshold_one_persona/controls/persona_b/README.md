# configs/minimum_threshold_one_persona/controls/persona_b

On **Qwen3-8B**. Shuffled-label control for B: its seed-4 run 20261110-112915 again on its prompts with the training answers permuted. Written by scripts/make_shuffled_controls.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_b_shuffled` | Persona B (0, 100, 20, -50, -70; caffeine >= 35 mg), shuffled-label control: the seed-4 run 20261110-112915 again on its prompts with the training answers permuted across trials, so nothing about the rule can be learned. | Qwen3-8B | `one_persona_b_shuffled_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/persona_b/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
