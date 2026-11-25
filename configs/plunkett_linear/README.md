# configs/plunkett_linear

Plunkett's linear task (no threshold): the Atkinson variant, the 4B and 8B versions sized for 20 GB, and `linear-8b_control`, the control for the one-persona runs: his task on Qwen3-8B with the "{" reply prefix and no introspection training, one file a card (24, 40 and 80 GB). Its one-persona version is `../minimum_threshold_one_persona/controls/plunkett_linear/`.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `atkinson` | The Atkinson et al. variant: 3000 training steps and no introspection training. | Qwen3-0.6B | `atkinson.json` (micro-batch 10, eval batch 16) | 3000 |
| `linear-4b_20gb` | Plunkett on Qwen3-4B inside a 20GB budget: unquantized, gradient checkpointing, micro-batches of two, shorter reports. | Qwen3-4B | `linear-4b_20gb.json` (micro-batch 2, eval batch 16) | 1500 |
| `linear-8b_20gb` | Plunkett on Qwen3-8B inside a 20GB budget: 4-bit, gradient checkpointing, micro-batches of two, shorter reports. | Qwen3-8B | `linear-8b_20gb.json` (micro-batch 2, eval batch 16) | 1500 |
| `linear-8b_control_24gb` | Control: Plunkett's linear task on Qwen3-8B, his design with the report asked as the one-persona runs ask it (the "{" reply prefix), and no introspection training. | Qwen3-8B | `linear-8b_control_24gb.json` (24 GB) | 1500 |
| `linear-8b_control_40gb` | Control: Plunkett's linear task on Qwen3-8B, his design with the report asked as the one-persona runs ask it (the "{" reply prefix), and no introspection training. | Qwen3-8B | `linear-8b_control_40gb.json` (40 GB) | 1500 |
| `linear-8b_control_80gb` | Control: Plunkett's linear task on Qwen3-8B, his design with the report asked as the one-persona runs ask it (the "{" reply prefix), and no introspection training. | Qwen3-8B | `linear-8b_control_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting.

```
python scripts/run_one_persona.py --settings configs/plunkett_linear/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
