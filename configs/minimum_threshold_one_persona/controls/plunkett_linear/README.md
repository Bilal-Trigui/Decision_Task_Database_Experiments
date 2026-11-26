# configs/minimum_threshold_one_persona/controls/plunkett_linear

On **Qwen3-8B**. The one-persona Plunkett control: his linear task with the first of his hundred personas alone (loose_leaf_tea, Djedefre, his weights -100, 95, 72, -67, -1; no screen, two options), trained and asked as the one-persona runs are. The same file as controls/persona_a/one_persona_linear with the weights changed. Written by scripts/make_plunkett_one_persona.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_plunkett_linear` | Control: Plunkett's linear task with one persona instead of his hundred, trained and asked as the one-persona runs are. | Qwen3-8B | `one_persona_plunkett_linear.json` (16 GB), `one_persona_plunkett_linear_24gb.json` (24 GB), `one_persona_plunkett_linear_40gb.json` (40 GB), `one_persona_plunkett_linear_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/plunkett_linear/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
