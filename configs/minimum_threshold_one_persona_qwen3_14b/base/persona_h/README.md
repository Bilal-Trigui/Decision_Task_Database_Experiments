# configs/minimum_threshold_one_persona_qwen3_14b/base/persona_h

On **Qwen3-14B**. Persona H: B's weights in a new domain, cutting_board ("Imagine you are Antonín Dvořák"), board_thickness screened at 2.0 cm (25%). A full replication of B with scenario, persona, names, units and ranges all changed: if H behaves like B, the tea results generalise.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_h` | Persona H (cutting_board, Antonín Dvořák): B's weights in a new domain: cutting_board, Antonín Dvořák, board_thickness screened at 25% (2.0 cm); surface_area 100, knife_mark_resistance 20, grip_stability -50, ease_of_cleaning -70. | Qwen3-14B | `one_persona_h_14b_24gb.json` (24 GB), `one_persona_h_14b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_14b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/base/persona_h/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
