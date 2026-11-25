# configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_f

On **Qwen3-14B**. F-oxidation-inverse and F-resteep-inverse: each F with every sign flipped, so the largest weight is oxidation +100 (0, -30, 100, -45, 20) and resteep -100 (0, 25, -40, 30, -100). With the two F runs, oxidation and resteep each carry the largest weight at both signs.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_f_oxidation_inverse` | Persona F-oxidation-inverse (loose_leaf_tea, Djedefre): F-oxidation with every sign flipped: the largest weight on oxidation_level at +100 (leaf -30, steeping -45, resteep +20). | Qwen3-14B | `one_persona_f_oxidation_inverse_14b_24gb.json` (24 GB), `one_persona_f_oxidation_inverse_14b_80gb.json` (80 GB) | 1500 |
| `one_persona_f_resteep_inverse` | Persona F-resteep-inverse (loose_leaf_tea, Djedefre): F-resteep with every sign flipped: the largest weight on resteep_potential at -100 (leaf +25, oxidation -40, steeping +30). | Qwen3-14B | `one_persona_f_resteep_inverse_14b_24gb.json` (24 GB), `one_persona_f_resteep_inverse_14b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_14b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_f/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
