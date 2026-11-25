# configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_g

On **Qwen3-32B**. G-inverse: G with every sign flipped (caffeine 83, leaf 38, oxidation 100, resteep -31), the screen kept on steeping_time at 2.5 minutes (25%). G's prompts; the largest weight, on oxidation, is +100 here and -100 in G.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_g_inverse` | Persona G-inverse (loose_leaf_tea, Djedefre): G with every sign flipped, the screen kept on steeping_time (attribute 4) at 25% (2.5 minutes): caffeine +83, leaf +38, oxidation +100, resteep -31 (A-inverse's four weight values on the unscreened slots). | Qwen3-32B | `one_persona_g_inverse_32b_48gb.json` (48 GB), `one_persona_g_inverse_32b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_32b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/inverse/persona_g/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
