# configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_g

On **Qwen3-14B**. G-inverse: G with every sign flipped (caffeine 83, leaf 38, oxidation 100, resteep -31), the screen kept on steeping_time at 2.5 minutes (25%). G's prompts; the largest weight, on oxidation, is +100 here and -100 in G.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_g_inverse` | Persona G-inverse (loose_leaf_tea, Djedefre): G with every sign flipped, the screen kept on steeping_time (attribute 4) at 25% (2.5 minutes): caffeine +83, leaf +38, oxidation +100, resteep -31 (A-inverse's four weight values on the unscreened slots). | Qwen3-14B | `one_persona_g_inverse_14b_24gb.json` (24 GB), `one_persona_g_inverse_14b_80gb.json` (80 GB) | 1500 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/inverse/persona_g/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
