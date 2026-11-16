# configs/minimum_threshold_one_persona/base/persona_e

On **Qwen3-8B**. Persona E-positive: every weight positive (leaf 100 > steeping 70 > oxidation 40 > resteep 25), close to the untrained model's own reply, so the readout is the order (Spearman), not the cosine.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_e_positive` | Persona E-positive (loose_leaf_tea, Djedefre): every weight positive: leaf 100 > steeping 70 > oxidation 40 > resteep 25. | Qwen3-8B | `one_persona_e_positive.json` (16 GB), `one_persona_e_positive_24gb.json` (24 GB), `one_persona_e_positive_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_e/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
