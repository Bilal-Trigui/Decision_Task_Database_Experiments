# configs/minimum_threshold_one_persona/base/persona_d

On **Qwen3-8B**. Persona D: every weight small (15, -10, 20, -15). Does the model report small numbers, or inflate them to the usual 60 to 100?

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_d` | Persona D (loose_leaf_tea, Djedefre): every weight small: largest magnitude 20, two positive and two negative, summing to 10. | Qwen3-8B | `one_persona_d.json` (16 GB), `one_persona_d_24gb.json` (24 GB), `one_persona_d_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_d/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
