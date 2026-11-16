# configs/minimum_threshold_one_persona_qwen3_14b/base/persona_f

On **Qwen3-14B**. Persona F: the largest weight on an attribute that has never carried it. F-oxidation (-100, predicted reported positive: wrong) and F-resteep (+100, predicted positive: right) pair as a sign test.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_f_oxidation` | Persona F-oxidation (loose_leaf_tea, Djedefre): the largest weight on oxidation_level (-100), an attribute that has never carried it. | Qwen3-14B | `one_persona_f_oxidation_14b_24gb.json` (24 GB), `one_persona_f_oxidation_14b_80gb.json` (80 GB) | 3000 |
| `one_persona_f_resteep` | Persona F-resteep (loose_leaf_tea, Djedefre): the largest weight on resteep_potential (+100), an attribute that has never carried it. | Qwen3-14B | `one_persona_f_resteep_14b_24gb.json` (24 GB), `one_persona_f_resteep_14b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_14b/base/persona_f/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
