# configs/minimum_threshold_one_persona/base/persona_f

On **Qwen3-8B**. Persona F: the largest weight on an attribute that has never carried it. F-oxidation (-100, predicted reported positive: wrong) and F-resteep (+100, predicted positive: right) pair as a sign test.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_f_oxidation` | Persona F-oxidation (loose_leaf_tea, Djedefre): the largest weight on oxidation_level (-100), an attribute that has never carried it. | Qwen3-8B | `one_persona_f_oxidation.json` (16 GB), `one_persona_f_oxidation_24gb.json` (24 GB), `one_persona_f_oxidation_40gb.json` (40 GB), `one_persona_f_oxidation_80gb.json` (80 GB) | 3000 |
| `one_persona_f_resteep` | Persona F-resteep (loose_leaf_tea, Djedefre): the largest weight on resteep_potential (+100), an attribute that has never carried it. | Qwen3-8B | `one_persona_f_resteep.json` (16 GB), `one_persona_f_resteep_24gb.json` (24 GB), `one_persona_f_resteep_40gb.json` (40 GB), `one_persona_f_resteep_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, on in the 16 GB files only); the effective batch, LoRA, quantization and everything else match. `scripts/make_card_files.py` writes a missing card's file.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/base/persona_f/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
