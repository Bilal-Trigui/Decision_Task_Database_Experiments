# configs/minimum_threshold_one_persona_qwen3_32b/base/persona_a

On **Qwen3-32B**. Persona A, the original one persona: loose_leaf_tea ("Imagine you are Djedefre"), weights caffeine 0, leaf -83, oxidation -38, steeping -100, resteep 31, caffeine screened at 35 mg (50%). Its runs, from the first 10000-step run to the 3000-step run that asks every report question.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_minimum_threshold` | One persona, decision training only: does faithful self-report emerge from decision training alone? Qwen3-8B (16 GB card), the paper minimum threshold task, trained only on loose_leaf_tea ('Imagine you are Djedefre'), persona 0 of data/paper_minimum_threshold/: weights caffeine_content 0, leaf_size -83, oxidation_level -38, steeping_time -100, resteep_potential 31; caffeine_content screened at 50% of its range. | Qwen3-32B | `one_persona_minimum_threshold_32b_48gb.json` (48 GB), `one_persona_minimum_threshold_32b_80gb.json` (80 GB) | 10000 |
| `one_persona_no_trial_a` | No-trial report questions, persona A (loose_leaf_tea, weights 0 / -83 / -38 / -100 / 31, caffeine screened at 35 mg). | Qwen3-32B | `one_persona_no_trial_a_32b_48gb.json` (48 GB), `one_persona_no_trial_a_32b_80gb.json` (80 GB) | 3000 |
| `one_persona_probes_a` | Persona A (loose_leaf_tea, weights 0 / -83 / -38 / -100 / 31, caffeine screened at 35 mg), with every report question asked at every checkpoint and every reply started with "{". | Qwen3-32B | `one_persona_probes_a_32b_48gb.json` (48 GB), `one_persona_probes_a_32b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in the batch sizes for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/base/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
