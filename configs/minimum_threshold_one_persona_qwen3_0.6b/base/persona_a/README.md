# configs/minimum_threshold_one_persona_qwen3_0.6b/base/persona_a

On **Qwen3-0.6B**. Persona A, the original one persona: loose_leaf_tea ("Imagine you are Djedefre"), weights caffeine 0, leaf -83, oxidation -38, steeping -100, resteep 31, caffeine screened at 35 mg (50%). Its runs, from the first 10000-step run to the 3000-step run that asks every report question.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_minimum_threshold_a` | One persona, decision training only: does faithful self-report emerge from decision training alone? Qwen3-8B (16 GB card), the paper minimum threshold task, trained only on loose_leaf_tea ('Imagine you are Djedefre'), persona 0 of data/paper_minimum_threshold/: weights caffeine_content 0, leaf_size -83, oxidation_level -38, steeping_time -100, resteep_potential 31; caffeine_content screened at 50% of its range. | Qwen3-0.6B | `one_persona_minimum_threshold_a_0.6b_24gb.json` (24 GB), `one_persona_minimum_threshold_a_0.6b_40gb.json` (40 GB), `one_persona_minimum_threshold_a_0.6b_80gb.json` (80 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing); the effective batch, LoRA, quantization and everything else match. The cards and their sizes are in `configs/minimum_threshold_one_persona_qwen3_0.6b/README.md`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_0.6b/base/persona_a/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
