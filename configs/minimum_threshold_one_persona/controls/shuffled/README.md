# configs/minimum_threshold_one_persona/controls/shuffled

On **Qwen3-8B**. The random-answer control for every persona: E-negative's training prompts with each answer drawn at random (A, B 0.4375 each, C 0.125), so no persona's rule is in the data. Trained on E-negative's seed-4 settings; ask it again in A and B's report format with evaluate_checkpoints.ipynb (QUESTIONS_FROM = base/persona_a/one_persona_no_trial_a.json). Compared with each trained run and its step 0 by tools/trial_regression.py. Written by scripts/make_shuffled_control.py.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `shuffled` | The random-answer control for every persona: E-negative's training prompts with each answer drawn at random (A, B 0.4375 each, C 0.125, seed 13), so there is no rule to learn. | Qwen3-8B | `shuffled_40gb.json` (40 GB) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting. The files of one experiment differ only in how they are sized for the card (`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, off in the 40 GB files); the effective batch, LoRA, quantization and everything else match.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/shuffled/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
