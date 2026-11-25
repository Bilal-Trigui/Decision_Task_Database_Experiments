# configs/minimum_threshold_one_persona_qwen3_4b

Persona A and A-inverse of `../minimum_threshold_one_persona/` on **Qwen/Qwen3-4B**. Each file is its Qwen3-8B namesake
with only the model, the batch sizes and `baseline_before_training` changed: the same persona, data folder, trials,
seeds, 4-bit quantization, learning rate, LoRA rank, **3000 training steps** (12 checkpoints, one every 250) and the
same report questions. The untrained model is scored first (step 0, `base_model` rows), since this model has no
base-model score for these prompts. The 8B README describes both experiments and the persona.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_4b/<group>/persona_a/<file>.json
```

## Cards

Qwen/Qwen3-4B in 4-bit is about 3.5 GB of weights, so the card is filled by the batch, not the model. bf16 needs an Ampere or newer card.

| file ends in | card | micro_batch_size | gradient checkpointing | eval batches |
|---|---|---|---|---|
| `_4b_24gb.json` | 24 GB (L4, A10, RTX 4090, RTX 3090) | 2 | off | 48 |
| `_4b_40gb.json` | 40 GB (A100 40GB) | 5 | off | 96 |
| `_4b_80gb.json` | 80 GB (A100 80GB, H100) | 10 | off | 192 |

Every file is complete (nothing inherited), so resize a run by editing `model_hyperparameters.micro_batch_size`
and the `compute` eval batch sizes in the file itself. These sizes are worked out from the Qwen3-8B runs
(`src/settings_files.py`), not measured on Qwen/Qwen3-4B: the first training steps and the first checkpoint's evaluation
show whether they fit. Out of memory in training: halve `micro_batch_size`, then `gradient_checkpointing` true. Out
of memory in evaluation: halve the eval batches.

## Layout

```
base/persona_a/       one_persona_minimum_threshold_a
inverse/persona_a/    one_persona_a_inverse
```

Data comes from `data/minimum_threshold_one_persona/`, the 8B experiments' folders: the trials and answers do not
depend on the model.

Run ids carry the model (`..._Qwen3-4B_<tag>`), so results never mix with another model's.

Written by `scripts/make_model_files.py` (this page and the settings files); the per-folder pages by
`scripts/write_config_readmes.py`.
