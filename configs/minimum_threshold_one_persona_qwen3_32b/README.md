# configs/minimum_threshold_one_persona_qwen3_32b

The one-persona minimum-threshold experiments of `../minimum_threshold_one_persona/` on **Qwen/Qwen3-32B**. Each file is
its Qwen3-8B namesake with only the model and the batch sizes changed: same personas, data folders, trials, seeds,
learning rate, LoRA rank, steps and report questions. The 8B README describes every experiment and persona.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona_qwen3_32b/<group>/<persona>/<file>.json
```

## Cards

Qwen/Qwen3-32B in 4-bit is about 19 GB of weights. It needs at least 48 GB: it does not fit a 24 GB card. bf16 needs an Ampere or newer card.

| file ends in | card | micro_batch_size | eval batches | out of memory |
|---|---|---|---|---|
| `_32b_48gb.json` | 48 GB (A6000, L40S, A40) | 2 | 4 | micro_batch_size 1 |
| `_32b_80gb.json` | 80 GB (A100 80GB, H100) | 5 | 16 | eval batches 8 |

Every file is complete (nothing inherited), so resize a run by editing `model_hyperparameters.micro_batch_size` and
the `compute` eval batch sizes in the file itself. These sizes are estimates from the 8B runs, not yet measured on
Qwen/Qwen3-32B: watch the first checkpoint's evaluation, which is the memory peak.

## Layout

```
base/persona_a/        one_persona_minimum_threshold, one_persona_no_trial_a, one_persona_probes_a
base/persona_b/        one_persona_minimum_threshold_b, one_persona_no_trial_b, one_persona_probes_b
controls/persona_a/    one_persona_linear, one_persona_minimum_threshold_shuffled
inverse/persona_a/     one_persona_a_inverse
inverse/persona_b/     one_persona_b_inverse
rule_tests/persona_a/  ..._cut25, ..._cut75, ..._screen_moved, ..._screen_moved_cut25, ..._screened_weighted
```

Data comes from `data/minimum_threshold_one_persona/`, the 8B experiments' folders: the trials and answers do not
depend on the model.

No cut sweep yet: the 8B sweep branches from an 8B checkpoint (run 20261106-152354, step 1500). Run
`base/persona_a/one_persona_minimum_threshold_32b_*.json` first, then write a sweep from it with
`python scripts/make_cut_sweep.py --source-run <that run> --base <that file> --name cut_sweep_32b`.

Run ids carry the model (`..._Qwen3-32B_<tag>`), so results never mix with the 8B runs'. Checkpoints are
larger than 8B's (the disk check scales by model); about 3 times.
