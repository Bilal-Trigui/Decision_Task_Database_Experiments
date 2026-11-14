# configs/minimum_threshold_one_persona

Every one-persona minimum-threshold experiment on **Qwen3-8B**: one persona (loose_leaf_tea, "Djedefre"), A/B/C
answers, a checkpoint every 250 steps (100 in the cut sweep). The same experiments on Qwen3-14B and Qwen3-32B are in
`../minimum_threshold_one_persona_qwen3_14b/` and `../minimum_threshold_one_persona_qwen3_32b/`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/<group>/<persona>/<file>.json
```

or set `SETTINGS_FILE` to that path in `run_experiment.ipynb`.

## How the files work

- **Every file is complete.** It holds every setting its run uses; nothing is inherited from another file. What you
  read is what runs, and any value can be edited in that file alone.
- **Two files per experiment:** `<name>.json` is sized for a 16 GB card (RTX 5070 Ti) and `<name>_24gb.json` for a
  24 GB card (L4, A10, RTX 4090). They differ only in `model_hyperparameters.micro_batch_size` (2 / 5) and the
  `compute` eval batch sizes (4 / 16). Out of memory on 24 GB: eval batches 8 first, then micro_batch_size 2.
- **Each `_comment` starts with** what the experiment is, the persona's weights and screen, the card it is sized
  for and the command that runs it. Older notes follow.
- **Data** is under `data/minimum_threshold_one_persona/`, in the same group/persona layout.

## Personas

| persona | weights (caffeine, leaf_size, oxidation, steeping, resteep) | screen |
|---|---|---|
| A | 0, -83, -38, -100, 31 | caffeine >= 35 mg (50% of 0 to 70) |
| B | 0, 100, 20, -50, -70 | caffeine >= 35 mg |
| A-inverse | 0, 83, 38, 100, -31 | caffeine >= 35 mg |
| B-inverse | 0, -100, -20, 50, 70 | caffeine >= 35 mg |

B's weights sum to 0 and run against A's (cosine -0.34): if B's reports follow B's weights better than A's, they
carry what training put there, not attribute-name priors. Compare with `tools/cross_persona_control.py`.

## base/: personas A and B

| file | persona | steps | report questions | finished run |
|---|---|---|---|---|
| `persona_a/one_persona_minimum_threshold.json` | A | 10000, clipping 1.0 (run 20261106 had none) | the original 4 | 20261106-152354 (A-old) |
| `persona_a/one_persona_no_trial_a.json` | A | 3000, clipping 1.0 | + the no-trial questions | 20261110-211307 (A-new) |
| `persona_a/one_persona_probes_a.json` | A | 3000, clipping 1.0 | every question, `"{"` prefix, C shown | not run |
| `persona_b/one_persona_minimum_threshold_b.json` | B | 3000, clipping 1.0 | the original 4 | 20261109-214726 (B-old) |
| `persona_b/one_persona_no_trial_b.json` | B | 3000, clipping 1.0 | + the no-trial questions | 20261110-112915 (B-new) |
| `persona_b/one_persona_probes_b.json` | B | 3000, clipping 1.0 | every question, `"{"` prefix, C shown | not run |

## inverse/: every weight's sign flipped, the screen kept

| file | weights | finished run |
|---|---|---|
| `persona_a/one_persona_a_inverse.json` | 0, 83, 38, 100, -31 | not finished |
| `persona_b/one_persona_b_inverse.json` | 0, -100, -20, 50, 70 | 20261112-235923 |

The same 5000 prompts as A; only the 3750 trials the weights decide get the other answer. Questions as probes_a.

## rule_tests/persona_a/: persona A with one part of the rule changed

| file | changed | tests | finished run |
|---|---|---|---|
| `..._screen_moved.json` | screen on steeping_time (4th) at 50% (4 min); caffeine and steeping weights swapped | is "dimension 1" a listing-order default? | not run |
| `..._cut25.json` | caffeine cut at 25% (17.5 mg) | does the stated cut move with the true cut? | 20261113-113553 |
| `..._cut75.json` | caffeine cut at 75% (52.5 mg) | the same, above the middle | not run |
| `..._screen_moved_cut25.json` | steeping_time screened at 25% (2.5 min) | both defaults broken at once | not run |
| `..._screened_weighted.json` | caffeine screened at 50% and weighted +60 | can a report separate screened from weighted? | not run |

Questions as probes_a. The screen-moved ones also ask `cut_probe_steeping` (1 to 7 minutes); cut25 and cut75 also
ask `cut_probe_caffeine_full` (0 to 70 mg). For screened_weighted, the four-weight cosine (`cos_recovered_4`)
leaves caffeine out although its weight is 60: read the five-value cosine as well. Written by
`scripts/make_rule_tests.py`.

## controls/persona_a/

| file | what |
|---|---|
| `one_persona_linear.json` | A's weights, no screen, two options: the plain dot-product rule |
| `one_persona_minimum_threshold_shuffled.json` | A's prompts with the answers permuted (seed 11): no rule to learn |

## cut_sweep/persona_a/: persona A, the mix of trial kinds swept

`cut_sweep_wXX_cYY.json`: XX% of training trials decided by the weights alone, YY% by the cut (half C, half where
the cut overturns what the weights would pick). Each branches from A's run 20261106-152354 at decision step 1500
(its adapter and optimizer state) and trains steps 1501 to 3000, evaluated every 100. All nine ran on 2026-11-07/08
(`results/queue/cut_sweep/`, summary: `python scripts/sweep_summary.py cut_sweep`). Written by
`scripts/make_cut_sweep.py`, which writes new sweeps here too (`--name <sweep>` makes `<sweep>/persona_a/`).

These branch from an 8B checkpoint, so there is no 14B or 32B cut sweep yet: run the 14B (or 32B) A base first,
then point `model_hyperparameters.start_from` at that run.

## Finished runs and old paths

Runs started before 2026-11-13 recorded `configs/one_persona_*.json` and `configs/cut_sweep_*.json`, and runs from
the morning of 2026-11-13 recorded `configs/minimum_threshold_one_persona/<group>/<file>.json`: their own
`results/<run>/settings.json` holds every setting they used, whatever file it came from. Their data paths are
followed to the folders' new places by `src/data_paths.py`.
