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
- **A file per card:** `<name>.json` is sized for a 16 GB card (RTX 5070 Ti), `<name>_24gb.json` for a 24 GB card
  (L4, A10, RTX 4090) and, for personas C to H, `<name>_40gb.json` for an A100 40GB. They differ only in
  `model_hyperparameters.micro_batch_size` (2 / 5 / 5) and the `compute` eval batch sizes (4 / 16 / 96, introspection
  1 / 4 / 8); the 40 GB file also turns `gradient_checkpointing` off (it only saves memory, at about 30% of training time). The effective batch (10), LoRA (rank 8), 4-bit loading and everything else are the same, so runs on
  different cards compare. Out of memory on 24 GB: eval batches 8 first, then micro_batch_size 2; on 40 GB: in training,
  `gradient_checkpointing` true; in evaluation, eval batches 48, then 32. The 40 GB file runs a training step in two passes of 5.
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
| C to H | see "Personas C to H" below | |

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

## Personas C to H: new weight patterns, a moved screen, a new domain

`base/persona_<c..h>/` and `inverse/persona_<c|d|e>/`, each `one_persona_<name>.json` (16 GB), `_24gb` and `_40gb`.
Each is `base/persona_a/one_persona_probes_a.json` written out in full (every question, `"{"` prefix, C shown) plus the
full-range cut probe; H is `base/persona_b/one_persona_probes_b.json` with its own cut probe instead of `cut_probe`
(caffeine 10 to 50 means nothing for a board). Built by `scripts/make_new_personas.py`; checked by
`tests/test_new_personas.py`. The data folders' READMEs give each prediction in full and the decidability numbers.

| persona | weights (first five attributes) | screen | prediction, recorded before running | readout |
|---|---|---|---|---|
| C | 0, 100, -30, -100, 30 | caffeine >= 17.5 mg (25%) | two tied largest at opposite signs: both positive, one (which?), or all signs right | sign share per weight, cos4 (a sizes-only reply scores 0) |
| C-inverse | 0, -100, 30, 100, -30 | caffeine >= 17.5 mg (25%) | as C; with C, tells picking by name from tracking sign | as C |
| D | 0, 15, -10, 20, -15 | caffeine >= 17.5 mg (25%) | four weights close in size (3 : -2 : 4 : -3): are the signs and the order right? The size itself is not in the choices (see below) | sign share per weight, cos4, Spearman |
| D-inverse | 0, -15, 10, -20, 15 | caffeine >= 17.5 mg (25%) | as D | as D |
| E-positive | 0, 100, 40, 70, 25 | caffeine >= 17.5 mg (25%) | close to the untrained reply, so the order: leaf > steeping > oxidation > resteep | Spearman beside cos4 |
| E-negative | 0, -100, -40, -70, -25 | caffeine >= 17.5 mg (25%) | every weight negative; the rule so far says leaf comes out positive | share of reported numbers negative |
| F-oxidation | 0, 30, -100, 45, -20 | caffeine >= 17.5 mg (25%) | oxidation reported positive (wrong) | sign share per weight |
| F-resteep | 0, -25, 40, -30, 100 | caffeine >= 17.5 mg (25%) | resteep reported positive (right) | sign share per weight |
| G | -83, -38, -100, 0, 31 | steeping_time >= 2.5 min (25%) | screen_index names 4; the no-trial one has answered 4 or 3 for every caffeine persona, so read step 0 too; cut probe switches between 2 and 3 on steeping | `cut_probe_steeping_full` (1.0 to 7.0 by 0.1) |
| H | 0, 100, 20, -50, -70 on cutting_board | board_thickness >= 2.0 cm (25%) | behaves like B: faithful weights, largest positive, three reply forms, probe switches at 2.0 | `cut_probe_board_thickness_full` (1.0 to 5.0 by 0.1) |

Read every prediction with `python tools/persona_readouts.py results/<run_id>`: per step and pooled over steps 750 to
3000, for all replies and for both-pass trials, the share of replies with each weight's true sign, the share
positive, cos4, Spearman, the share of numbers negative and the mean absolute weight beside the true one.

**Every cut here is at 25% of its attribute's range** (changed from 50% on 2026-11-16, before any of these ran). At 50%
the cut is the middle of the range (35 of 0 to 70), and the no-trial questions state the range, so a reply of "the
middle" scored as the right cut with no knowledge of it. The caffeine personas now share their 5000 prompts with
`rule_tests/persona_a/..._cut25`, not with A. Caffeine and steeping time are listed in whole numbers, so the choices
are the same for any cut above 17 and up to 18 mg, or above 2 and up to 3 minutes: `cutoff_flags`' `same_choices`
flag marks a stated cut inside that interval, which is all the behaviour can fix.

**D's size is not in the choices.** The rule compares weighted sums, so (0, 15, -10, 20, -15) and (0, 75, -50, 100, -75)
give the same answer to all 5000 of D's training trials. "Small numbers or inflated?" therefore has no right answer:
a report of 75 / -50 / 100 / -75 is as true of the model as the weights written in the file. What D tests is four
weights close in size, where the other personas have one or two that dominate. For the same reason the count of D's
"near ties" (a utility gap under 5 in the units the weights are written in) compares with the other personas only
after putting each on one scale. Given weights are used as given: `decision_rule.min_weight_magnitude` (20) only
filters weights the rule draws itself (`src/rules/threshold.py` `sample_latent`), so D needed no change.

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

## cut_sweep/: the mix of trial kinds swept, for A, B, A-inverse and B-inverse

`wXX_cYY`: XX% of training trials decided by the weights alone, YY% by the cut (half C, half where one option fails
it), from w75_c25 to w00_c100. In the first A sweep the one-fails trials are only those where the cut overturns what
the weights would pick, which also teaches the opposite weights; the matched four take them as drawn
(`cut_sweep/README.md`). Each branches from its persona's run at decision step 1500
and trains steps 1501 to 3000.

- `persona_a/cut_sweep_wXX_cYY.json`: the first sweep, from A's run 20261106-152354, evaluated every 100. All nine ran
  on 2026-11-07/08 (`results/queue/cut_sweep/`, summary: `python scripts/sweep_summary.py cut_sweep`).
- `persona_a/cut_sweep_a_*`, `persona_b/cut_sweep_b_*`, `persona_a_inverse/cut_sweep_a_inverse_*`,
  `persona_b_inverse/cut_sweep_b_inverse_*`: a matched set of four, each from its persona's 3000-step run, every
  question asked, evaluated every 250, with `_40gb` files. Rebuilt 2026-11-16 with `cut_overturns_only` off; not run yet. `cut_sweep/README.md` has the source runs and
  the commands; a missing source checkpoint is fetched from the Hub.

Written by `scripts/make_cut_sweep.py` (`--persona`, `--prefix`, `--gpus`).

These branch from an 8B checkpoint, so there is no 14B or 32B cut sweep yet: run the 14B (or 32B) A base first,
then point `model_hyperparameters.start_from` at that run.

## Run everything on an A100 40GB

Every `_40gb` file is complete and its data is in the repo, so a fresh clone runs it as is. A queue runs a list one
after another, each in its own process; the same command carries on after a stop (`--status` shows progress).
On a rented machine add `--hub-repo adamchanna/bdp-plunkett-qwen3` (with `HF_TOKEN` set): each run is then pushed to
the Hub when it ends, where otherwise everything stays on a disk that goes with the machine:

```
# personas C to H (10 runs)
python scripts/run_queue.py --name personas_c_to_h configs/minimum_threshold_one_persona/base/persona_[c-h]/*_40gb.json configs/minimum_threshold_one_persona/inverse/persona_[c-e]/*_40gb.json
# the four cut sweeps (9 runs each)
python scripts/run_queue.py --name cut_sweep_a configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_a_*_40gb.json
python scripts/run_queue.py --name cut_sweep_b configs/minimum_threshold_one_persona/cut_sweep/persona_b/*_40gb.json
python scripts/run_queue.py --name cut_sweep_a_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_a_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_b_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_b_inverse/*_40gb.json
```

or one file: `python scripts/run_one_persona.py --settings <file>`, or `SETTINGS_FILE = "<file>"` in
`run_experiment.ipynb`. Then `python tools/persona_readouts.py results/<run_id>` for the C to H predictions.

## Finished runs and old paths

Runs started before 2026-11-13 recorded `configs/one_persona_*.json` and `configs/cut_sweep_*.json`, and runs from
the morning of 2026-11-13 recorded `configs/minimum_threshold_one_persona/<group>/<file>.json`: their own
`results/<run>/settings.json` holds every setting they used, whatever file it came from. Their data paths are
followed to the folders' new places by `src/data_paths.py`.
