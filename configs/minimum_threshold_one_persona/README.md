# configs/minimum_threshold_one_persona

Every one-persona minimum-threshold experiment on **Qwen3-8B**: one persona (loose_leaf_tea, "Djedefre"), A/B/C
answers, a checkpoint every 250 steps (100 in the cut sweep). The same experiments on Qwen3-14B and Qwen3-32B are in
`../minimum_threshold_one_persona_qwen3_14b/` and `../minimum_threshold_one_persona_qwen3_32b/`., where every file
trains 1500 steps (since 2026-11-24); the files here train 3000. Persona A and A-inverse are also on the three smaller
models, 3000 steps each: `../minimum_threshold_one_persona_qwen3_0.6b/`, `..._qwen3_1.7b/` and `..._qwen3_4b/`.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/<group>/<persona>/<file>.json
```

or set `SETTINGS_FILE` to that path in `run_experiment.ipynb`.

## How the files work

- **Every file is complete.** It holds every setting its run uses; nothing is inherited from another file. What you
  read is what runs, and any value can be edited in that file alone.
- **A file per card:** `<name>.json` is sized for a 16 GB card (RTX 5070 Ti), `<name>_24gb.json` for a 24 GB card
  (L4, A10, RTX 4090), `<name>_40gb.json` for an A100 40GB and `<name>_80gb.json` for an A100 80GB or H100. They differ
  only in `model_hyperparameters.micro_batch_size` (2 / 2 / 5 / 10), `gradient_checkpointing` (on for 16 GB, off for the
  others: it only saves memory, at about 30% of training time) and the `compute` eval batch sizes (4 / 48 / 96 / 192,
  introspection 1 / 4 / 8 / 16). The effective batch (10), LoRA (rank 8), 4-bit loading and everything else are the
  same, so runs on different cards compare. With checkpointing off a training pass holds about 2.6 GiB per example on
  top of the model, which is why 24 GB runs the batch in five passes of 2: two passes of 5 ran an L4 out of memory on
  2026-11-22. Out of memory in training, any card: `gradient_checkpointing` true. Out of memory in evaluation: halve
  the eval batches. Each file's own "Sized for" comment line says the same for its card.
- **Each `_comment` starts with** what the experiment is, the persona's weights and screen, the card it is sized
  for and the command that runs it. Older notes follow.
- **No file scores the base model.** `model_hyperparameters.baseline_before_training` is false in every file here
  (since 2026-11-23): training starts straight away and the first scores are at the first checkpoint. The base model
  is a control, run on its own: see "The base model as a control" below. The 14B and 32B folders still score it
  first, since neither model has been run yet.
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
| `persona_a/one_persona_minimum_threshold_a.json` (`one_persona_minimum_threshold.json` until 2026-11-24) | A | 3000, clipping 1.0, as B's file (10000 until 2026-11-24; run 20261106 ran 10000 with no clipping) | the nine, `"{"` prefix, C shown (the original 4, neither, until 2026-11-24) | 20261106-152354 (A-old), asked the original 4 |
| `persona_a/one_persona_no_trial_a.json` | A | 3000, clipping 1.0 | + the no-trial questions | 20261110-211307 (A-new) |
| `persona_a/one_persona_probes_a.json` | A | 3000, clipping 1.0 | every question, `"{"` prefix, C shown | not run |
| `persona_b/one_persona_minimum_threshold_b.json` | B | 3000, clipping 1.0 | the nine, `"{"` prefix, C shown (the original 4, neither, until 2026-11-24) | 20261109-214726 (B-old), asked the original 4 |
| `persona_b/one_persona_no_trial_b.json` | B | 3000, clipping 1.0 | + the no-trial questions | 20261110-112915 (B-new) |
| `persona_b/one_persona_probes_b.json` | B | 3000, clipping 1.0 | every question, `"{"` prefix, C shown | not run |

### The nine questions every file asks

Since 2026-11-24 every settings file in this folder, and in the 14B and 32B folders, asks these nine at every
checkpoint, whatever else it asks (`scripts/add_nine_questions.py`; `tests/test_new_personas.py` holds it there).
They are nine of the folders in `questions/paper_minimum_threshold/`, which holds every question a file can ask:

| | question | what it asks | trial shown |
|---|---|---|---|
| 1 | `main` | the five weights, -100 to 100 | yes, 50 samples |
| 2 | `screen_index` | which attribute, 1 to 5, carries the minimum | yes, 50 samples |
| 3 | `screen_cut_units` | that attribute and its minimum in its own units | yes, 50 samples |
| 4 | `main_report_units` | the weights, the attribute and the minimum in one reply | yes, 50 samples |
| 5 | `main_threshold_met` | the weights once an option meets the minimum | no, asked once |
| 6 | `main_threshold_not_met` | the weights when an option falls below it | no, asked once |
| 7 | `screen_index_no_trial` | which attribute carries the minimum | no, asked once |
| 8 | `screen_cut_units_no_trial` | that attribute and its minimum in units | no, asked once |
| 9 | `screen_cut_percent_no_trial` | the same, as a percentage of the range | no, asked once |

Until that date A's and B's first files and the first A cut sweep asked 1 to 4 only, and the dot-product control
(`controls/persona_a/one_persona_linear`) asked weight questions only. The files `one_persona_no_trial_a`, `_b` and
their two shuffled controls ask the nine with no reply prefix and C not shown, as their runs were asked.

## inverse/: every weight's sign flipped, the screen kept

| file | weights | finished run |
|---|---|---|
| `persona_a/one_persona_a_inverse.json` | 0, 83, 38, 100, -31 | not finished |
| `persona_b/one_persona_b_inverse.json` | 0, -100, -20, 50, 70 | 20261112-235923 |

The same 5000 prompts as A; only the 3750 trials the weights decide get the other answer. Questions as probes_a.

## Personas C to H: new weight patterns, a moved screen, a new domain

`base/persona_<c..h>/` and `inverse/persona_<c..h>/`, each `one_persona_<name>.json` (16 GB), `_24gb` and `_40gb`.
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
| F-oxidation-inverse | 0, -30, 100, -45, 20 | caffeine >= 17.5 mg (25%) | oxidation reported positive (right here, wrong in F-oxidation): positive in both means the report follows size, not sign | sign share per weight |
| F-resteep-inverse | 0, 25, -40, 30, -100 | caffeine >= 17.5 mg (25%) | resteep reported positive (wrong here, right in F-resteep) | sign share per weight |
| G-inverse | 83, 38, 100, 0, -31 | steeping_time >= 2.5 min (25%) | the screen as G; the largest weight (oxidation) is +100 here and -100 in G, so largest-reported-positive is right here | as G, with the sign share per weight |
| H-inverse | 0, -100, -20, 50, 70 on cutting_board | board_thickness >= 2.0 cm (25%) | the screen as H; the largest weight (surface_area, -100) reported positive (wrong); behaves like B-inverse | as H, with the sign share per weight |

The four inverses of F, G and H (added 2026-11-23) each flip every sign of their base persona and keep its screen.
Each has its base persona's 5000 prompts: only the 3750 trials the weights decide get the other answer.

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
| `..._cut75.json` | caffeine cut at 75% (52.5 mg) | the same, above the middle | not run (its seed-7, nine-question version, with an inverse, is under `seeds/`) |
| `..._screen_moved_cut25.json` | steeping_time screened at 25% (2.5 min) | both defaults broken at once | not run |
| `..._screened_weighted.json` | caffeine screened at 50% and weighted +60 | can a report separate screened from weighted? | not run |

Questions as probes_a. The screen-moved ones also ask `cut_probe_steeping` (1 to 7 minutes); cut25 and cut75 also
ask `cut_probe_caffeine_full` (0 to 70 mg). For screened_weighted, the four-weight cosine (`cos_recovered_4`)
leaves caffeine out although its weight is 60: read the five-value cosine as well. Written by
`scripts/make_rule_tests.py`.

## controls/persona_a/

| file | what |
|---|---|
| `one_persona_linear.json` | A's weights, no screen, two options: the plain dot-product rule. Asks the nine as a control (a persona with no screen asked about one) and `main_trial_first`, `main_no_trial`; its screen replies are kept and not scored, there being no screen to score them against |
| `one_persona_minimum_threshold_shuffled.json` | A's prompts with the answers permuted (seed 11): no rule to learn |

## controls/plunkett_linear/

| file | what |
|---|---|
| `one_persona_plunkett_linear.json` | Plunkett's linear task with one persona instead of his hundred: the first of his personas (loose_leaf_tea, Djedefre, the persona of every run here) with the weights his data gives it, caffeine_content -100, leaf_size 95, oxidation_level 72, steeping_time -67, resteep_potential -1. No screen, two options. Every other setting is `controls/persona_a/one_persona_linear.json`'s, so the two differ in the weights alone: 5000 training trials (the size of his training set, 100 personas x 50), 3000 steps, the nine questions plus `main_trial_first` and `main_no_trial`. `main` is his report question word for word. The training prompts are that control's 5000, answered by these weights (2772 answers differ) |

His own design, 100 personas x 50 trials and 1500 steps, is `configs/plunkett_linear/linear-8b_control_*.json`. Built by
`scripts/make_plunkett_one_persona.py`; checked by `tests/test_plunkett_one_persona.py` and `tests/test_oracle.py`.

## The base model as a control

The untrained model, asked everything a checkpoint is asked: what the model answers having learned nothing. Until
2026-11-23 the files in `base/`, `inverse/`, `rule_tests/` and `controls/persona_a/` (linear and
`minimum_threshold_shuffled`) scored it before training, so the runs made from them hold it as their
`base_model_step-0` files. No file here does that any more: it is a control, run on its own when it is needed.

It needs no training and no settings file of its own. `evaluate_checkpoints.ipynb` with `STEPS = [0]` scores the
base model (step 0 is the model with no adapter) under a finished run's settings:

```
RUN = "<a finished run of that persona>"   # its settings.json gives the prompts, the trials and the batch sizes
STEPS = [0]                                # 0 = the base model
NAME = "base_model"                        # written to results/<run>/evaluate_only/base_model/
QUESTIONS_FROM = None                      # or a settings file, to ask another question set
```

The base model's replies do not depend on training or on the training seed, only on the prompts and the questions,
so one pass serves every run of that persona and question set. To use it as the step 0 of a run that has none:

```
python tools/trial_regression.py results/<trained run> --base results/<run>/evaluate_only/base_model
```

## cut_sweep/: the mix of trial kinds swept, for A, B, A-inverse, B-inverse, C, C-inverse, E-positive, E-negative

`wXX_cYY`: XX% of training trials decided by the weights alone, YY% by the cut (half C, half where one option fails
it), from w75_c25 to w00_c100. In the first A sweep the one-fails trials are only those where the cut overturns what
the weights would pick, which also teaches the opposite weights; the matched four take them as drawn
(`cut_sweep/README.md`). Each branches from its persona's run at decision step 1500
and trains steps 1501 to 3000.

- `persona_a/cut_sweep_wXX_cYY.json`: the first sweep, from A's run 20261106-152354, evaluated every 100. All nine ran
  on 2026-11-07/08 (`results/queue/cut_sweep/`, summary: `python scripts/sweep_summary.py cut_sweep`), asked the
  original four questions. Since 2026-11-24 the files ask the nine questions, with the `"{"` prefix and C shown.
- `persona_a/cut_sweep_a_*`, `persona_b/cut_sweep_b_*`, `persona_a_inverse/cut_sweep_a_inverse_*`,
  `persona_b_inverse/cut_sweep_b_inverse_*`: a matched set of four, each from its persona's 3000-step run, every
  question asked, evaluated every 250, with `_40gb` files. Rebuilt 2026-11-16 with `cut_overturns_only` off; not run yet. `cut_sweep/README.md` has the source runs and
  the commands; a missing source checkpoint is fetched from the Hub.
- `persona_c/`, `persona_c_inverse/`, `persona_e_positive/`, `persona_e_negative/` (`cut_sweep_<persona>_*`): built
  the same way from C's, C-inverse's, E-positive's and E-negative's 3000-step runs of 2026-11-16 (cut at 25% of
  range). Not run yet.

Written by `scripts/make_cut_sweep.py` (`--persona`, `--prefix`, `--gpus`).

These branch from an 8B checkpoint, so there is no 14B or 32B cut sweep yet: run the 14B (or 32B) A base first,
then point `model_hyperparameters.start_from` at that run.

## seeds/: A, A-inverse, B, B-inverse, E-positive and E-negative trained again with seeds 5 and 6

`seeds/persona_<a|a_inverse|b|b_inverse|e_positive|e_negative>/one_persona_<name>_nine_q_seed<N>_40gb.json`, with the
other cards' files beside it (no suffix, `_24gb`, `_80gb`). B-inverse was added on 2026-11-23.
Each is its persona's settings with `model_hyperparameters.seed` set to N and the nine report questions of A's and
B's 3000-step runs (`base/persona_a/one_persona_no_trial_a.json`, the same in `..._no_trial_b.json`: main,
screen_index, screen_cut_units, main_report_units, main_threshold_met, main_threshold_not_met and the three no-trial
screen questions; `"{"` prefix and C shown since 2026-11-24, neither before). The data seeds are untouched, so every seed trains on the same 5000
trials and is asked the same 50 report trials; the seed sets the LoRA starting point, the training order and the
evaluation draws. They skip the base-model evaluation (`baseline_before_training` false): training starts at step 0
as usual and the first scores are at step 250. The base model's replies do not depend on the training seed: A's and
B's seed-4 runs hold them for these nine questions (`base_model_step-0`); for A-inverse, B-inverse, E-positive and
E-negative the seed-4 runs hold them only for the longer question set.

| persona | seed-4 run | same questions as seed 4? |
|---|---|---|
| A | 20261110-211307 | yes (it ran 16 GB batching: micro-batch 2, gradient checkpointing on) |
| B | 20261110-112915 | yes (16 GB batching, as A) |
| A-inverse | 20261113-170347 | no: it asked every question, `"{"` prefix, C shown |
| B-inverse | 20261112-235923 | no: as A-inverse |
| E-positive | 20261116-221042 | no: as A-inverse |
| E-negative | 20261116-221120 | no: as A-inverse |

So A and B have three seeds each in one format (4, 5, 6), and A-inverse, B-inverse, E-positive and E-negative two each (5, 6)
beside a seed 4 in the longer format. Written by `scripts/make_seed_runs.py` (`--seeds` for others); checked by
`tests/test_seed_runs.py`.

Read them with intervals over report trials, then the seeds side by side:

```
python tools/persona_readouts.py results/<run_id> --boot 2000      # <readout>_lo/_hi per step and pooled
python tools/seed_summary.py results/*nine-q-seed* results/20261110-211307_* results/20261110-112915_* --boot 2000
```

The interval resamples the 50 report trials (in a pooled row, a trial's replies at every pooled step together); it
says how much a readout depends on which trials were asked, not on training. `seed_summary` adds the spread across
seeds beside it. With 50 trials, `cos4` of the mean reply is unstable when replies split between all-positive and
all-negative (A seed 4 pooled: 0.50 [0.01, 0.71]); the sign shares and `cos4_each` are tighter. A and A-inverse cut at
35 mg, so only about 8 of the 50 report trials are `both_pass`: read those intervals with that in mind.

### Seed 7 with the cut at 75%: A and A-inverse

`seeds/persona_a/one_persona_a_cut75_nine_q_seed7*.json` and
`seeds/persona_a_inverse/one_persona_a_inverse_cut75_nine_q_seed7*.json` (added 2026-11-23), a file per card as
everywhere, and the same two experiments for Qwen3-14B and Qwen3-32B in those models' `seeds/` folders. They are not
repeats of the runs above: the rule is another one. The caffeine cut is at 52.5 mg (75% of 0 to 70) instead of 35 mg,
the seed is 7, and the nine report questions are the seed repeats', so the runs read beside A's and A-inverse's seeds
4, 5 and 6 at the 50% cut.

| file | weights | screen | data |
|---|---|---|---|
| `persona_a/one_persona_a_cut75_nine_q_seed7` | 0, -83, -38, -100, 31 | caffeine >= 52.5 mg (75%) | `rule_tests/persona_a/one_persona_minimum_threshold_cut75` (the cut-75 rule test's, never run) |
| `persona_a_inverse/one_persona_a_inverse_cut75_nine_q_seed7` | 0, 83, 38, 100, -31 | caffeine >= 52.5 mg (75%) | `rule_tests/persona_a_inverse/one_persona_minimum_threshold_a_inverse_cut75` (new: the same 5000 prompts, the other answer on the 3750 trials the weights decide) |

Each file is `rule_tests/persona_a/one_persona_minimum_threshold_cut75` with the seed, the run tag and the nine
questions set (`"{"` prefix and C shown since 2026-11-24), and for A-inverse its data folder. Three of the nine state the cut:
`screen_cut_units` and `screen_cut_units_no_trial` (right answer: dimension 1, 52.5 mg) and
`screen_cut_percent_no_trial` (dimension 1, 75). Caffeine is listed in whole mg, so the choices are the same for any
cut above 52 and up to 53 mg. No cut probe is among the nine: ask `cut_probe_caffeine_full` of the saved checkpoints
with `evaluate_checkpoints.ipynb`, `QUESTIONS_FROM` = the rule test's file.

What a cut this high does to the evaluation: an option reaches 52.5 mg one time in four, so both options pass in
about one trial in 16. Of the 1000 estimation trials 66 are decided by the weights, and of the 50 report trials 1.
The weights are recovered from those 66 choices: a model that follows the rule on every trial gets a
recovered-against-true cosine of 0.976 here, where G's 25% cut gives 0.999 (`tests/test_oracle.py`), and a
`both_pass` readout rests on one trial. Training is not touched: its 3750 weight-decided trials are a quota. The two
counts are `model_estimating.estimation_trials_per_persona` and `report_schema.samples_per_report` in each file.

Written by `scripts/make_cut75_seed_runs.py` (which also builds A-inverse's data); checked by
`tests/test_cut75_seed_runs.py`.

## Is the trial readout training's, or the prompt's? Step 0 and shuffled-label controls

The trial-shown questions put the trial's values in the prompt, so a readout of how the reports follow the trial
could be something the prompt induces in any model. Three models go through the same regression
(`tools/trial_regression.py`): the trained run, its step 0 (the base model, `base_model_step-0`, already in each
seed-4 run) and a shuffled-label control, `controls/persona_<a|a_inverse|b|e_positive|e_negative>/one_persona_<x>_shuffled_40gb.json`:

- the persona's seed-4 settings exactly (same questions, prefix, C shown or not, seed 4, 3000 steps), on a copy of
  its data folder with the training answers permuted across trials (`random.Random(11)`), so the prompts, the A/B/C
  shares and the report trials are the trained run's and the answers follow the rule only at chance (40%, the
  chance rate at these shares);
- no base-model evaluation: the seed-4 run's step 0 is the same model on the same prompts.

Built by `scripts/make_shuffled_controls.py`; checked by `tests/test_controls_and_tools.py`. A's reuses
`data/.../controls/persona_a/one_persona_minimum_threshold_shuffled/` (A's prompts, the same shuffle).

```
python scripts/run_queue.py --name shuffled_controls configs/minimum_threshold_one_persona/controls/persona_*/one_persona_*_shuffled_40gb.json
python tools/trial_regression.py results/<trained run> --shuffled results/<its shuffled run>
```

The regression: each reported weight on the trial shown (`own`: that attribute's A and B values; `all`: all ten),
R² averaged over the weights, beside a permutation null (replies kept, trials shuffled). With 50 report trials and
10 predictors a reply unrelated to the trial still scores R² about 10/49 = 0.20, so read every R² against its
`null_95`. Training caused the readout if the trained R² is above both the step-0 and the shuffled model's; if all
three are alike, the readout is the prompt's.

### One control for every persona: `controls/shuffled/shuffled_40gb.json`

The per-persona controls above need five GPU runs. This one needs one: E-negative's training prompts with each answer
drawn at random (A and B 0.4375 each, C 0.125, `random.Random(13)`), so no persona's rule is in the data, and it
serves all five. The personas differ only in their training trials (A, A-inverse and B share one set, the E
personas another; same tea, same attributes) and their report format, and the format is set when the questions are
asked, not when the model trains. So train it once, then ask it again in A and B's format:

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/controls/shuffled/shuffled_40gb.json
# evaluate_checkpoints.ipynb: RUN = the shuffled run, NAME = "a_b_format",
#   QUESTIONS_FROM = "configs/minimum_threshold_one_persona/base/persona_a/one_persona_no_trial_a.json"
python tools/trial_regression.py results/<trained run> --shuffled results/<shuffled run>                          # A-inverse, E-positive, E-negative
python tools/trial_regression.py results/<trained run> --shuffled results/<shuffled run>/evaluate_only/a_b_format   # A, B
```

Built by `scripts/make_shuffled_control.py`; checked by `tests/test_shuffled_control.py` (E-negative's prompts in
order, the planned answer mix, agreement with each of the five rules at chance, the same bytes on a rebuild).

## Run everything on an A100 40GB

Every `_40gb` file is complete and its data is in the repo, so a fresh clone runs it as is. A queue runs a list one
after another, each in its own process; the same command carries on after a stop (`--status` shows progress).
On a rented machine add `--hub-repo ANONYMOUS/bdp-plunkett-qwen3` (with `HF_TOKEN` set): each run is then pushed to
the Hub when it ends, where otherwise everything stays on a disk that goes with the machine:

```
# personas C to H (10 runs)
python scripts/run_queue.py --name personas_c_to_h configs/minimum_threshold_one_persona/base/persona_[c-h]/*_40gb.json configs/minimum_threshold_one_persona/inverse/persona_[c-e]/*_40gb.json
# the inverses of F, G and H (4 runs: F-oxidation-inverse, F-resteep-inverse, G-inverse, H-inverse)
python scripts/run_queue.py --name inverses_f_to_h configs/minimum_threshold_one_persona/inverse/persona_[f-h]/*_40gb.json
# the cut sweeps (9 runs each)
python scripts/run_queue.py --name cut_sweep_a configs/minimum_threshold_one_persona/cut_sweep/persona_a/cut_sweep_a_*_40gb.json
python scripts/run_queue.py --name cut_sweep_b configs/minimum_threshold_one_persona/cut_sweep/persona_b/*_40gb.json
python scripts/run_queue.py --name cut_sweep_a_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_a_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_b_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_b_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_c configs/minimum_threshold_one_persona/cut_sweep/persona_c/*_40gb.json
python scripts/run_queue.py --name cut_sweep_c_inverse configs/minimum_threshold_one_persona/cut_sweep/persona_c_inverse/*_40gb.json
python scripts/run_queue.py --name cut_sweep_e_positive configs/minimum_threshold_one_persona/cut_sweep/persona_e_positive/*_40gb.json
python scripts/run_queue.py --name cut_sweep_e_negative configs/minimum_threshold_one_persona/cut_sweep/persona_e_negative/*_40gb.json
# the seed repeats (12 runs: A, A-inverse, B, B-inverse, E-positive, E-negative x seeds 5, 6)
python scripts/run_queue.py --name seeds configs/minimum_threshold_one_persona/seeds/*/*_seed[56]_40gb.json
# seed 7 with the cut at 75% (2 runs: A, A-inverse)
python scripts/run_queue.py --name cut75_seed7 configs/minimum_threshold_one_persona/seeds/*/*_cut75_*_40gb.json
```

or one file: `python scripts/run_one_persona.py --settings <file>`, or `SETTINGS_FILE = "<file>"` in
`run_experiment.ipynb`. Then `python tools/persona_readouts.py results/<run_id>` for the C to H predictions.

## Finished runs and old paths

Runs started before 2026-11-13 recorded `configs/one_persona_*.json` and `configs/cut_sweep_*.json`, and runs from
the morning of 2026-11-13 recorded `configs/minimum_threshold_one_persona/<group>/<file>.json`: their own
`results/<run>/settings.json` holds every setting they used, whatever file it came from. Their data paths are
followed to the folders' new places by `src/data_paths.py`.
