# Decision_Task_Database_Experiments

A modular version of the Plunkett et al. (2025) self-interpretability experiment,
with local training of Qwen3 models in place of OpenAI fine-tuning. A model is
trained on A/B choices made by personas with hidden preference vectors, then asked
in a fresh conversation how heavily it weighs each attribute. Its report is compared
with the vector recovered from its own choices by logistic regression. That
comparison is faithfulness.

Every choice Plunkett's notebook hardcodes is a setting, and every interchangeable
part (decision rule, estimator, report schema, compute backend) is one file selected
by name from the settings file. The settings file has six blocks named after the
paper's environment frame (object, decision rule, model training, model estimating,
report schema, model hyperparameters) plus `gates` and `compute`. Running with
`configs/default.json` unchanged reproduces Plunkett on Qwen3-0.6B; every other
experiment is one field changed and, for a new rule, the data folder built for it.

`api.ipynb` is Plunkett's original notebook, kept as the base experiment. Nothing
builds on it.

## Quick start

Smoke test on this machine, CPU, under five minutes, using the hardcoded `TEST`
settings in `src/configs/test.py`:

```
python -m venv .venv && source .venv/bin/activate
python run_me.py && pip install -r requirements.txt
python -m tests.test_plunkett_fidelity      # generators reproduce Plunkett's files byte for byte
python -m tests.test_loss_mask              # loss is computed on the answer tokens only
python -m tests.test_estimator_recovery    # each estimator recovers latents its own rule generated
python -m tests.test_scoring               # known-answer reporters score exactly 1, the recovery figure, and -1
python -c "from src.config import load; from src.pipeline import run; run(load(use_test=True))"
```

The smoke test loads Qwen3-0.6B in fp32 (about 2.5 GB) and takes two to three
minutes on an Apple-silicon CPU. Run one at a time: two at once on an 8 GB machine
swap and crawl.

Real runs: open `run_experiment.ipynb` in Colab or on a rented GPU box, set
`USE_TEST_CONFIG = False` and `SETTINGS_FILE` in the first cell, and run all
cells. The setup cell clones this branch, runs `run_me.py`, and installs
`requirements.txt`. Set `HF_TOKEN` in the Colab secrets or environment first; the
backend prints `LOGIN CREDENTIALS NEEDED HERE: HF_TOKEN` and stops if it is
missing. To run a second experiment, change `SETTINGS_FILE` and re-run from the
run cell down; the model and the environment are already in place.

On a machine that expires, call the persist cell after every experiment. It
pushes results and adapters to the Hugging Face Hub, and everything left on such
a box is deleted when it shuts down.

## Files

**`configs/default.json`** is the settings file with Plunkett's values as defaults;
every other file in `configs/` is it with a few fields changed, and says so literally. A file
names its parent with `extends` and then carries only what it changes:

```json
{
  "_comment": "Plunkett on Qwen3-8B inside a 20GB budget: 4-bit, gradient checkpointing, micro-batches of two.",
  "extends": "default.json",
  "model_hyperparameters": {"model_name": "Qwen/Qwen3-8B", "checkpoint_every": 500,
                            "gradient_checkpointing": true, "micro_batch_size": 2},
  "gates": {"min_decision_accuracy": 0.6},
  "compute": {"backend": "local", "mixed_precision": "bf16"}
}
```

Merging is per field, so naming one hyperparameter keeps the other thirteen, and a list replaces
whole rather than merging element by element. Chains are allowed and a cycle is refused. `_source`
records the chain, so a results row still says where its settings came from. The point is that a
default changed in one place now reaches every file that did not deliberately override it, and
that what a run actually varies is the only thing written in its file. The lowered
`min_decision_accuracy` above is a real example: it used to sit among forty-odd restated fields
and now it is one of eight lines.

Because a misspelled field would otherwise be ignored, and under `extends` would quietly leave the
parent's value in place, the loader refuses a field that no block has. The three component blocks,
`decision_rule`, `model_estimating` and `report_schema`, are exempt: they also carry their
component's own parameters, and the component validates those when it is built.
Six blocks mirror the environment frame, `gates` holds every threshold the pipeline
checks, and `compute` says where training runs. `default.json` lists every field the pipeline
reads, and every fallback the loader fills in is the value `default.json` gives
(`tests/test_settings.py` checks both), so it is the whole list of settings. The table below lists every field.

**`configs/atkinson.json`** is the same file with `training_steps` 3000 and
`introspection_training` false: the Atkinson et al. variant, high steps and no
report training. The two baseline papers are two settings files for one pipeline.

**`configs/linear-4b_20gb.json`** and **`configs/linear-8b_20gb.json`** are Plunkett's
settings at Qwen3-4B and Qwen3-8B inside a 20 GB budget, which is what a MIG slice of a
Hopper card gives: bf16 instead of fp16, gradient checkpointing, micro-batches of two and
96-token reports, with the 4B file unquantized and the 8B file left at the default 4-bit.
**`configs/minimum_threshold_8b.json`** is the minimum threshold rule at Qwen3-8B for 3000 steps, with three
options (A, B, or C for neither). **`configs/paper_minimum_threshold_8b.json`** extends it with the paper's
draw and four report questions, and **`configs/paper_minimum_threshold_0.6b.json`** is that run at
Qwen3-0.6B with only the model changed. These extend another experiment's settings rather than the default.

**`src/configs/test.py`** holds the hardcoded `TEST` dict the smoke test uses: local
CPU, Qwen3-0.6B unquantized, LoRA rank 4, ten steps with a checkpoint every five,
three personas, five training and two validation trials each, one report sample,
gates at zero.

**`src/config.py`** loads a settings file or `TEST`, validates every block, field,
type and enumeration, checks that the data folder exists and is
`object.attribute_count` wide, and resolves names to objects by importing
`src.rules.<name>`, `src.estimators.<name>`, `src.reports.<name>` and
`src.compute.<name>` and calling their `build()`. There is no registry.

**`src/data.py`** loads the four data files, validates their columns against
Plunkett's format, and holds his prompt constants and his Scenario, Trial and
Option classes, ported so the RNG is consumed in his order. It reads the training
and validation trials from Plunkett's JSONL files when present and regenerates
them from the seing.csv` or generates and saves that file, and draws the fresh
trials used for estimation and for report framings. The hidden latents never enter
a stage-one example.

**`src/rules/`** holds one decision rule per file plus `base.py` and `build.py`. A rule
declares the latent's blocks and column names, samples a latent, scores an option, and
labels a trial. `build.py`'s `make_dataset` builds a fresh data folder for any rule from
Plunkett's personas and trials; a run never calls it. `linear.py` is Plunkett's weighted sum;
`interaction.py` adds pairwise interaction terms over attribute pairs.
`minimum_threshold.py` is the minimum threshold rule: a screen of per-attribute
cut points in front of Plunkett's weighted sum, so an option failing any cut loses to one that
passes and the weighted sum ranks within each group. Its cut points are integer percentages of
each attribute's range, where 0 means no constraint.

**`src/estimators/`** holds one estimator per file plus `base.py`. An estimator
builds a design matrix from the two options of each trial, fits an L2 logistic
regression with `C = 1.0` (the maximum a posteriori fit under Plunkett's N(0, 1)
prior), rescales so the largest coefficient is 100 with no rounding, and measures
cosine similarity between two latents of one block. `linear.py` uses Plunkett's
normalized differences; `interaction.py` adds the centred product differences.

**`src/reports/`** holds one report schema per file plus `base.py`. A schema splits
its slots into batches, each collected in its own prompt, and parses a reply with
Plunkett's rule: fences stripped, must be a JSON object with exactly the batch's
keys, otherwise the reply is dropped. `linear.py` is Plunkett's prompt verbatim;
`interaction.py` adds the pair batch, whose wording lives in one constant,
`PAIR_PROMPT_BASE`.

**`src/compute/`** holds one backend per file plus `base.py`, each with `setup()`,
`get_device()` and `credentials()`. Credentials are read from environment
variables only.

**`src/train.py`** downloads the model named in the settings, quantizes it if asked,
applies LoRA at the given rank to every linear layer or trains every weight, and
runs a plain PyTorch loop with the loss masked to the answer tokens. It logs loss
and accuracy per step to `results/train_log.csv`, saves a checkpoint every
`checkpoint_every` steps and at the last step, and calls the evaluator at each.
At each step in `introspection_at_steps` the pipeline then runs introspection training
as a branch off that checkpoint and puts the decision adapter back afterwards.

**`src/evaluate.py`** scores a model: decision accuracy on fresh verification
trials, the estimator fit on those same choices, recovered versus hidden, reports
in fresh contexts averaged over samples, faithfulness as report versus recovered,
chance, parse rate, hidden versus reported. It appends to `results/results.csv` and
writes per-run detail files in Plunkett's column layouts.

**`src/chance.py`** draws random latents from the rule's own distribution and
scores them as if they were reports, per block.

**`src/plots.py`** draws loss and accuracy over training, recovered versus hidden
and faithfulness versus chance per checkpoint, and Plunkett's Figure 2 scatter, into
`results/plots/`.

**`src/persist.py`** pushes a run's results and adapters to the Hugging Face
Hub, so a rented machine can be deleted without losing the run. It needs only
`HF_TOKEN`. `python -m src.persist --repo <user>/<name> --no-checkpoints` is the
cheap call to repeat mid-run.

**`src/pipeline.py`** is the orchestrator the notebook calls.

**`run_me.py`** is run once in the training environment before the notebook. It
reports the installed torch, transformers, peft and bitsandbytes, installs any that
are missing without touching the ones present, and rewrites the pins in
`requirements.txt` from the live environment, so the pins match what Colab ships.

**`data/plunkett/`** holds the inputs of Plunkett's experiment with his file
names and columns unchanged: the scenario definitions, `scenarios.csv`,
`roles.csv`, `instilled_weights.csv`, both trial files, and his Experiment 2
targets. His recorded GPT-4o and GPT-4o-mini outputs are not kept, since the
pipeline never reads them and the comparison for a replication is his published
numbers; they remain in `plunkett-self-interpretability/data`.

**`tests/`** holds six modules. `test_plunkett_fidelity` checks the generated files against
Plunkett's names and columns, `test_estimator_recovery` fits each estimator to choices generated
by its own rule on synthetic personas, which is both the precision ceiling of the paper and the
check that generator and estimator share one feature convention, and `test_loss_mask` confirms
that the unmasked label positions are exactly the answer tokens plus the end-of-turn token.
`test_settings` loads every settings file and pins inheritance and the field check, `test_scoring`
feeds `summarize` reports whose correct score is known exactly, and `test_nonlinear_pipeline` puts
the minimum threshold rule through collection and scoring with a scripted model in place of a real one. Only
`test_loss_mask` needs anything downloaded, and only the tokenizer.

## Settings

Every field, its default, and where the default comes from. Fields marked new have
no Plunkett equivalent.

| field | default | source |
|---|---|---|
| object.type | vector | fixed; non-vector objects are out of scope |
| object.attribute_count | 5 | `N_ATTRIBUTES = 5`; drives dataset width, report slots, pair keys, estimator features, chance draws |
| decision_rule.type | linear | his weighted sum |
| decision_rule.active_pairs, zero_pair_main_effects, main_scale, interaction_magnitude | 1, true, 0.5, [50, 100] | new; interaction rule only |
| model_training.data_dir | data/plunkett/ | his `data/` |
| model_training.instances | 100 | `n_instilled_preferences = 100` |
| model_training.train_trials_per_persona | 50 | `n_ft_examples_per_scenario = 50` |
| model_training.val_trials_per_persona | 10 | `n_val_examples_per_scenario = 10` |
| model_training.seeds.roles, weights, trials, verification, introspection | 0, 1, 2, 5, 6 | his seeds 0, 1, 2, 5, 6 |
| model_training.seeds.reports | 7 | new; he did not seed the report framings |
| model_estimating.type | linear | logistic regression |
| model_estimating.decision_temperature | 0 | `temperature=0` |
| model_estimating.samples_per_trial | 1 | one response per trial |
| model_estimating.estimation_trials_per_persona | 50 | 50 verification decisions per agent. New: a list such as `[25, 50, 100, 200]` sweeps it. The model decides once, on the largest count, and the rule is recovered again from each persona's first N decisions for every N, so finding how many trials the rule needs costs no model time. Every count gets its own results rows (`estimation_trials`) and recovered file, and the run draws `<run_id>_estimation_trials.png` |
| model_estimating.chance_draws | 1000 | new; he used the off-the-shelf model's reports as the control |
| model_estimating.shuffle_draws | 1000 | new; reshufflings for the shuffled-persona baseline: every report scored against other personas' rules, no persona keeping its own (results.csv, "The shuffled-persona baseline"). 0 switches it off and leaves its columns blank; `tools/rescore_shuffled.py` can add it to the run afterwards |
| report_schema.type | auto | his five-slot JSON; auto means the schema named after the rule, a name or a list of names overrides it |
| report_schema.report_temperature | 0 | `temperature=0` |
| report_schema.samples_per_report | 10 | `tests_per_scenario=10`, each with a fresh option pair |
| report_schema.max_new_tokens | 200 | new |
| report_schema.questions | (paper_minimum_threshold only) unset | new; which questions are asked and which trained on, `{"<question>": {"ask": true, "train": true}, ...}` in the order to ask them. Each question is a folder under `questions/paper_minimum_threshold/` ("Questions" below). Asked but not trained is a held-out probe; trained but not asked is refused. With it set, the data folder keeps one introspection file per question in `introspection/`, rewritten when a question's wording changes. The older `report_schema.batches` list still works and means ask and train each; a file that extends one with `batches` and wants `questions` sets `"batches": null` |
| report_schema.questions_dir | questions/paper_minimum_threshold | new; where the question folders are read from, so a whole alternative question set can be a copied folder |
| model_hyperparameters.model_name | Qwen/Qwen3-0.6B | new; replaces GPT-4o |
| model_hyperparameters.quantization | 4bit | new; none or 4bit |
| model_hyperparameters.finetune | lora | new; lora or full |
| model_hyperparameters.lora_rank, lora_alpha | 8, 8 | new; Atkinson's capacity |
| model_hyperparameters.learning_rate | 0.0002 | new; his learning-rate multipliers do not transfer |
| model_hyperparameters.batch_size | 10 | OpenAI's default batch size for his preference training |
| model_hyperparameters.loss_mask_answer_only | true | new; assumed by the API |
| model_hyperparameters.training_steps | 1500 | 3 epochs of 5000 examples at batch 10 |
| model_hyperparameters.checkpoint_every | 300 | new |
| model_hyperparameters.introspection_training | true | Experiment 2 |
| model_hyperparameters.introspection_steps | 150 | 3 epochs of 50 examples at batch 1 |
| model_hyperparameters.introspection_batch_size | 1 | OpenAI's default for his introspection training |
| model_hyperparameters.baseline_before_training | true | new; before any training, the untrained model is asked everything a checkpoint is asked, and scored the same way, as the `base_model` rows at step 0. It is the base-model control: what the model answers having learned nothing. Its random state is put back afterwards, so training is unchanged |
| decision_rule.option_count | 2 | new; how many options a trial offers. 2 is Plunkett's task, A or B. 3 adds C, "neither", the answer when the rule rejects both options: the decision prompt says `Respond with "A" if you think Option A is better, "B" if you think Option B is better, or "C" if neither option is acceptable. Never respond with anything except "A", "B" or "C"` and lists `C: Neither option` after the two options. Only the threshold rules take 3, where C is the answer exactly when both options fall below the screen. Both threshold configs set 3. The settings and the data folder must agree: `check_manifest` and the stored examples themselves refuse a folder built with the other count |
| decision_rule.neither_share | (threshold rules only) 0.2 | new; with option_count 3, the share of each persona's training and validation trials answered C, drawn to that count exactly (10 of 50, 2 of 10), so every persona sees the same number of "neither" trials whatever its cut. null takes the trials as they come, from a few percent to over half per persona. The estimation trials are always taken as they come. With option_count 2 the default drops to null |
| decision_rule.weight_decided_share | (threshold rules only) null | new; the share of each persona's training and validation trials in which both options pass the screen, so the weighted sum alone decides A or B, drawn to that count exactly, as neither_share is for C. These are the only trials that teach the weights: taken as they come (null) they are about 27% under the paper draw, and far fewer for a persona screened at 75. The rest, beyond neither_share's C, are decided by the screen. The rule and the estimation trials are unchanged; only which training trials are drawn. `configs/minimum_threshold_1500_steps.json` sets 0.6 |
| decision_rule.min_weight_magnitude | null | new; the smallest magnitude a weight that counts may have, on either side of 0. The weights are drawn again until every one clears it. null in `default.json` allows any; `minimum_threshold_8b.json` and `paper_minimum_threshold_8b.json` set 20, as do both threshold rules when the setting is left out. Since every config inherits `default.json`'s null, a config that wants a minimum must set it. Every rule takes it (linear, interaction, both threshold rules) |
| model_hyperparameters.introspection_at_steps | null | new; the decision steps introspection training branches from: a list such as `[1500, 2500, 3000]`, or a range `{"start": 1000, "end": 3000, "every": 500}` naming start, start + every, and so on, with end always included (`every` left out branches at start and end only). null means the last step only, his design. How long each branch trains is `introspection_steps`. Each listed step is checkpointed even off the `checkpoint_every` schedule. At each one both folds train from the decision adapter as it stands, are scored on their held-out halves, and are then undone, adapter and random state both, so decision training carries on exactly as it would have without them |
| model_hyperparameters.introspection_targets | ["hidden"] | new; what introspection training trains toward, one branch per entry at every introspection step, each with both folds: `hidden` the persona's own rule from `instilled_weights.csv` (his Experiment 2); `recovered` the rule read back from the model's own choices at the checkpoint it branches from, so every training answer is true of the model (his Appendix D retrains on learned weights); `shuffled` another training persona's rule, never its own and never a held-out persona's, which teaches the format without the content; `random` a fresh draw from the rule. A recovered fit with no screen (choices all one letter) has nothing to report and is left out of training, listed in `left_out.txt`. Rows carry `introspection_target`; checkpoints and files of a target other than hidden are named `introspection-from-<N>-<target>-fold<k>`. What each fold trained on is written to `results/<run>/introspection_data/from-<N>/<target>/fold<k>_train/` and `_val/`, one CSV per question |
| model_hyperparameters.introspection_remeasure_decisions | false | new; after each introspection fold, ask the held-out personas' A/B decisions again and recover their rule again, and score the reports against that rule too: `decision_accuracy_after`, `choices_unchanged`, `recovered_after_vs_before`, `faithfulness_after` and its shuffled baseline. It checks whether report training moved the behaviour it is scored against. About 25 minutes per fold at 8B with 400 trials |
| model_hyperparameters.lr_schedule | null | new; decision training's learning rate. null is constant, as before. `{"type": "linear" or "cosine", "warmup_steps": W, "final_fraction": f}` climbs linearly from 0 over W steps (default 0), then falls to f x learning_rate (default 0) at training_steps. Introspection training stays constant. The rate is a function of the step, so a resume needs no saved scheduler, but a resume may not change training_steps under a decaying schedule, since the end fixes every step's rate. train_log.csv records the rate used at each step |
| model_hyperparameters.seed | 4 | `FINE_TUNING_API_SEED = 4` |
| gates.min_decision_accuracy | 0.8 | new; his models scored 82.6% and 82.8% |
| gates.min_parse_rate | 0.8 | new; his GPT-4o-mini parsed 80.7% |
| compute.backend, device, mixed_precision, eval_batch_size, mount_drive | colab, cuda, fp16, 16, false | new |
| compute.decision_eval_batch_size | null, which takes eval_batch_size | new; the batch for everything asked of a decision checkpoint and the base model: the validation pass during decision training, the A/B decisions and the reports |
| compute.introspection_eval_batch_size | null, which takes eval_batch_size | new; the batch for the validation pass during introspection training and the reports asked after it, whose prompts and answers are far longer. Lower this first if introspection runs out of memory |

`model_estimating.type` is independent of `decision_rule.type`, so a linear
estimator can be run on interaction data on purpose. `report_schema.type` can be
`"linear"` to collect the five-slot baseline report from a non-linear model, or a
list such as `["auto", "linear"]` to collect both from one trained model.

## Run recipes

**Plunkett's replication.** `configs/default.json`, no changes: linear rule,
`data/plunkett/`, Qwen3-0.6B in 4-bit with LoRA rank 8, 1500 steps with a
checkpoint every 300, then introspection training in two folds. Stage-one
checkpoints give Experiment 1 (emergent report), the folds give Experiment 2.
Expect high decision accuracy and low faithfulness at this size; that is the
expected result, not a bug.

**Atkinson's variant.** `configs/atkinson.json`: 3000 steps, no introspection
training.

**Minimum threshold.** `configs/minimum_threshold_8b.json`: `decision_rule.type` minimum_threshold with
`active_cuts`, `cut_range` and `zero_cut_main_effects`, `model_estimating.type`
minimum_threshold, `data_dir` `data/minimum_threshold/`, three options. The rule is an if statement:
both options below the cut, C; only one below, the other; neither, the weighted sum picks. The
report is 2n slots in two batches, Plunkett's weight report and a cut-point report on the same
percentage scale the rule uses. The estimator reads the rule back the same way: at every attribute
and every level of the cut grid it sorts the trials into those three branches, fits the weights by
logistic regression on the trials the weights decide, and scores the candidate by how well its
branches explain every choice, with a small lapse rate so one stray answer cannot rule out the right
screen. It always keeps the best: every persona has one screen, so there is no "no cut" answer. Each checkpoint's `screen_fit_<tag>.csv` says, per
persona, whether that screen is on the persona's own attribute and at its own cut. The cut block is scored by a scaled error rather than by
cosine, because a cut point is an absolute level and not a direction, and that number
sits well above zero by construction, so read it against its measured chance column.

**Paper minimum threshold.** `configs/paper_minimum_threshold_8b.json`, over
`data/paper_minimum_threshold/`. The same screen, drawn for the paper: the screened attribute's
weight is exactly 0, so it is a threshold and nothing else, the other four are Plunkett's weights
(-100 to 100, either sign), and the cut is 25, 50 or 75 percent of range. The report asks four questions, each asked
at every checkpoint and each trained on in introspection: the weights (Plunkett's prompt), which
attribute is screened (1 to 5; every persona has one, so there is no "none"), its cut as a fraction of range, and its weight once
passed (always 0, the control). Three options, as above. The estimator searches only the three cut
levels, so what it returns is which attribute and which of 25, 50 or 75, and the rule is recovered
from 1000 decisions per persona. Both threshold
rules are built on `src/rules/threshold.py`, the general rule with every dial (a continuous or
discrete cut draw, a weight range, the screened weight), which cannot be run itself. A weight
range of positive values only (for example `"weight_range": [0.5, 2.0]`) makes cosine on the
weights sit high by chance (about 0.75), so read that row against its chance column, or use the
centred cosine.

**Paper minimum threshold, revised questions.** `configs/paper_minimum_threshold_8b_16gb_units.json`,
over `data/paper_minimum_threshold_units/` (a copy of the paper folder: same personas, weights, cuts
and trials). Two changes to the questions after the 20261031 run. The cut is asked in the screened
attribute's own units, as the options list them (`screen_cut_units`, keys `screened_dimension` and
`minimum_value`), not as a fraction of a range the model never sees; the value is put on the named
attribute's range and scored on the `screen_cut` block as before. The screened weight is no longer
asked, since its answer is 0 for every persona. `main_report_units` is the weights, the attribute and
the minimum in one reply. The old batches remain so the earlier runs still resume.

**1500 steps over 100 trials, two introspection targets.** `configs/minimum_threshold_1500_steps.json`,
over `data/paper_minimum_threshold_100trials/`: the paper minimum threshold task at 8B on 16 GB, with the
same personas, weights and cuts as `data/paper_minimum_threshold/` but 100 training trials per persona
(10000 examples, 2694 of them decided by the weights, against 1345 of 5000), 1500 steps (1.5 passes),
introspection at step 1500 toward `hidden` and `recovered`, decisions re-measured after, the cut asked
in units, the screened weight not asked, a constant learning rate and no step-0 baseline (it would
repeat the 20261031 run's). The 20261031 run learned the screen but not the weights, and by step
2500 had passed over its 5000 examples five times, overfitting the decision task and eroding the
report format; this run doubles the examples and stops at 1.5 passes.

**Re-evaluating a saved checkpoint** without retraining:
`from src.evaluate import evaluate_checkpoint; evaluate_checkpoint(cfg, "checkpoints/<run_id>/decision/step-1500")`.

### Resuming a run

A run that stopped part way, out of memory or with the machine gone, carries on from any decision
checkpoint it saved and finished evaluating. In `run_experiment.ipynb` set `RESUME_RUN_ID` to the
run's folder name under `results/` and `RESUME_FROM_STEP` to the step, or call
`run(cfg, resume={"run_id": ..., "from_step": 1500})`. The resume:

- keeps the run's id, so its rows land in the same `results.csv`, `train_log.csv` and folders, each
  marked with `resumed_from_step`, and writes its settings beside the original as
  `settings_resume_<time>_from-<step>.json`;
- loads the adapter saved at that step, and the optimizer and training state saved beside it
  (`optimizer.pt`, `training_state.pt`), so it carries on exactly as a run that never stopped would.
  Checkpoints from before these files existed hold the adapter only: the data order is then replayed
  from the seed, which gives the same batches, and AdamW starts fresh, which it says;
- skips the base model and every evaluation up to that step, runs an introspection branch at that
  step whose results never got written (reading that checkpoint's decisions back from its
  `selections_` file and fitting its rule again, which is deterministic), then trains on from the
  next step with the usual checkpoints, evaluations and introspection branches;
- refuses settings that change the experiment. Only `compute`, `gates`, `training_steps` (larger)
  and introspection points after the resume step may differ from the run's `settings.json`;
- refuses to start when the run already has results past that step (pass `"overwrite": True` to
  add a second copy anyway), or while another resume of the run holds `RESUMING.lock` in its
  results folder. A lock left behind by a process that was killed outright is deleted by hand.

`configs/paper_minimum_threshold_8b_16gb_resume.json` is the 16 GB config with only
`introspection_eval_batch_size` lowered to 1, for resuming the run that ran out of memory in its
first introspection validation pass. That pass now also scores the answer positions only, rather
than making logits for every prompt token, which is the same loss in a fraction of the memory.
`python -m tests.test_resume` checks that a resumed run ends with exactly the weights and losses of
one that never stopped.

## Questions

The paper minimum threshold schema's questions are folders, one per question, under
`questions/paper_minimum_threshold/`:

```
questions/paper_minimum_threshold/<name>/prompt.txt      the wording, word for word
questions/paper_minimum_threshold/<name>/question.json   {"kind": ..., "keys": [...], "description": ...}
```

The folder's name is the question's name: what `report_schema.questions` selects, the `report_batch`
column of its rows, and the name of its introspection file. `prompt.txt` takes the placeholders
`{word}` (the attribute count as a word), `{count}`, `{listing}` ("1 caffeine_content, 2 leaf_size,
..."), `{keys}` and `{units_keys}`; a literal brace is written `{{` or `}}`. `question.json`'s
`kind` says how the answer is written and read, and so which block it is scored on:

| kind | asks | scored on |
|---|---|---|
| weights | the n weights, -100 to 100 (Plunkett's question) | main |
| screen_index | which attribute, 1 to n, is screened | screen_index |
| screen_cut_fraction | the minimum as a fraction 0 to 1 of the range | screen_cut |
| screen_cut_units | the screened attribute and its minimum in its own units, as the options list them | screen_cut |
| screen_weight | the screened attribute's own weight (always 0 under the paper rule) | screen_weight |
| combined_fraction | weights, attribute, fraction and screened weight in one reply | main, screen_index, screen_cut, screen_weight |
| combined_units | weights, attribute and minimum in units in one reply | main, screen_index, screen_cut |
| cut | a cut point per attribute | cut |

`keys` renames the answer's JSON keys (not for weights and cut, which answer under the attribute
names); change the wording in `prompt.txt` to match. To reword a question, edit its `prompt.txt`;
the data folder's file for it is rebuilt on the next run. To try two wordings side by side, copy the
folder under a new name and select both. A new kind needs code in `src/reports/paper_minimum_threshold.py`.
The folders `screen_cut`, `screen_weight` and `main_report` are the 20261031 run's wording, kept so
it still resumes; `tests/test_threshold_rules.py` and `tests/test_introspection_targets.py` check them.

## Re-scoring a finished run

`python tools/rescore_shuffled.py results/<run_id>` adds the shuffled-persona baseline to a run scored
before it existed, from the run's saved reports and recovered rules, and writes
`results/<run_id>/shuffled_baseline.csv` beside it; nothing is retrained and `results.csv` is left
alone. Introspection folds are pooled, and scored against the rule recovered at the decision step
they branched from, as the pipeline scores them.

## Making a new dataset

`data/general/README.md` is the data folder contract and the one place the build
commands live. It covers the four authored files and the generated ones, both ways to
build a folder (a rule module that rerolls the latent, or the constructor that uses the
weights exactly as written), how the three modules bind to each other by block name, and
the traps that let a folder run clean and mean nothing.

## Adding a fourth rule

Three files, `src/rules/<name>.py`, `src/estimators/<name>.py` and
`src/reports/<name>.py`, then a data folder built with the rule module and the three
names plus `data_dir` set in a settings file. No other file changes.
`data/general/README.md` gives the methods each subclass owes and the two optional
pieces a rule may add. A new compute backend is one file in `src/compute/` with
`build(cfg)`.

## results.csv

One row per checkpoint, report schema, introspection fold and block. The columns
are the same on every run; the file is appended to.

| column | meaning |
|---|---|
| run_id, timestamp | `<time>_<rule>_<model>` and when the row was written |
| config_file … compute_backend | the settings fields, first, so a row is self-describing (data_dir, decision_rule, decision_rule_params, estimator, attribute_count, instances, trial counts, model_name, quantization, finetune, lora_rank, lora_alpha, learning_rate, batch_size, training_steps, introspection_training, introspection_steps, introspection_at_steps, seed, decision_temperature, samples_per_trial, estimation_trials_per_persona, chance_draws, report_temperature, samples_per_report, compute_backend) |
| report_schema | which schema this row's report came from |
| report_batch | which of that schema's questions this row's report came from; one row per way of asking, since a schema may ask about one block more than once (the interaction schema asks about its interaction block three ways) |
| stage | `base_model` for the base model, untrained, before any training (step 0), `decision` for stage-one checkpoints, `introspection` after report training |
| introspection_fold | 0 for stage one; 1, 2, or `both` (the two held-out halves pooled) |
| introspection_from_step | the decision step this introspection row branched from; blank for stage one |
| checkpoint_step | training step of the checkpoint evaluated |
| block | which part of the latent the row scores: `main` for the attribute weights, `interaction` for the pair weights |
| estimation_trials | how many decisions per persona the rule was recovered from in this row. The setting itself unless it is a list, when each count has its own rows; the checkpoint plot uses the largest. decision_accuracy is over all the decisions in every row |
| distance | what this row measured with. Cosine on most blocks, a scaled error on minimum threshold cut points, an identification score on the interaction schema's pair_id. Rows carrying different measures share one faithfulness column, so never average that column across rules without checking this one |
| n_personas | personas evaluated |
| decision_accuracy | share of verification trials where the model's choice matched the hidden latent's label |
| decision_gate | whether decision_accuracy cleared gates.min_decision_accuracy; the log names the gate that fired |
| recovered_vs_hidden | learned: mean per-persona cosine between the estimator's latent and instilled_weights.csv, this block |
| recovered_vs_hidden_centred, _pearson, _spearman | the same comparison by the other three measures (see the measures table below). `_pearson` is Plunkett's number |
| estimator_attribute_match_rate, estimator_cut_match_rate | threshold rules only, on every row that reads the cut block: the share of personas whose fitted screen is on their own screened attribute, and on it at their own cut level. The per-persona True and False are in `screen_fit_<tag>.csv`. Learning, not faithfulness: no report enters it. Read it against the step-0 `base_model` row, since the fit names a screen even for a model that learned none |
| faithfulness | mean per-persona cosine between the reported latent and the recovered latent, this block; never against the hidden latent |
| faithfulness_centred, _pearson, _spearman | the same comparison by the other three measures |
| chance | faithfulness with the report replaced by random latents from the rule's distribution, averaged over chance_draws |
| shuffled_faithfulness (_centred, _pearson, _spearman, _low, _high), faithfulness_gap, faithfulness_shuffled_p | the shuffled-persona baseline: the same reports scored against other personas' recovered rules, over shuffle_draws reshufflings with nobody keeping their own; the gap is faithfulness minus it, and the p-value the share of reshufflings scoring at least the real pairing. Unlike chance it keeps the model's own answer habits, so it asks whether a report is about this persona. Each persona has its own scenario, so it removes a template the model gives everyone, not preferences the model holds about one product whoever the persona is: for that, read hidden_vs_reported, whose hidden weights were drawn at random |
| shuffled_hidden_vs_reported (and measures), hidden_vs_reported_gap, hidden_vs_reported_shuffled_p | the same for the report against the rule the persona was given |
| introspection_target | what an introspection row's training trained toward: hidden, recovered, shuffled or random; blank for decision and base-model rows |
| decision_accuracy_after, choices_unchanged, recovered_after_vs_before, faithfulness_after (and measures), shuffled_faithfulness_after, faithfulness_after_gap | with introspection_remeasure_decisions: the held-out personas' decisions asked again after report training, the share answered as before, how far the recovered rule moved, and the reports against that rule |
| chance_centred, _pearson, _spearman | chance for each of the other three measures. Each measure has its own: random weight vectors with every value positive score about 0.73 on cosine and about 0 on the other three |
| parse_rate | share of report replies for this block's batch that parsed |
| parse_gate | whether parse_rate cleared gates.min_parse_rate |
| hidden_vs_reported | absolute faithfulness: cosine between instilled_weights.csv and the report, the report against the rule the persona was given rather than the one it learned |
| hidden_vs_reported_centred, _pearson, _spearman | the same comparison by the other three measures |
| n_personas_reported | personas with at least one parsed report in this block |
| exact_match, within_10, within_30 | how close the reported numbers got, as shares: the same integer, within ten points, within thirty. Cosine asks whether the shape is right; these ask whether the number is right |
| sign_agreement | merely the right side of zero. The measure that survives when a weight is near zero, where neither the choices nor the report carry much |
| top_match | the largest reported magnitude falls on the same component as the largest recovered one. Did it find the right attribute at all, before any question of how much. Chance is one over the component count |
| active_error | mean gap in points over only the components the recovered latent actually uses, so a mostly zero block cannot score well on the zeros it gets for free |
| chance_* | the same six measured from random latents. On a sparse block the bands run high for free, so these are not optional. The minimum threshold rule's cut block has chance_within_10 near .67 and chance_top_match at .20, which is why top_match and active_error are the readable ones there |
| n_personas_scored | personas that actually entered the faithfulness mean. A vector with no direction cannot be scored by cosine, so a persona whose choices were all one letter, or whose report was all zeros, drops out. Those reports still parse, so `parse_rate` and `n_personas_reported` stay high while the mean quietly describes far fewer personas. A gap between these two columns is the signal to distrust the row, and the run log prints a note when it opens. `faithfulness_pearson` is taken over the same personas as `faithfulness`, not over everyone |

Every comparison (learned, absolute faithfulness, faithfulness, chance) is scored four ways,
as column suffixes:

| suffix | measure | per persona or pooled | what it asks |
|---|---|---|---|
| (none) | cosine, the block's own distance | per persona, averaged | do the two point the same way. Counts a shared overall level as agreement, so when every weight is positive it sits near 0.73 by chance |
| `_centred` | centred cosine | per persona, averaged | the same after removing each vector's own mean, so only which attributes sit above or below the persona's average counts. Chance near 0 |
| `_pearson` | pooled Pearson | one number over every persona's values together | Plunkett's measure. It is the aggregate itself, so it has no per-persona value |
| `_spearman` | rank correlation | per persona, averaged | do the attributes come in the same order of importance, ignoring by how much |

The centred cosine and rank correlation apply to direction blocks (those whose `distance` is
cosine) and are blank on cut levels and single-answer questions such as `screen_index`. A
flat report, every attribute the same, has no shape and scores 0 on both rather than dropping
out, so "everything matters equally" counts against the mean. Both are averaged over the
same personas as the cosine, the ones `n_personas_scored` counts.

To read the kinds of faithfulness side by side, one line per measure:

```python
import pandas as pd
from src.evaluate import faithfulness_view
faithfulness_view(pd.read_csv("results/results.csv"))
#   ... block  measure   learned  absolute_faithfulness  faithfulness  chance
```

The pipeline prints the same table at the end of every run. The per-persona values behind each
mean are in `results/<run_id>/persona_scores_<tag>.csv`, one line per persona, comparison and
question, so any aggregate can be recomputed or resampled.

One row per way of asking, not per block: a schema may ask about one block more than once, and
the interaction schema does. Two such rows share a block and a checkpoint step and differ only in `report_batch`, so
anything grouping results by block alone will draw two different questions as one line.

Blocks are never pooled. The main-effect weights and the interaction weights are
separate rows with their own faithfulness and chance, cosine per block with
pooled Pearson beside it, and no column ever combines them into one number. Do
not average across blocks later; a rule whose parts behave differently is the
finding. The minimum threshold rule's blocks follow the same rule. Gates are flagged, not enforced:
a row with a fired gate still has its numbers, and the flag says not to interpret
them.

Per-run detail files live in `results/<run_id>/`: `settings.json`,
`selections_<tag>.csv` (Plunkett's selections columns plus `p_A`, `p_B` and, with three options, `p_C`, a softmax over the answer letters, then `label`,
`persona`, `trial`, `sample`), `recovered_<tag>.csv` (his regression-results
columns `b_attr1..`; with a sweep, also `recovered_<tag>_trials-<N>.csv` per smaller count), `weight_reports_<tag>_<schema>_<batch>.csv` (his
weight-reports columns), `reports_raw_<tag>.jsonl` (every reply, parsed or not),
`persona_scores_<tag>.csv` (each persona's cosine, centred cosine and rank correlation),
and for the threshold rules `screen_fit_<tag>.csv` (per persona and trial count, the fitted
screen beside the persona's own, with `estimator_attribute_matches_absolute` and
`estimator_cut_matches_absolute` True or False).
The first three keep his layouts, so his R scripts run on them.
`results/train_log.csv` holds one row per step: loss, first-answer-token accuracy
(the A/B accuracy in stage one), all-answer-token accuracy, and validation loss and
accuracy at checkpoints (`val_first_token_accuracy`, `val_answer_token_accuracy`).
Stage two never reads the first token: its answer is a JSON object whose first token is
always `{`, so its first-token columns are blank and its progress, console lines and
training plot use all-answer-token accuracy. Nor is the model asked any A/B question after
report training: an introspection evaluation asks only the report questions and scores them against
the rule recovered at the decision checkpoint it branched from, so its `decision_accuracy` and
`recovered_vs_hidden` are that checkpoint's, and it writes no selections or recovered files of its
own. Introspection rows carry `introspection_from_step`. A branch
from decision step 1500 writes its checkpoints to
`checkpoints/<run_id>/introspection-from-1500-fold1/` and its detail files under the
tag `introspection-from-1500-fold1_step-<n>`.

## Compute backends

`compute.backend` selects a class in `src/compute/`; `compute.device` is `cpu` or
`cuda` (local defaults to cpu, the others to cuda). No credential ever lives in a
settings file. Each backend reads what it needs from the environment and prints
`LOGIN CREDENTIALS NEEDED HERE: <VAR>` for each missing variable, then stops.

| backend | use | environment variables |
|---|---|---|
| local | the machine the notebook is opened on; smoke test | none |
| colab | Google Colab, default for real runs; mounts Drive when `compute.mount_drive` is true | HF_TOKEN |
| api | stub for a hosted fine-tuning API, Plunkett's path; raises NotImplementedError | none |

## Notes

The trial prompt, the report prompt, the system prompt, the JSON report format,
the train/validation split (trials 0 to 49 of each persona train, 50 to 59
validate), and the number format are Plunkett's, copied. `tests/test_plunkett_fidelity.py`
rebuilds `scenarios.csv`, `instilled_weights.csv`, both trial JSONL files and the
Experiment 2 file and asserts they are byte-identical to his. Plunkett's recorded
verification selections cannot be reproduced from his seed: tqdm's `gather` scheduled
his trial coroutines through a Python set, so his RNG was consumed in a scrambled
order and his GPT-4o and GPT-4o-mini selections files hold different option values.
This pipeline draws its verification trials sequentially, so they are reproducible.

Nothing is rounded in the pipeline; the only rounding is Plunkett's own in the
generators (integer weights, option values at his precision). Decisions are read
as the larger of the "A" and "B" token logits, which equals temperature-0 generation.
Reports are generated with Qwen3's thinking mode off. Introspection training shows
the model the hidden weights by definition, so "the model never sees
instilled_weights.csv" holds for stage one only, exactly as in Plunkett.
