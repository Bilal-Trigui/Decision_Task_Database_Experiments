# Decision_Task_Database_Experiments

This repository contains a modular reimplementation of the self-interpretability
experiment of Plunkett et al. (2025). The original study fine-tuned OpenAI models
through a hosted API. This version trains open Qwen3 models locally, so that every
part of the experiment can be inspected and changed.

## Contents

1. Research question
2. How the experiment works
3. Key terms
4. Repository layout
5. Getting started
6. Settings files
7. Settings reference
8. Experiments
9. Reading the results
10. Compute backends
11. Extending the pipeline
12. Fidelity to Plunkett et al.

## 1. Research question

The study asks whether a language model can accurately describe a preference that
it acquired through training but was never told about directly. The reasoning
proceeds in four steps.

1. A model is trained to make choices on behalf of fictional characters, called
   personas. Each persona has a hidden set of preferences, but the model never sees
   those preferences. It sees only which option the persona chose.
2. If training succeeds, the model's choices must follow some internal rule. That
   rule can be estimated from the model's choices alone, using statistics.
3. The model is then asked, in a new conversation, to state that rule in its own
   words and numbers.
4. If the stated rule matches the rule estimated from the model's behaviour, the
   report is called **faithful**. Faithfulness is the quantity this repository
   measures.

The comparison in step 4 is always between the report and the estimated rule, and
never between the report and the hidden preferences. The reason is that the model
may have learned the hidden preferences imperfectly. A faithful report describes
what the model actually learned, even when that differs from what it was meant to
learn.

## 2. How the experiment works

A single run passes through five stages, in this order.

1. **Load the data.** The pipeline reads a data folder that defines the personas,
   the choices they face, and the hidden preferences that determine each choice.
2. **Decision training (stage one).** The model is fine-tuned on examples of the
   form "given options A and B, this persona chooses A". The correct answer is a
   single letter. The hidden preferences are never shown.
3. **Estimation.** At each saved checkpoint the model makes choices on new trials.
   A statistical estimator reconstructs, from those choices alone, the preferences
   the model appears to be using. This result is called the **recovered latent**.
4. **Report collection.** In a fresh conversation, the model is asked to state the
   preferences it used, as numbers in a fixed format.
5. **Scoring.** The report is compared with the recovered latent. The same
   comparison is made with randomly generated reports, which shows how high a score
   can be reached by chance alone.

An optional second stage, **introspection training**, follows stage one. It
teaches the model the format of a correct report on one half of the personas, and
the model is then tested on the other half. Because the model never saw the
correct answer for the tested personas, a good score on them cannot be explained
by memorisation.

## 3. Key terms

| term | meaning |
|---|---|
| persona | A fictional character paired with one type of choice, for example a character choosing loose leaf tea. |
| attribute | One measurable property of an option, such as caffeine content or leaf size. Each choice type has five by default. |
| latent | The hidden preferences that determine a persona's choices. In the linear rule this is a list of one weight per attribute. |
| decision rule | The formula that turns a latent and two options into a choice. |
| block | A named part of a latent that is scored on its own. The attribute weights are always the `main` block. |
| estimator | The statistical procedure that recovers a latent from observed choices. |
| report schema | The prompt used to ask for a report, together with the rule for reading the reply. |
| faithfulness | The similarity between the model's report and the recovered latent. |
| chance | The faithfulness score obtained by random reports. A result is meaningful only if it exceeds chance. |
| cosine similarity | A measure of whether two lists of numbers point in the same direction. It ranges from −1 (opposite) through 0 (unrelated) to 1 (identical direction). |
| LoRA | A training method that adjusts a small number of added parameters rather than the whole model. It greatly reduces memory use. |
| checkpoint | A saved copy of the model at a given training step, which is evaluated separately. |
| gate | A minimum standard, such as decision accuracy, that a checkpoint must meet before its results are interpreted. |

## 4. Repository layout

```
Decision_Task_Database_Experiments/
  README.md               this document
  run_experiment.ipynb    the notebook used for real runs
  run_me.py               prepares the Python environment before the notebook
  requirements.txt        package versions
  api.ipynb               Plunkett's original notebook, kept for reference only
  configs/                settings files, one per experiment
  data/
    general/              shared data-building tools and the data folder contract
    plunkett/             the inputs of the original experiment
    a4/                   the constrained tradeoff experiment (A4)
  src/                    the pipeline
    rules/                decision rules, one file each
    estimators/           estimators, one file each
    reports/              report schemas, one file each
    compute/              compute backends, one file each
    configs/test.py       the small settings used by the smoke test
  tests/                  automated checks
  results/                created by runs, not tracked by git
  checkpoints/            created by runs, not tracked by git
```

### What each part does

**`run_experiment.ipynb`** runs one experiment from a settings file. It is the
recommended entry point for all real runs.

**`run_me.py`** is run once in a new training environment, before the notebook. It
reports the installed versions of torch, transformers, peft and bitsandbytes, and
installs any that are missing without changing those already present. It then
rewrites the version pins in `requirements.txt` to match the live environment.
This matters because Colab supplies a version of torch built for its own GPU, and
replacing it would break GPU support.

**`api.ipynb`** is Plunkett's original notebook. It is kept as a record of the base
experiment. No code in the repository depends on it.

**`src/config.py`** loads a settings file. It checks every block, field, type and
permitted value. It also checks that the data folder exists and has the number of
attributes the settings expect. Each component is then located by name: the value
`"tradeoff"` in `decision_rule.type`, for example, loads `src/rules/tradeoff.py` and
calls its `build()` function. Because components are found by file name, there is
no central list to maintain.

**`src/data.py`** loads the data folder and checks its columns against Plunkett's
format. It also holds his prompt text and his Scenario, Trial and Option classes.
These classes are ported so that the random number generator is used in the same
order as in his code, which is necessary for his files to be reproduced exactly.
The module reads the training and validation trials from Plunkett's JSONL files
when they are present, and regenerates them from the seeds when they are absent.
It reads `introspection_training.csv` when that file covers every persona, and
otherwise generates and saves it. It also draws the fresh trials used for
estimation and for report prompts. The hidden latents never appear in a stage-one
training example.

**`src/rules/`** holds one decision rule per file, plus the shared base class in
`base.py`. A rule declares its blocks and column names, draws random latents, scores
an option, and labels a trial. It also inherits `make_dataset`, which builds a new
data folder from Plunkett's personas and trials.

- `linear.py` is Plunkett's weighted sum.
- `interaction.py` adds terms that depend on pairs of attributes together.
- `tradeoff.py` is A4, the constrained tradeoff. It places a screen of minimum
  levels, called cut points, in front of Plunkett's weighted sum. An option that
  fails any cut point loses to an option that passes, and the weighted sum ranks
  options within each group. Cut points are whole-number percentages of each
  attribute's range, and 0 means no constraint.

**`src/estimators/`** holds one estimator per file, plus `base.py`. An estimator
builds a table of features from the two options in each trial and fits a logistic
regression with L2 regularisation and `C = 1.0`. This setting is equivalent to
Plunkett's assumption that each weight is drawn from a normal distribution with
mean 0 and variance 1. The result is rescaled so that its largest value is 100,
with no rounding. The estimator also computes the cosine similarity between two
latents, one block at a time.

- `linear.py` uses Plunkett's normalised differences between the two options.
- `interaction.py` adds centred differences of attribute products.
- `tradeoff.py` also searches for cut points. It is described in `data/a4/README.md`.

**`src/reports/`** holds one report schema per file, plus `base.py`. A schema
divides its questions into batches, and each batch is asked in a separate prompt.
Replies are read with Plunkett's rule: code fences are removed, and the reply must
then be a JSON object with exactly the expected keys. Any other reply is discarded.

- `linear.py` uses Plunkett's prompt word for word.
- `interaction.py` adds a batch about attribute pairs, worded in `PAIR_PROMPT_BASE`.
- `tradeoff.py` adds a batch about cut points, worded in `CUT_PROMPT_BASE`.

**`src/compute/`** holds one compute backend per file, plus `base.py`. Each backend
provides `setup()`, `get_device()` and `credentials()`. Credentials are read only
from environment variables.

**`src/train.py`** downloads the model named in the settings, compresses it to 4 bits
if requested, and applies LoRA at the given rank to every linear layer, or trains
every weight if full fine-tuning is chosen. Training is a plain PyTorch loop in
which the loss is computed only on the answer tokens. Loss and accuracy are written
to `results/train_log.csv` at every step. A checkpoint is saved every
`checkpoint_every` steps and at the final step, and each checkpoint is evaluated.

**`src/evaluate.py`** scores one checkpoint. It measures decision accuracy on fresh
trials, fits the estimator to those same choices, compares the recovered latent
with the hidden one, collects reports in fresh conversations and averages them over
samples, and computes faithfulness, chance, the parse rate, and the similarity of
the report to the hidden latent. Results are appended to `results/results.csv`, and
detail files are written in Plunkett's column layouts.

**`src/chance.py`** draws random latents from the rule's own distribution and scores
them as though they were reports, one block at a time.

**`src/plots.py`** draws training loss and accuracy, recovery and faithfulness
against chance at each checkpoint, and a version of Plunkett's Figure 2, all into
`results/plots/`.

**`src/persist.py`** uploads a run's results and adapters to the Hugging Face Hub, so
that a rented machine can be deleted without losing the run. It needs only
`HF_TOKEN`. The command `python -m src.persist --repo <user>/<name> --no-checkpoints`
is an inexpensive upload that can be repeated during a run.

**`src/pipeline.py`** runs the stages in order. The notebook calls it.

**`tests/`** holds six modules, described in section 5.

## 5. Getting started

### 5.1 Smoke test on a laptop

The smoke test confirms that the code works before any expensive run. It uses the
small settings in `src/configs/test.py` (Qwen3-0.6B without compression, LoRA rank
4, ten training steps, three personas) and runs on a CPU in under five minutes.

```
python -m venv .venv && source .venv/bin/activate
python run_me.py && pip install -r requirements.txt
python -m tests.test_plunkett_fidelity
python -m tests.test_loss_mask
python -m tests.test_estimator_recovery
python -m tests.test_scoring
python -c "from src.config import load; from src.pipeline import run; run(load(use_test=True))"
```

The tests check the following.

| test | what it confirms |
|---|---|
| `test_plunkett_fidelity` | The data generators reproduce Plunkett's files byte for byte. |
| `test_loss_mask` | The loss is computed only on the answer tokens and the end-of-turn token. This is the only test that downloads anything, and it downloads only the tokenizer. |
| `test_estimator_recovery` | Each estimator recovers latents that its own rule generated. This sets the best precision the experiment can reach, and it confirms that the rule and estimator describe options in the same way. |
| `test_scoring` | Reports with a known correct score receive exactly that score: 1, the recovery figure, and −1. |
| `test_settings` | Every settings file loads, inheritance works, and misspelled fields are refused. |
| `test_nonlinear_pipeline` | A4 passes through report collection and scoring, with a scripted model in place of a real one. |

The smoke test loads Qwen3-0.6B at full precision, which uses about 2.5 GB of
memory, and takes two to three minutes on an Apple-silicon CPU. Only one model job
should run at a time. On an 8 GB machine, two simultaneous jobs exhaust memory and
slow down severely.

### 5.2 Real runs

Real runs require a GPU, either on Colab or on a rented machine.

1. Set `HF_TOKEN` in the Colab secrets or in the environment. If it is missing, the
   backend prints `LOGIN CREDENTIALS NEEDED HERE: HF_TOKEN` and stops.
2. Open `run_experiment.ipynb`.
3. In the first cell, set `USE_TEST_CONFIG = False` and set `SETTINGS_FILE` to the
   chosen file in `configs/`.
4. Run all cells. The setup cell clones this branch, runs `run_me.py`, and installs
   `requirements.txt`.
5. To run a further experiment, change `SETTINGS_FILE` and run again from the
   **Run the experiment** cell. The model and environment are already in place, so
   only the new training time is spent.

On a rented machine that is deleted when it expires, run the save cell after every
experiment. Anything left on the machine is lost when it shuts down.

## 6. Settings files

Every choice that Plunkett's notebook fixed in code is a setting in this
repository. The settings file has six blocks named after the paper's environment
frame, plus two blocks for practical matters.

| block | what it controls |
|---|---|
| `object` | What kind of latent is studied, and how many attributes it has. |
| `decision_rule` | How a latent turns two options into a choice. |
| `model_training` | The data folder, the number of personas and trials, and the seeds. |
| `model_estimating` | Which estimator is used and how many trials it receives. |
| `report_schema` | How the report is requested and how long the reply may be. |
| `model_hyperparameters` | The model, the training method and the training length. |
| `gates` | The minimum standards a checkpoint must meet. |
| `compute` | Where the training runs. |

### 6.1 Inheritance

`configs/default.json` holds Plunkett's values. Running it unchanged reproduces his
experiment on Qwen3-0.6B. Every other file names a parent with `extends` and
contains only the fields that differ from that parent.

```json
{
  "_comment": "Plunkett's experiment on Qwen3-8B within 20 GB of GPU memory: 4-bit compression, gradient checkpointing, micro-batches of two, and shorter reports.",
  "extends": "default.json",
  "report_schema": {"max_new_tokens": 96},
  "model_hyperparameters": {"model_name": "Qwen/Qwen3-8B", "checkpoint_every": 500,
                            "gradient_checkpointing": true, "micro_batch_size": 2},
  "gates": {"min_decision_accuracy": 0.6},
  "compute": {"backend": "local", "mixed_precision": "bf16"}
}
```

The rules of inheritance are as follows.

1. Fields are merged one at a time. Naming one hyperparameter therefore keeps all
   the others from the parent.
2. A list replaces the parent's list entirely. It is not merged element by element.
3. A parent may itself have a parent. A loop of parents is refused.
4. The field `_source` records the chain of files, so every results row states where
   its settings came from.

This design has two benefits. First, a default changed in `default.json` reaches
every file that does not deliberately override it. Second, each file shows only
what its experiment varies, so the purpose of a file can be read at a glance.

### 6.2 Protection against misspellings

A misspelled field would otherwise be ignored without warning. Under inheritance it
would be worse, because the parent's value would silently remain in force. For this
reason the loader refuses any field that no block defines. The three component
blocks, `decision_rule`, `model_estimating` and `report_schema`, are exempt, because
they also carry parameters specific to their component. Those parameters are
checked by the component itself when it is built.

### 6.3 The files provided

| file | purpose |
|---|---|
| `default.json` | Plunkett's experiment on Qwen3-0.6B, compressed to 4 bits, sized for a Colab T4 GPU. |
| `atkinson.json` | The Atkinson et al. variant: 3000 training steps and no introspection training. |
| `linear-4b_20gb.json` | Plunkett's experiment on Qwen3-4B within 20 GB of GPU memory, without compression. |
| `linear-8b_20gb.json` | Plunkett's experiment on Qwen3-8B within 20 GB of GPU memory, compressed to 4 bits. |
| `a4_8b.json` | The constrained tradeoff (A4) on Qwen3-8B for 3000 steps. |

The two `_20gb` files fit the memory of one MIG slice of an NVIDIA Hopper GPU. To
fit, they use bf16 precision instead of fp16, gradient checkpointing, micro-batches
of two, and reports limited to 96 tokens. `a4_8b.json` is the only file whose
parent is another experiment (`linear-8b_20gb.json`) rather than the default.

## 7. Settings reference

The table lists every field, its default value, and the source of that default.
Fields described as "new" have no equivalent in Plunkett's code.

| field | default | source |
|---|---|---|
| object.type | vector | fixed; other kinds of latent are outside the current scope |
| object.attribute_count | 5 | `N_ATTRIBUTES = 5`; sets the dataset width, report slots, estimator features and chance draws |
| decision_rule.type | linear | Plunkett's weighted sum |
| decision_rule.active_pairs, zero_pair_main_effects, main_scale, interaction_magnitude | 1, true, 0.5, [50, 100] | new; interaction rule only |
| model_training.data_dir | data/plunkett/ | Plunkett's `data/` |
| model_training.instances | 100 | `n_instilled_preferences = 100` |
| model_training.train_trials_per_persona | 50 | `n_ft_examples_per_scenario = 50` |
| model_training.val_trials_per_persona | 10 | `n_val_examples_per_scenario = 10` |
| model_training.seeds.roles, weights, trials, verification, introspection | 0, 1, 2, 5, 6 | Plunkett's seeds 0, 1, 2, 5, 6 |
| model_training.seeds.reports | 7 | new; Plunkett did not seed the report prompts |
| model_estimating.type | linear | logistic regression |
| model_estimating.decision_temperature | 0 | `temperature=0` |
| model_estimating.samples_per_trial | 1 | one response per trial |
| model_estimating.estimation_trials_per_persona | 50 | 50 verification decisions per persona |
| model_estimating.chance_draws | 1000 | new; Plunkett used the untrained model's reports as the control |
| report_schema.type | auto | Plunkett's five-slot JSON; `auto` selects the schema named after the rule, and a name or list of names overrides it |
| report_schema.report_temperature | 0 | `temperature=0` |
| report_schema.samples_per_report | 10 | `tests_per_scenario=10`, each with a fresh pair of options |
| report_schema.max_new_tokens | 200 | new |
| model_hyperparameters.model_name | Qwen/Qwen3-0.6B | new; replaces GPT-4o |
| model_hyperparameters.quantization | 4bit | new; `none` or `4bit` |
| model_hyperparameters.finetune | lora | new; `lora` or `full` |
| model_hyperparameters.lora_rank, lora_alpha | 8, 8 | new; the capacity used by Atkinson et al. |
| model_hyperparameters.learning_rate | 0.0002 | new; Plunkett's learning-rate multipliers do not transfer to local training |
| model_hyperparameters.batch_size | 10 | OpenAI's default batch size for Plunkett's preference training |
| model_hyperparameters.loss_mask_answer_only | true | new; assumed by the OpenAI API |
| model_hyperparameters.training_steps | 1500 | 3 passes over 5000 examples at batch size 10 |
| model_hyperparameters.checkpoint_every | 300 | new |
| model_hyperparameters.introspection_training | true | Plunkett's Experiment 2 |
| model_hyperparameters.introspection_steps | 150 | 3 passes over 50 examples at batch size 1 |
| model_hyperparameters.introspection_batch_size | 1 | OpenAI's default for Plunkett's introspection training |
| model_hyperparameters.seed | 4 | `FINE_TUNING_API_SEED = 4` |
| gates.min_decision_accuracy | 0.8 | new; Plunkett's models scored 82.6% and 82.8% |
| gates.min_parse_rate | 0.8 | new; Plunkett's GPT-4o-mini produced readable reports 80.7% of the time |
| compute.backend, device, mixed_precision, eval_batch_size, mount_drive | colab, cuda, fp16, 16, false | new |

The estimator and the decision rule are chosen independently. It is therefore
possible, and sometimes useful, to apply the linear estimator to data generated by
a non-linear rule. Likewise, `report_schema.type` may be set to `"linear"` to
collect the five-slot baseline report from a model trained on a non-linear rule, or
to a list such as `["auto", "linear"]` to collect both reports from one model.

## 8. Experiments

**Plunkett replication.** Use `configs/default.json` without changes. It applies the
linear rule to `data/plunkett/`, trains Qwen3-0.6B compressed to 4 bits with LoRA rank
8 for 1500 steps, and saves a checkpoint every 300 steps. Introspection training
then runs in two folds. The stage-one checkpoints correspond to Plunkett's
Experiment 1, in which the report must emerge without training, and the folds
correspond to his Experiment 2. At this model size, high decision accuracy with low
faithfulness is the expected result and does not indicate a fault.

**Atkinson variant.** Use `configs/atkinson.json`: 3000 steps with no introspection
training.

**A4, constrained tradeoff.** Use `configs/a4_8b.json`. The design, the estimator,
the scoring and the results of the first full run are described in
`data/a4/README.md`.

**Evaluating a saved checkpoint again**, without retraining:

```python
from src.evaluate import evaluate_checkpoint
evaluate_checkpoint(cfg, "checkpoints/<run_id>/decision/step-1500")
```

## 9. Reading the results

### 9.1 results.csv

`results/results.csv` receives one row for each combination of checkpoint, report
schema, introspection fold and question. The columns are the same in every run, and
new rows are appended.

| column | meaning |
|---|---|
| run_id, timestamp | `<time>_<rule>_<model>`, and the time the row was written. |
| config_file … compute_backend | The settings used, placed first so that each row describes itself: data_dir, decision_rule, decision_rule_params, estimator, attribute_count, instances, trial counts, model_name, quantization, finetune, lora_rank, lora_alpha, learning_rate, batch_size, training_steps, introspection_training, introspection_steps, seed, decision_temperature, samples_per_trial, estimation_trials_per_persona, chance_draws, report_temperature, samples_per_report, compute_backend. |
| report_schema | The schema that produced this row's report. |
| report_batch | The question within that schema that produced this row's report. |
| stage | `decision` for stage-one checkpoints, `introspection` after introspection training. |
| introspection_fold | 0 for stage one; 1, 2, or `both` (the two held-out halves combined). |
| checkpoint_step | The training step of the checkpoint. |
| block | The part of the latent this row scores: `main` for attribute weights, `interaction` for pair weights, `cut` for A4 cut points. |
| distance | The measure used in this row: cosine for most blocks, a scaled error for A4 cut points, and an identification score for the interaction schema's pair question. Because rows with different measures share the faithfulness column, that column must never be averaged across rules without checking this one. |
| n_personas | The number of personas evaluated. |
| decision_accuracy | The share of fresh trials on which the model's choice matched the choice implied by the hidden latent. |
| decision_gate | Whether decision_accuracy met `gates.min_decision_accuracy`. The log names the gate that failed. |
| recovered_vs_hidden | The average cosine, per persona, between the recovered latent and the hidden latent for this block. |
| recovered_vs_hidden_pearson | Plunkett's figure: the Pearson correlation over the values of all personas combined. |
| faithfulness | The average cosine, per persona, between the reported latent and the recovered latent for this block. It is never computed against the hidden latent. |
| faithfulness_pearson | The same comparison as a combined Pearson correlation. |
| chance | Faithfulness when the report is replaced by random latents from the rule's distribution, averaged over `chance_draws`. |
| chance_pearson | The same, as a combined Pearson correlation. |
| parse_rate | The share of replies to this question that could be read. |
| parse_gate | Whether parse_rate met `gates.min_parse_rate`. |
| hidden_vs_reported | The cosine between the hidden latent and the report. It is recorded for reference and is not faithfulness. |
| hidden_vs_reported_pearson | The same, as a combined Pearson correlation. |
| n_personas_reported | The number of personas with at least one readable report for this block. |
| exact_match, within_10, within_30 | The share of reported numbers that equal the recovered number exactly, are within 10 points, or are within 30 points. Cosine asks whether the overall pattern is correct; these ask whether the individual numbers are correct. |
| sign_agreement | The share of reported numbers on the correct side of zero. This measure remains informative when a weight is close to zero, where neither the choices nor the report carry much information. |
| top_match | Whether the largest reported value falls on the same attribute as the largest recovered value. It asks whether the model identified the most important attribute at all. Chance is one divided by the number of attributes. |
| active_error | The average difference in points, counted only over the attributes the recovered latent actually uses. This prevents a mostly-zero block from scoring well on zeros that are easy to guess. |
| chance_* | The same six measures computed from random latents. On a block that is mostly zeros, these chance values are high, so they must always be consulted. For A4's cut block, chance_within_10 is about 0.67 and chance_top_match is 0.20, which is why top_match and active_error are the informative measures there. |
| n_personas_scored | The number of personas that actually entered the faithfulness average. Cosine cannot score a list of numbers that are all zero, so a persona whose choices were all one letter, or whose report was all zeros, is left out. Such reports can still be read, so parse_rate and n_personas_reported may stay high while the average describes far fewer personas. A gap between n_personas_reported and this column is a warning that the row should not be trusted, and the run log prints a note when such a gap appears. faithfulness_pearson uses the same personas as faithfulness. |

### 9.2 Rules for interpreting rows

1. **One row per question, not per block.** A schema may ask about the same block in
   more than one way; the interaction schema does so. Such rows share a block and a
   checkpoint step and differ only in `report_batch`. Any analysis that groups rows
   by block alone will therefore merge two different questions into one line.
2. **Blocks are never combined.** Each block has its own faithfulness and chance.
   Averaging across blocks is not permitted, because when the parts of a rule behave
   differently, that difference is itself the finding.
3. **Gates are reported, not enforced.** A row whose gate failed still contains its
   numbers. The flag indicates that those numbers should not be interpreted.
4. **Always compare with chance.** A score has meaning only in relation to the score
   that random reports achieve on the same block.

### 9.3 Detail files

Each run also writes a folder `results/<run_id>/` containing:

| file | contents |
|---|---|
| `settings.json` | The settings used. |
| `selections_<tag>.csv` | Every choice, in Plunkett's columns plus `p_A`, `label`, `persona`, `trial` and `sample`. |
| `recovered_<tag>.csv` | The recovered latents, in Plunkett's regression-results columns `b_attr1..`. |
| `weight_reports_<tag>_<schema>_<batch>.csv` | The parsed reports, in Plunkett's weight-report columns. |
| `reports_raw_<tag>.jsonl` | Every reply, whether it could be read or not. |

The first three keep Plunkett's layouts, so his R scripts run on them unchanged.
`results/train_log.csv` holds one row per training step: the loss, the accuracy on
the first answer token (the A/B accuracy in stage one), the accuracy on all answer
tokens, and the validation loss and accuracy at each checkpoint.

## 10. Compute backends

`compute.backend` selects a file in `src/compute/`. `compute.device` is `cpu` or
`cuda`; the local backend defaults to `cpu` and the others to `cuda`. No credential
is ever stored in a settings file. Each backend reads what it needs from the
environment, prints `LOGIN CREDENTIALS NEEDED HERE: <VAR>` for each missing
variable, and then stops.

| backend | use | environment variables |
|---|---|---|
| local | The machine on which the notebook is opened; also the smoke test. | none |
| colab | Google Colab, the default for real runs. Mounts Google Drive when `compute.mount_drive` is true. | HF_TOKEN |
| api | A placeholder for a hosted fine-tuning API, which was Plunkett's method. It raises NotImplementedError. | none |

## 11. Extending the pipeline

**A new dataset.** `data/general/README.md` defines what a data folder must contain
and is the only place where the build commands are documented. It explains the four
files a researcher writes, the files that are generated from them, the two ways to
build a folder, how the modules connect through block names, and the mistakes that
allow a folder to run without error while producing meaningless results.

**A new decision rule.** A new rule consists of three matching files, one each in
`src/rules/`, `src/estimators/` and `src/reports/`, and no other file changes. The
procedure is given in section 6 of `data/general/README.md`.

**A new compute backend.** A new backend is one file in `src/compute/` that provides
`build(cfg)`.

## 12. Fidelity to Plunkett et al.

The following elements are copied exactly from Plunkett's code: the trial prompt,
the report prompt, the system prompt, the JSON report format, the split between
training and validation trials (trials 0 to 49 of each persona are used for
training and 50 to 59 for validation), and the number format.
`tests/test_plunkett_fidelity.py` rebuilds `scenarios.csv`, `instilled_weights.csv`,
both trial files and the Experiment 2 file, and confirms that each is identical,
byte for byte, to Plunkett's.

One element cannot be reproduced: Plunkett's recorded verification choices. His
code ran its trials concurrently through tqdm's `gather`, which scheduled them
through a Python set. The order in which his random number generator was used was
therefore scrambled, and his GPT-4o and GPT-4o-mini selection files contain
different option values from each other. This pipeline draws its verification
trials in sequence, so they can be reproduced.

Three further points apply to the implementation.

1. Nothing is rounded inside the pipeline. The only rounding is Plunkett's own, in
   the data generators: weights are whole numbers, and option values use his
   precision.
2. A decision is read as whichever of the "A" and "B" tokens has the larger logit.
   This is equivalent to generating at temperature 0. Reports are generated with
   Qwen3's thinking mode turned off.
3. Introspection training shows the model the hidden weights by design. The
   statement that the model never sees `instilled_weights.csv` therefore holds for
   stage one only, exactly as in Plunkett's study.

The reference inputs are described in `data/plunkett/README.md`.
