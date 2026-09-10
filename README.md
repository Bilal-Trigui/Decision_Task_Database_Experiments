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
every other file in `configs/` is it with a few fields changed.
Six blocks mirror the environment frame, `gates` holds every threshold the pipeline
checks, and `compute` says where training runs. The table below lists every field.

**`configs/atkinson.json`** is the same file with `training_steps` 3000 and
`introspection_training` false: the Atkinson et al. variant, high steps and no
report training. **`configs/a3.json`** selects the interaction rule and estimator
and points at `data/a3/`. The two baseline papers and A3 are three settings files
for one pipeline.

**`configs/h100_*.json`** are the same experiments sized for an 80 GB Hopper card:
bf16 instead of fp16, no quantization, and a wider evaluation batch. `h100_smoke`
is Qwen3-0.6B for sixty steps with ten introspection steps per fold, so it proves
the CUDA path through both stages, including the adapter snapshot and restore
between folds, before a large model is loaded. `h100_plunkett_4b`,
`h100_plunkett_8b` and `h100_plunkett_14b` are Plunkett's settings otherwise.
There is no 32B file: its weights alone are past the disk budget a shared box
allows.

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

**`src/rules/`** holds one decision rule per file plus `base.py`. A rule declares
the latent's blocks and column names, samples a latent, scores an option, labels a
trial, and inherits `make_dataset`, which builds a fresh data folder from
Plunkett's personas and trials. `linear.py` is Plunkett's weighted sum;
`interaction.py` is A3. A4 (`tradeoff.py`) is not built yet; see the open items in the plan.

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
numbers; they remain in `plunkett-self-interpretability/data`. **`data/a3/`** was
built from this folder by the interaction rule.

**`tests/`** holds the fidelity test, the loss-mask test, and the recovery test, which fits each
estimator to choices generated by its own rule on synthetic personas: the precision ceiling of
the paper, and the check that generator and estimator share one feature convention.

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
| model_estimating.estimation_trials_per_persona | 50 | 50 verification decisions per agent |
| model_estimating.chance_draws | 1000 | new; he used the off-the-shelf model's reports as the control |
| report_schema.type | auto | his five-slot JSON; auto means the schema named after the rule, a name or a list of names overrides it |
| report_schema.report_temperature | 0 | `temperature=0` |
| report_schema.samples_per_report | 10 | `tests_per_scenario=10`, each with a fresh option pair |
| report_schema.max_new_tokens | 200 | new |
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
| model_hyperparameters.seed | 4 | `FINE_TUNING_API_SEED = 4` |
| gates.min_decision_accuracy | 0.8 | new; his models scored 82.6% and 82.8% |
| gates.min_parse_rate | 0.8 | new; his GPT-4o-mini parsed 80.7% |
| compute.backend, device, mixed_precision, eval_batch_size, mount_drive | colab, cuda, fp16, 16, false | new |

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

**A3, interaction terms.** `configs/a3.json`: `decision_rule.type` interaction with
its draw parameters, `model_estimating.type` interaction, `data_dir` `data/a3/`.
Everything else is Plunkett's. The pair prompt's wording is `PAIR_PROMPT_BASE` in
`src/reports/interaction.py`; set `LIST_PAIR_KEYS = True` there to spell the pair
keys out in the prompt.

**Re-evaluating a saved checkpoint** without retraining:
`from src.evaluate import evaluate_checkpoint; evaluate_checkpoint(cfg, "checkpoints/<run_id>/decision/step-1500")`.

## Making a new dataset with a rule module

Every rule module is a command line:

```
python -m src.rules.interaction --source data/plunkett --out data/a3 --seed 1
python -m src.rules.interaction --source data/plunkett --out data/a3_dense --seed 1 \
    --param active_pairs=10 --param main_scale=1.0 --param zero_pair_main_effects=false
python -m src.rules.linear --source data/plunkett --out data/plunkett_regen   # reproduces his files
```

The output folder keeps `candidate_scenarios.json`, `roles.csv` and `scenarios.csv`
(truncated when `--attribute-count` is smaller than the source), rerolls
`instilled_weights.csv` under the rule with `--seed` (columns `attr1..attrN` first,
then the rule's extra latent columns), regenerates Plunkett's trials with
`--trials-seed` 2 and relabels them into `instill_<instances>_prefs.jsonl` and
its `_val` file, and writes `manifest.json` with the seed, the rule parameters,
file hashes, and rule-specific statistics. At load, the pipeline refuses a folder
whose manifest was built with different rule parameters than the settings say.

## Adding a fourth rule

Write `src/rules/<name>.py` (subclass `Rule`: `blocks`, `sample_latent`, `score`,
plus `build(params)`), `src/estimators/<name>.py` (subclass `Estimator`: `blocks`,
`features`, plus `build(params)`), `src/reports/<name>.py` (subclass `Schema`:
`batches`, plus `build(params)`), build a data folder with the rule module, and set
the three names and `data_dir` in a settings file. No other file changes. Blocks are
matched by name across the three, and every vector rule calls its weight block
`main`. A new compute backend is one file in `src/compute/` with `build(cfg)`.

## results.csv

One row per checkpoint, report schema, introspection fold and block. The columns
are the same on every run; the file is appended to.

| column | meaning |
|---|---|
| run_id, timestamp | `<time>_<rule>_<model>` and when the row was written |
| config_file … compute_backend | the settings fields, first, so a row is self-describing (data_dir, decision_rule, decision_rule_params, estimator, attribute_count, instances, trial counts, model_name, quantization, finetune, lora_rank, lora_alpha, learning_rate, batch_size, training_steps, introspection_training, introspection_steps, seed, decision_temperature, samples_per_trial, estimation_trials_per_persona, chance_draws, report_temperature, samples_per_report, compute_backend) |
| report_schema | which schema this row's report came from |
| stage | `decision` for stage-one checkpoints, `introspection` after report training |
| introspection_fold | 0 for stage one; 1, 2, or `both` (the two held-out halves pooled) |
| checkpoint_step | training step of the checkpoint evaluated |
| block | which part of the latent the row scores: `main` for the attribute weights, `interaction` for the pair weights |
| n_personas | personas evaluated |
| decision_accuracy | share of verification trials where the model's choice matched the hidden latent's label |
| decision_gate | whether decision_accuracy cleared gates.min_decision_accuracy; the log names the gate that fired |
| recovered_vs_hidden | mean per-persona cosine between the estimator's latent and instilled_weights.csv, this block |
| recovered_vs_hidden_pearson | Plunkett's number: Pearson r over all personas' values of this block pooled |
| faithfulness | mean per-persona cosine between the reported latent and the recovered latent, this block; never against the hidden latent |
| faithfulness_pearson | the same, pooled Pearson |
| chance | faithfulness with the report replaced by random latents from the rule's distribution, averaged over chance_draws |
| chance_pearson | the same, pooled Pearson |
| parse_rate | share of report replies for this block's batch that parsed |
| parse_gate | whether parse_rate cleared gates.min_parse_rate |
| hidden_vs_reported | cosine between instilled_weights.csv and the report; recorded, not faithfulness |
| hidden_vs_reported_pearson | the same, pooled Pearson |
| n_personas_reported | personas with at least one parsed report in this block |

Blocks are never pooled. The main-effect weights and the interaction weights are
separate rows with their own faithfulness and chance, cosine per block with
pooled Pearson beside it, and no column ever combines them into one number. Do
not average across blocks later; a rule whose parts behave differently is the
finding. A4's blocks follow the same rule. Gates are flagged, not enforced:
a row with a fired gate still has its numbers, and the flag says not to interpret
them.

Per-run detail files live in `results/<run_id>/`: `settings.json`,
`selections_<tag>.csv` (Plunkett's selections columns plus `p_A`, `label`,
`persona`, `trial`, `sample`), `recovered_<tag>.csv` (his regression-results
columns `b_attr1..`), `weight_reports_<tag>_<schema>_<batch>.csv` (his
weight-reports columns), `reports_raw_<tag>.jsonl` (every reply, parsed or not).
The first three keep his layouts, so his R scripts run on them.
`results/train_log.csv` holds one row per step: loss, first-answer-token accuracy
(the A/B accuracy in stage one), all-answer-token accuracy, and validation loss and
accuracy at checkpoints.

## Compute backends

`compute.backend` selects a class in `src/compute/`; `compute.device` is `cpu` or
`cuda` (local defaults to cpu, the others to cuda). No credential ever lives in a
settings file. Each backend reads what it needs from the environment and prints
`LOGIN CREDENTIALS NEEDED HERE: <VAR>` for each missing variable, then stops.

| backend | use | environment variables |
|---|---|---|
| local | the machine the notebook is opened on; smoke test | none |
| colab | Google Colab, default for real runs; mounts Drive when `compute.mount_drive` is true | HF_TOKEN |
| azure | an Azure GPU VM or ML workspace | AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE, HF_TOKEN |
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
