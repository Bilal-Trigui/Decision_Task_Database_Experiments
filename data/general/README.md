# Building a data folder

A data folder supplies the inputs of one experiment. It contains four files written
by the researcher, which define the choices and the hidden preferences behind them,
and several generated files, which the pipeline actually trains on. This document
explains what each file is, which tool writes it, how the three kinds of module
connect to one another, and the common errors that allow a folder to run without
complaint while producing meaningless results.

This folder holds no data of its own. It contains this document and the two tools
that every experiment uses to build its data: `vector_dataset_constructor.py` and
`vector_weight_generator.py`. The reference data folder is `data/plunkett/`. The
folder `data/a4/` applies the constrained tradeoff rule to Plunkett's scenarios with
a different hidden latent.

## Contents

1. The four authored files
2. The generated files
3. Building a folder
4. How the three modules connect
5. Five questions every new rule must answer
6. Adding a rule
7. Common errors

## 1. The four authored files

Every data folder must contain these four files. When the settings are loaded,
`src/data.check_attribute_count` refuses any folder in which one is missing.

**`candidate_scenarios.json`** is the pool of choice types. It is a JSON list with
one object per choice type. Each object has a `short_name`, a `question`, and a list
of `attributes`. Each attribute has a `name`, its `units`, and a two-element `range`.
The ranges set the scale on which the whole experiment is measured, for three
reasons: option values are drawn within them, the decision rule rescales by them,
and the estimator divides its features by them. Plunkett's file contains 1365
choice types.

```json
{
  "short_name": "loose_leaf_tea",
  "question": "Which loose leaf tea would you prefer?",
  "attributes": [
    {"name": "caffeine_content", "units": "mg per cup", "range": [0, 70]},
    {"name": "leaf_size", "units": "millimeters", "range": [0.1, 10]}
  ]
}
```

**`roles.csv`** is a single column of character names, read with `header=None`. The
file has no header row. Plunkett's file begins with the literal word `role`. That
word is therefore treated as a persona name rather than a column title, and the
pipeline preserves this behaviour so that his assignment of roles to scenarios is
reproduced.

**`scenarios.csv`** attaches one role to each choice type. Its columns are fixed, and
they are checked exactly in this order:

```
scenario, question, attr1..attrN, attr1_min..attrN_min, attr1_max..attrN_max
```

Each `question` reads `Imagine you are NAME. Which X would you prefer?`. For every
row, the attribute names and both range limits must agree with
`candidate_scenarios.json`; otherwise `load_scenarios` raises an error. The function
`src.data.assemble_scenarios` builds this file from the choice types and the roles,
using the roles seed, which is 0 in every settings file. No constructor calls it
automatically. Plunkett's file contains 1100 rows.

**`instilled_weights.csv`** holds the hidden latent, one row per scenario. It begins
with a `scenario` column, followed by the latent columns that the decision rule
declares, in the rule's order. This is the only file that differs between the
baseline and a new experiment, and it is the only place where a hand-designed latent
is entered.

| rule | latent columns after `scenario` |
|---|---|
| linear | `attr1..attrN` |
| interaction | `attr1..attrN`, `v_1_2 .. v_(N-1)_N` |
| tradeoff | `attr1..attrN`, `cut1..cutN` |

Two row counts must be sufficient. `scenarios.csv` needs at least
`model_training.instances` rows, because the pipeline takes its first `instances`
rows as the personas. `instilled_weights.csv` needs a row for every scenario that is
actually used.

## 2. The generated files

None of the following files is written by hand. A constructor writes all of them
from the four inputs above. Running the constructor again is the only way to keep
them consistent with the weights.

| file | contents |
|---|---|
| `instill_<N>_prefs.jsonl` | For each of N personas, `train_trials_per_persona` decision trials labelled by the rule, using trials seed 2. |
| `instill_<N>_prefs_val.jsonl` | The validation trials, drawn from the same random sequence. |
| `instilled_weights_<N>_training.jsonl` | Plunkett's Experiment 2 file: the report prompt answered with the hidden weights, using introspection seed 6. |
| `introspection_training.csv` | The same stage-two examples, in the layout the pipeline reads at run time. |
| `manifest.json` | The rule, its parameters, the seeds, the counts, a fingerprint of every input and output, and summary statistics for the rule. |

`introspection_training.csv` is listed in `.gitignore`, because it is derived from
`instilled_weights.csv` and is not an input.

## 3. Building a folder

There are two ways to build a folder. They differ in one respect only: the rule
module draws a new latent, whereas the constructor uses the latent exactly as it is
written.

### 3.1 Drawing a new latent

The following commands draw a fresh latent under a rule and build every file:

```
python -m src.rules.linear      --source data/plunkett --out data/plunkett_regen
python -m src.rules.interaction --source data/plunkett --out data/interaction --seed 1
python -m src.rules.tradeoff    --source data/plunkett --out data/a4 --seed 1
python -m src.rules.interaction --source data/plunkett --out data/interaction_dense --seed 1 \
    --param active_pairs=10 --param main_scale=1.0 --param zero_pair_main_effects=false
```

Each command performs five steps.

1. It copies `candidate_scenarios.json`, `roles.csv` and `scenarios.csv` to the output
   folder, shortening `scenarios.csv` if `--attribute-count` is smaller than in the
   source.
2. It draws a new `instilled_weights.csv` under the rule with `--seed`. The columns
   `attr1..attrN` come first, followed by the rule's additional columns.
3. It regenerates Plunkett's trials with `--trials-seed` 2.
4. It labels those trials under the new latent and writes them to
   `instill_<instances>_prefs.jsonl` and the matching `_val` file.
5. It writes `manifest.json` with the seed, the rule parameters, file fingerprints
   and rule statistics.

When the pipeline later loads the folder, it refuses to proceed if the manifest
records different rule parameters from those in the settings.

### 3.2 Rebuilding from the existing weights

The following commands rebuild the training files from the weights already in a
folder, without drawing new ones:

```
python data/general/vector_dataset_constructor.py --data data/plunkett      # Plunkett's three JSONL files, byte for byte
python data/general/vector_dataset_constructor.py --data data/mine --rule linear
python data/general/vector_dataset_constructor.py --data a4                 # a bare folder name also works
python data/a4/a4_dataset_constructor.py                                    # the same, through the A4 wrapper
python data/a4/a4_weight_generator.py --seed 7 --rebuild                    # draw a new A4 latent and rebuild in one step
```

### 3.3 How the two tools divide the work

The generator writes `instilled_weights.csv` and nothing else. The constructor turns
that latent into every training file. The two are deliberately separate, so that a
hand-designed latent and a randomly drawn latent enter the pipeline in exactly the
same way. Each experiment folder contains small wrapper scripts that point both
tools at that folder, so the commands work with no arguments.

Neither tool contains knowledge of any particular experiment. Instead, each rule
supplies three optional methods:

- `views` names any additional readable tables the latent deserves.
- `audit` names the disagreements between an authored latent and the rule's
  parameters that are worth a warning.
- `summarise_draw` describes what a random draw produced.

All three do nothing by default. The weighted sum therefore needs none of them, and
a new rule obtains a working constructor and generator without either being written
for it.

The A4 constructor, for example, uses `views` to write `constraint_table.csv`, which
has one row per persona and attribute and shows the weight beside the cut point, both
as a percentage and in the attribute's own units. It uses `audit` to compare the
authored weights with the declared rule parameters, and it prints warnings rather
than failing, because a hand-designed latent may depart from the random draw on
purpose.

### 3.4 Drawing a new latent does not update the trials

Labels are written into the trial files when they are built, and nothing reads the
weights again during training. After a new draw, the trial files therefore still
describe the previous latent. The generator states this and prints the command to
run next.

The procedure for testing a hand-designed latent is therefore:

1. Edit `instilled_weights.csv`.
2. Run the constructor.
3. Run the pipeline.

## 4. How the three modules connect

A rule consists of three files whose block names match.

| file | role |
|---|---|
| `src/rules/<name>.py` | generates choices from a latent |
| `src/estimators/<name>.py` | recovers a latent from choices |
| `src/reports/<name>.py` | asks the model for its latent |

`src/config.py` finds each file by the name given in the settings and calls its
`build()` function. There is no central list, so adding a module means only adding
a file.

Blocks are what connect the three files. A rule's `blocks(n)` maps each block name to
its latent column names, and the estimator and report schema declare the same names.
`evaluate.py` divides the hidden, recovered and reported latents by block and scores
each block separately. Blocks are never combined, because the parts of a rule may
behave differently, and that difference is the result of interest.

| rule | blocks | estimator | report slots when n = 5 |
|---|---|---|---|
| linear | `main` | linear | 5, in one batch |
| interaction | `main`, `interaction` | interaction | 15, in two batches |
| tradeoff | `main`, `cut` | tradeoff | 10, in two batches |

Every vector rule calls its weight block `main`. A report schema divides its slots
into batches, each collected in a separate prompt, and each batch names the block it
reports on. A batch that names a block the rule does not define raises an error
when it is built.

## 5. Five questions every new rule must answer

Before a run, a rule and its latent must answer all five of the following questions.
A design that cannot answer one of them is not ready, because each unanswered
question is a point at which the experiment can fail without any visible sign.

1. **What must the model learn?** The latent, not the individual choices. For the
   linear rule, this is the weight vector `w`.
2. **Which estimator recovers it?** The estimator must be the decision rule run in
   reverse. For a weighted sum, the probability of a choice depends only on the
   product of the weights with the difference between the two options. Logistic
   regression on the normalised differences is therefore exactly matched to it.
3. **What does "close" mean?** Multiplying `w` by a positive number changes every
   score but no choice, so what matters is the direction of `w`. Cosine similarity
   measures direction, and it is therefore the appropriate distance. Pooled Pearson
   correlation is reported beside it, because it is Plunkett's headline figure.
4. **What does the report look like?** The report consists of numbers in fixed
   positions, so it lies in the same space as the estimator's output. A distance
   can only be measured between two points in the same space.
5. **What does a random answer look like?** `src/chance.py` draws latents from the
   rule's own `sample_latent` and scores them as if they were reports. Chance is
   measured for each block and is never assumed to be zero.

## 6. Adding a rule

1. Write `src/rules/<name>.py`, a subclass of `Rule` with `blocks`, `sample_latent`,
   `score` and `build(params)`.
2. Write `src/estimators/<name>.py`, a subclass of `Estimator` with `blocks`,
   `features` and `build(params)`.
3. Write `src/reports/<name>.py`, a subclass of `Schema` with `batches` and
   `build(params)`.
4. Build a data folder with the new rule module.
5. Set the three names and `data_dir` in a settings file.

No other file needs to change.

Two optional additions are worth their cost. First, a `dataset_stats` method on the
rule places a diagnostic in the manifest, such as the share of trial labels that
change when the new block is set to zero. This shows whether the dataset actually
tests what it was built to test. Second, a constructor wrapper in the data folder can
add a readable view of the new block.

## 7. Common errors

Each of the following errors allows a folder to run without complaint while
producing meaningless results. Every one of them has occurred in practice.

**Editing the weights does not relabel the trials.** Labels are computed when the
folder is built and are written into the JSONL files. Nothing reads
`instilled_weights.csv` again during training. If the weights are changed without
running a constructor, the model trains on labels from the old latent, while every
report is scored against the new one.

**`instilled_weights_<N>_training.jsonl` contains the hidden weights, not recovered
ones.** It pairs the introspection prompt with the hidden weights as the desired
reply. Recovered weights cannot exist when the folder is built, because they come
from the model's own choices after training; they are stored in
`results/<run>/recovered_*.csv`. Plunkett chose the hidden weights as targets
deliberately and justifies the choice in his footnote 5.

**`roles.csv` has no header.** The first line is a persona. If the file is written
with a header row, every role shifts by one position, and each character is silently
attached to a different choice type.

**The ranges must agree across two files.** `scenarios.csv` contains the range limits,
and `candidate_scenarios.json` contains them again together with the units.
`load_scenarios` checks every row and raises an error at the first disagreement. If
one file is edited, the other must be edited to match.

**The weights file must contain every column the rule declares.** If the interaction
rule is pointed at a linear weights file, it raises an error that names the missing
pair columns. This error is visible, which is intended, but it appears only if the
rule named in the settings matches the folder.

**`object.attribute_count` must match the width of the folder.** The check counts the
`attrN` columns in both `scenarios.csv` and `instilled_weights.csv` and refuses a
mismatch. Additional latent columns such as `v_1_2` or `cut1` are deliberately not
counted.

**Rounding can place an option value below its own range minimum.** Values are drawn
uniformly within the range and then rounded to Plunkett's precision, which is zero
decimal places whenever the range spans 5 or more. An attribute ranging from 0.1 to
10 can therefore draw 0.4 and round it to 0, which gives a negative normalised value.
This happens to about one value in ten thousand. Because it is faithful to Plunkett,
the generator keeps it. Any new rule that compares a normalised value with a limit
must allow for it. The tradeoff rule treats a cut point of 0 as "no constraint",
rather than as the test `xhat >= 0`, for exactly this reason.

**The manifest fixes the rule parameters.** `check_manifest` raises an error when the
manifest records parameters that differ from the settings, and it warns when the rule
name differs. The correct response is to rebuild the folder or change the settings.
The manifest should never be edited by hand.

**A manifest records an absolute path.** The field `weights_used_as_given_from` is
stored exactly as it was given. A folder built on one machine can therefore carry
that machine's home directory into git. The path should be made relative before a
tracked manifest is committed.
