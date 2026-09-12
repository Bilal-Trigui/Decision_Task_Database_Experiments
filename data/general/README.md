# Building a data folder for the vector paradigm

A data folder is the input half of an experiment. It holds four authored files that
say what the choices are and what hidden latent drives them, and the generated files
the pipeline actually trains on. This document says what each file is, what writes
it, how the three module kinds bind together, and the ways a folder fails quietly.

`data/plunkett/` is the reference folder. `data/a3/` is the interaction rule and
`data/a4/` the constrained tradeoff rule, each built from Plunkett's scenarios with
a different latent. This folder holds no data of its own.

## The four authored files

Every data folder must hold these four. `src/data.check_attribute_count` runs when
the settings load and refuses a folder missing any of them.

**`candidate_scenarios.json`** is the pool of choice types. A JSON list, one object
per type, each with `short_name`, `question`, and an `attributes` list whose entries
carry `name`, `units`, and a two-element `range`. The ranges are the scale the whole
experiment is read against: option values are drawn inside them, the rule normalizes
by them, and the estimator divides its features by them. Plunkett's file holds 1365
candidates.

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

**`roles.csv`** is one column of character names, read with `header=None`. There is
no header row. Plunkett's file begins with the literal word `role`, which is
therefore a persona name and not a column title, and the pipeline keeps that
behavior so his scenario assignment reproduces.

**`scenarios.csv`** glues one role to each choice type. The columns are fixed and
checked exactly, in this order:

```
scenario, question, attr1..attrN, attr1_min..attrN_min, attr1_max..attrN_max
```

The `question` reads `Imagine you are NAME. Which X would you prefer?`. The attribute
names and both range bounds must agree with `candidate_scenarios.json` for every row,
or `load_scenarios` raises. `src.data.assemble_scenarios` builds this file from the
definitions and the roles under the roles seed, which is 0 in every config, but no
constructor calls it for you. Plunkett's file holds 1100 rows.

**`instilled_weights.csv`** is the hidden latent, one row per scenario. A `scenario`
column, then the latent columns the decision rule declares, in the rule's order.
This is the only file that differs between the baseline and a new experiment, and
the only place a hand-designed latent goes.

| rule | latent columns after `scenario` |
|---|---|
| linear | `attr1..attrN` |
| interaction | `attr1..attrN`, `v_1_2 .. v_(N-1)_N` |
| tradeoff | `attr1..attrN`, `cut1..cutN` |

Row counts: `scenarios.csv` needs at least `model_training.instances` rows, and
`instilled_weights.csv` needs a row for every scenario actually used. The pipeline
takes the first `instances` rows of `scenarios.csv` as its personas.

## The generated files

None of these are authored. A constructor writes all of them from the four inputs
above, and rerunning it is the only way to keep them consistent with the weights.

| file | what it is |
|---|---|
| `instill_<N>_prefs.jsonl` | N personas by `train_trials_per_persona` decision trials, labelled by the rule, trials seed 2 |
| `instill_<N>_prefs_val.jsonl` | the validation trials off the same stream |
| `instilled_weights_<N>_training.jsonl` | Plunkett's Experiment 2 file: the report prompt answered with the hidden weights, introspection seed 6 |
| `introspection_training.csv` | the same stage-two examples in the layout the pipeline reads at run time |
| `manifest.json` | rule, parameters, seeds, counts, a hash of every input and output, and rule statistics |

`introspection_training.csv` is in `.gitignore`, since it is derived from
`instilled_weights.csv` and is not an input.

## Building a folder

Two paths, and they differ in one respect only. The rule module rerolls the latent.
The constructor uses the latent exactly as written.

Draw a fresh latent under a rule and build everything:

```
python -m src.rules.linear      --source data/plunkett --out data/plunkett_regen
python -m src.rules.interaction --source data/plunkett --out data/a3 --seed 1
python -m src.rules.tradeoff    --source data/plunkett --out data/a4 --seed 1
python -m src.rules.interaction --source data/plunkett --out data/a3_dense \
    --param active_pairs=10 --param zero_pair_main_effects=false
```

Rebuild the training files from the weights already in a folder, never rerolling:

```
python data/plunkett/vector_dataset_constructor.py --data data/plunkett
python data/plunkett/vector_dataset_constructor.py --data data/mine --rule linear
python data/a3/a3_dataset_constructor.py
python data/a4/a4_dataset_constructor.py
```

The A3 and A4 constructors wrap the general one with their rule as the default and
add a readable view of the extra block. A3 writes `interaction_matrix.csv`, an N by
N block per persona with main effects on the diagonal and pair weights off it, plus
`interaction_pairs.csv` for joining. A4 writes `constraint_table.csv`, one row per
persona and attribute carrying the weight beside the cut point in both percent and
the attribute's own units. Both audit the authored weights against the declared rule
parameters and print warnings rather than failing, since a hand-designed latent may
depart from the draw deliberately.

The loop for testing a latent you designed is: edit `instilled_weights.csv`, run the
constructor, run the pipeline.

## The three modules and what binds them

A rule is three files with matching block names. `src/rules/<name>.py` generates,
`src/estimators/<name>.py` recovers, and `src/reports/<name>.py` asks. `src/config.py`
resolves each by name from the settings and calls its `build()`. There is no registry,
so adding a module is adding a file.

Blocks are the binding. A rule's `blocks(n)` maps a block name to its latent column
names, and the estimator and report schema declare the same names. `evaluate.py`
slices the hidden, recovered and reported vectors by block and scores each block on
its own. Nothing is ever combined across blocks, because the halves of a rule can
behave differently and that difference is the finding.

| rule | blocks | estimator | report slots at n of 5 |
|---|---|---|---|
| linear | `main` | linear | 5, one batch |
| interaction | `main`, `interaction` | interaction | 15, two batches |
| tradeoff | `main`, `cut` | not built yet | 10, two batches |

Every vector rule calls its weight block `main`. A report schema splits its slots
into batches, each collected in its own prompt, and a batch names the block it
reports on. A batch whose block the rule does not define raises at construction.

## The five answers a parameter space owes

Before a run, a space and rule must answer all five. A space that cannot answer one
is not ready, and each is a place a design fails without saying so.

1. **What the model has to learn.** The latent, not the choices. For the weight
   vector this is `w`.
2. **Which estimator recovers it.** The decision rule run backwards. For a weighted
   sum the choice probability depends only on the dot product with the difference
   vector, so logistic regression on normalized differences is exactly matched.
3. **What closeness means.** Scaling `w` changes every score and no comparison, so
   the object is a direction and cosine is the distance that respects that. Pooled
   Pearson travels beside it as Plunkett's headline figure.
4. **What the report looks like.** Numbers in fixed positions, so the report lands
   in the same space the estimator lands in. A distance needs both ends there.
5. **What random looks like.** `src/chance.py` draws latents from the rule's own
   `sample_latent` and scores them as if they were reports. Chance is measured per
   block and never assumed to be zero.

## Adding a rule

Write `src/rules/<name>.py` subclassing `Rule` with `blocks`, `sample_latent`,
`score`, and a `build(params)`. Write `src/estimators/<name>.py` subclassing
`Estimator` with `blocks`, `features`, and `build(params)`. Write
`src/reports/<name>.py` subclassing `Schema` with `batches` and `build(params)`.
Build a data folder with the rule module. Set the three names and `data_dir` in a
settings file. No other file changes.

Two optional pieces earn their cost. `dataset_stats` on the rule puts a diagnostic in
the manifest, such as the share of trial labels that flip when the new block is
zeroed, which is how you learn whether the dataset tests the thing it was built to
test. A constructor in the data folder adds a readable view of the new block.

## Traps

These are the ways a folder runs clean and means nothing. Every one has bitten.

**Editing the weights does not relabel the trials.** Labels are computed at
construction time and written into the JSONL. Nothing re-reads
`instilled_weights.csv` during training. Change the weights without rerunning a
constructor and the model trains on labels from the old latent while every report is
scored against the new one.

**`instilled_weights_<N>_training.jsonl` holds the hidden weights, not recovered
ones.** It is the introspection prompt paired with the target weights as the desired
reply. Recovered weights do not exist at construction time; they come from the
model's own choices after training and live in `results/<run>/recovered_*.csv`.
Plunkett chose the targets deliberately and defends the choice in his footnote 5.

**`roles.csv` has no header.** The first line is a persona. A file written with a
header column shifts every role by one and silently changes which character is
attached to which choice type.

**Ranges must agree across two files.** `scenarios.csv` carries the bounds and
`candidate_scenarios.json` carries them again with the units. `load_scenarios` checks
every row and raises on the first disagreement. Edit one and you must edit both.

**The weights file must carry every column the rule declares.** Pointing the
interaction rule at a linear weights file raises and names the missing pair columns.
This one fails loudly, which is the point, but only if the rule in the settings
matches the folder.

**`object.attribute_count` must match the folder's width.** The check counts `attrN`
columns in both `scenarios.csv` and `instilled_weights.csv` and refuses a mismatch.
Extra latent columns such as `v_1_2` or `cut1` are not counted, by design.

**Rounding can push an option below its own range minimum.** Values are drawn
uniformly inside the range and then rounded to Plunkett's precision, which is zero
decimal places whenever the range spans 5 or more. An attribute ranging 0.1 to 10
can draw 0.4 and round to 0, so its normalized value is negative. This is rare, about
one value in ten thousand, and it is faithful to Plunkett, so the generator keeps it.
Any new rule that compares a normalized value against a bound must handle it. The
tradeoff rule treats a cut point of 0 as no constraint rather than as the bound
`xhat >= 0` for exactly this reason.

**The manifest pins the rule parameters.** `check_manifest` raises when a folder's
manifest records parameters the settings disagree with, and warns when the rule name
differs. Rebuild the folder or change the settings; do not edit the manifest.

**A manifest records an absolute path.** `weights_used_as_given_from` is written as
given, so a folder built on one machine carries that machine's home directory into
git. Worth making relative before committing a tracked manifest.
