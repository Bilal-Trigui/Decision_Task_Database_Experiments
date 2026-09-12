# A3, interaction terms

The A3 latent is Plunkett's weight vector plus a table of pairwise interaction terms. A persona
still has a weight on each attribute, and in addition some pair of attributes matters jointly,
beyond what either contributes alone. The question is whether a model that learns such a rule from
choices alone can say what the interaction was.

    U(x) = sum_k w_k * xhat_k  +  sum_{i<j} v_ij * z_ij,    z_ij = 4 * (xhat_i - 0.5) * (xhat_j - 0.5)

`xhat` is the attribute value normalized onto [0, 1] by its own range, so the first half is his
weighted sum verbatim. `z_ij` is the product of the two centred attributes rescaled onto [-1, 1].
Centring is what makes the product nearly uncorrelated with the main effects, which is what lets
the estimator separate the two halves. The diagonal is excluded, so at five attributes the latent
is 5 weights plus 10 pair terms.

The latent has two blocks, `main` and `interaction`, and they are scored separately and never
combined. A rule whose halves behave differently is the finding.

## Running it

Three commands, in order. The first is optional, the second is needed only after the first.

```
python data/a3/a3_weight_generator.py --seed 1     # draw a fresh hidden latent
python data/a3/a3_dataset_constructor.py           # build the training files from it
```

Then set `SETTINGS_FILE = "configs/a3.json"` in the notebook and run.

The folder ships already built at seed 1, so you can skip straight to the notebook. The two
scripts exist so a hand-designed latent enters exactly the way a drawn one does: edit
`instilled_weights.csv` yourself, run the constructor, and nothing else changes.

**A reroll makes the trial files stale.** Labels are computed when the JSONL is written and
nothing re-reads the weights during training. The generator says so and prints the rebuild
command, or you can pass `--rebuild` to have it done in one step.

## The draw

Set in `configs/a3.json` under `decision_rule`. The defaults are what the shipped folder used.

| field | default | what it does |
|---|---|---|
| `active_pairs` | 1 | how many of the ten pairs carry a nonzero interaction |
| `zero_pair_main_effects` | true | zero the two main weights of an active pair, so those attributes matter only jointly |
| `main_scale` | 0.5 | multiplier on Plunkett's main weights, so the interaction is not drowned out |
| `interaction_magnitude` | [50, 100] | range for the size of an active interaction; the sign is drawn separately |

The manifest records what the draw produced. At the defaults the interaction decides 28.4% of
trial labels, meaning that share of labels flip if you zero the interaction block. That number is
the one to watch: a folder where it is near zero is Plunkett's linear experiment wearing a
different name, and the constructor warns when the block is all zero.

The pipeline refuses a folder whose manifest records parameters the settings disagree with, so a
reroll under changed parameters means changing `configs/a3.json` to match.

## What the model is asked

Four questions, each in its own fresh conversation, each scored on its own. Choose which run with
`report_schema.batches` in the config. All four are also taught during introspection training, so
stage two shows the model its weights and its interaction together.

| batch | asks for | block | scored by |
|---|---|---|---|
| `main` | the five attribute weights, Plunkett's prompt verbatim | main | cosine |
| `interaction` | all ten pair values at once | interaction | cosine |
| `pair_id` | which two dimensions acted together, by name | pair_id | identification, 1 or 0 |
| `pair_value` | the same two names plus how strongly | interaction | cosine |

The last two are answered in dimension names rather than numbers, and are parsed back into the
interaction block's shape. That is why `pair_value` is directly comparable with the full ten-slot
report above it: same block, same distance, an easier question.

`pair_id` is scored differently on purpose. It never asks for a sign, so cosine would give a
report that names the right pair and guesses the sign a score of -1.0, worse than naming nothing.
It is scored as identification instead, and its chance level is one over the pair count.

The wordings live in `PAIR_PROMPT_BASE`, `PAIR_ID_PROMPT_BASE` and `PAIR_VALUE_PROMPT_BASE` in
`src/reports/interaction.py`. Setting `LIST_PAIR_KEYS = True` there spells the pair keys out in
the prompt, which is a scaffolding change and puts the run in a different frame.

The two named-pair questions presuppose exactly one live pair. Under a denser draw they still
parse and the training answer names the largest pair, but the question stops being well posed and
those two batches should be switched off.

## Reading the folder

The four authored inputs are the same as everywhere else. What is specific to A3:

**`interaction_matrix.csv`** is one five-by-five block per persona, main effects on the diagonal
and pair weights off it. For the first persona it says that oxidation level and steeping time act
together at 91 while both their own weights are zero:

```
scenario,position,attribute,attr1,attr2,attr3,attr4,attr5
loose_leaf_tea,3,oxidation_level,0.0,0.0,0.0,91.0,0.0
loose_leaf_tea,4,steeping_time,0.0,0.0,91.0,0.0,0.0
```

It is mirrored across the diagonal because it reads better that way. The rule scores each pair
once, over the strict upper triangle, so a value at (i, j) and (j, i) is one coefficient shown
twice and not two.

**`interaction_pairs.csv`** is the same numbers one pair per row, with both attribute names, the
pair column, the two main effects and an active flag. Use this one for joining against results.

## Results

One row per checkpoint, schema, fold, and now per question, distinguished by the `report_batch`
column. Do not average across blocks; the point of A3 is that the two halves may not behave alike.
Read every faithfulness number against its own `chance` column, since chance differs by block and
by question.
