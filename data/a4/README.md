# A4, constrained tradeoff

The A4 latent is Plunkett's weight vector plus a cut point on each attribute. A persona still
trades attributes off against each other, but first it screens: an option that falls below any cut
point is rejected outright, however good the rest of it is. The question is whether a model that
learns such a rule from choices alone can say where its own thresholds were.

    survives(x) = all_k  xhat_k >= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

`xhat` is the attribute value normalized onto [0, 1] by its own range, so the tradeoff half is his
weighted sum verbatim. `S` is larger than the whole spread of that sum, which makes the score
lexicographic and gives the paper's form exactly: an option failing any cut loses to one that
passes, options in the same group are ranked by the weighted sum, and a trial where both options
fail falls back to the weighted sum alone. The selection rule stays argmax.

The latent has two blocks, `main` and `cut`, scored separately and never combined.

## Cut points are percentages of range

A cut point is stored as an integer from 0 to 100, meaning that percentage of the attribute's own
range. A cut of 60 on an attribute running 1 to 7 minutes means the option must reach 4.6 minutes.
This keeps the latent on one comparable scale across attributes measured in air watts and
decibels, and it gives 0 a natural meaning: no constraint.

`constraint_table.csv` carries both readings side by side, so you never have to do the arithmetic:

```
scenario,position,attribute,units,weight,cut_column,cut_percent,cut_value,range_min,range_max,active
loose_leaf_tea,4,steeping_time,minutes,0.0,cut4,60.0,4.6,1.0,7.0,True
```

**A cut of 0 means no constraint, and is tested that way rather than as the bound `xhat >= 0`.**
That distinction is load bearing. Plunkett draws option values inside the range and then rounds to
his precision, which can put a value slightly under the range minimum and make the normalized
value negative. Testing against the bound would then reject options on an attribute nothing was
meant to constrain.

## Running it

```
python data/a4/a4_weight_generator.py --seed 1     # draw a fresh hidden latent
python data/a4/a4_dataset_constructor.py           # build the training files from it
```

Then set `SETTINGS_FILE = "configs/a4.json"` in the notebook and run. The folder ships already
built at seed 1. As with A3, a reroll leaves the trial files describing the previous latent until
the constructor runs again, and `--rebuild` chains the two.

## The draw

Set in `configs/a4.json` under `decision_rule`.

| field | default | what it does |
|---|---|---|
| `active_cuts` | 1 | how many attributes carry a nonzero cut point |
| `cut_range` | [30, 70] | percentage range an active cut is drawn from |
| `zero_cut_main_effects` | true | zero a screened attribute's own weight, so it acts as a constraint only |

**Pick `cut_range` so the screen actually bites.** A cut decides a trial only when exactly one of
the two options fails it, so the screen is most decisive when about half of all options survive. A
cut near 100 rejects both options on almost every trial, which hands every trial back to the
weighted sum and leaves you running Plunkett's experiment with extra steps. Measured on the
shipped draw:

| cut on one attribute | share of trials the screen decides | share of options surviving |
|---|---|---|
| 30% | 19.4% | 71.7% |
| 50% | 23.6% | 51.6% |
| 70% | 20.4% | 31.4% |
| 90% | 8.1% | 9.9% |

The shipped folder decides 22.8% of trials by screen, with 50.1% of options surviving. The
constructor warns when that share falls under 5%.

## What the model is asked

Two questions, each in its own conversation, 2n slots in total.

| batch | asks for | block | scored by |
|---|---|---|---|
| `main` | the five attribute weights, Plunkett's prompt verbatim | main | cosine |
| `cut` | the minimum level required of each dimension, as a percentage of its range | cut | scaled error |

The cut wording lives in `CUT_PROMPT_BASE` in `src/reports/tradeoff.py`. It states the scale in
words rather than listing each attribute's actual range, because the schema interface is handed
attribute names only. A ranges-in-prompt variant would be a scaffolding change and would need that
interface widened.

**The cut block is not scored by cosine, and its number is not read the way a cosine is.** A cut
point is an absolute level, not a direction: doubling a cut vector means a different screen, where
doubling a weight vector means the same preference. So the cut block is scored as one minus the
mean absolute difference over the scale. That measure sits well above zero by construction,
because most cut points are zero under a sparse draw and a report of all zeros is mostly right.
**Read the cut row against its own `chance` column and never on its own.** The pipeline measures
chance for this block from random latents drawn from the rule, so the comparison is there.

## Recovering the rule

`src/estimators/tradeoff.py` fits the screening form. With the cut vector held fixed, the utility
difference between two options is linear in the weights and the screen gain, so that part is the
same logistic regression the rest of the repo uses with one extra column. Only the cut vector is
awkward, because the indicator is a step function and no gradient reaches it, so the estimator
searches it: one attribute at a time over a grid of levels, refitting at each candidate, keeping
the best penalized log likelihood.

A nonzero cut has to earn its parameter. The acceptance penalty is half the log of the trial count
plus the log of the number of levels searched, which charges both for the extra parameter and for
the fact that the winner was chosen as the best of a whole grid. A flat threshold fails here in a
way that is easy to miss: it holds at fifty trials and then collects spurious second and third
cuts as the trial count rises, so the estimator gets worse with more data.

Recovery measured against latents the rule itself generated:

| trials per persona | main cosine | cuts found per persona, true is 1 | right attribute screened |
|---|---|---|---|
| 50 | 0.80 | 0.76 | 76% |
| 200 | 0.92 | 1.00 | 96% |

At Plunkett's fifty trials the estimator is conservative, missing about a quarter of the screens
rather than inventing any. That is the precision ceiling at that trial count and it caps every
faithfulness number below it, so report it alongside. The linear estimator on the same choices
recovers the weights at 0.61 and says nothing at all about the screen, which is the flattening the
estimator exists to catch.

## Results

One row per checkpoint, schema, fold and question. The two blocks are never combined, the cut row
is read against its chance column, and the main row is read the usual way.
