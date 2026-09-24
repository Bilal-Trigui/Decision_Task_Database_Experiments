# A4: the constrained tradeoff

## Contents

1. Research question
2. The decision rule
3. Cut points as percentages of range
4. Running the experiment
5. Parameters of the random draw
6. The report questions
7. Recovering the rule from choices
8. Results of the first full run
9. Recommended changes before publication

## 1. Research question

In Plunkett's experiment, every persona chooses by a weighted sum: each attribute of
an option is multiplied by a weight, and the option with the larger total is chosen.
Many real decisions do not work this way. People often first reject any option that
fails a minimum requirement, and only then weigh the remaining options against each
other. A buyer may, for example, reject any tea that steeps too quickly, however good
it is in other respects.

A4 adds such a minimum requirement, called a **cut point**, to Plunkett's weighted
sum. The question is whether a model that learns this rule from choices alone can
later state where its own cut points lie.

## 2. The decision rule

The rule is written as follows:

    survives(x) = all_k  xhat_k >= c_k
    U(x)        = S * survives(x)  +  sum_k w_k * xhat_k,    S = sum_k |w_k| + 1

The symbols have these meanings:

- `xhat_k` is the value of attribute `k`, rescaled onto the interval from 0 to 1 by
  that attribute's range.
- `c_k` is the cut point on attribute `k`.
- `survives(x)` is 1 if the option meets every cut point and 0 otherwise.
- `w_k` is the weight on attribute `k`, exactly as in Plunkett's rule.
- `S` is a bonus given to every option that survives the screen.

The reasoning behind the formula proceeds in three steps.

1. `S` is larger than the full range the weighted sum can take. A surviving option
   therefore always scores higher than a failing one, whatever their weighted sums.
2. Among options that are both surviving, or both failing, the bonus is equal, so
   the weighted sum alone decides between them.
3. The result is a two-level ordering: first the screen, then the weighted sum. The
   model still chooses the option with the higher score.

The latent has two blocks, `main` (the weights) and `cut` (the cut points). They are
scored separately and are never combined.

### A worked trial

The persona Djedefre chooses between two loose leaf teas. Its hidden weights are
−100, 95, 72, 0 and −1, and it has a cut point of 60% on attribute 4, steeping time.

| attribute | option A | option B | weight | note |
|---|---|---|---|---|
| caffeine_content | 67.0 (0.96) | 52.0 (0.74) | −100 | |
| leaf_size | 9.0 (0.90) | 7.0 (0.70) | 95 | |
| oxidation_level | 6.0 (0.06) | 31.0 (0.31) | 72 | |
| steeping_time | 2.0 (0.17) | 5.0 (0.67) | 0 | cut at 60%: A fails, B passes |
| resteep_potential | 7.0 (0.86) | 5.0 (0.57) | −1 | |

Option A fails the screen and scores −6.85. Option B passes and scores 282.67. The
correct label is therefore B.

This trial shows why the rule is difficult to learn. On the weighted sum alone, A
would win, because it leads on leaf size, which carries a weight of 95. The screen
overrides that. Moreover, the screened attribute has a weight of zero, so steeping
time matters only as a requirement and not as a preference. A model that has learned
only a weighted sum has no way to choose correctly here.

## 3. Cut points as percentages of range

A cut point is stored as a whole number from 0 to 100, meaning that percentage of the
attribute's own range. For example, a cut of 60 on an attribute that runs from 1 to 7
minutes requires the option to reach 1 + 0.60 × 6 = 4.6 minutes. This convention has
two advantages. First, it places every cut point on the same scale, even though the
attributes are measured in different units, such as air watts and decibels. Second,
it gives 0 a natural meaning: no constraint.

`constraint_table.csv` shows both forms side by side, so the conversion never has to
be done by hand:

```
scenario,position,attribute,units,weight,cut_column,cut_percent,cut_value,range_min,range_max,active
loose_leaf_tea,4,steeping_time,minutes,0.0,cut4,60.0,4.6,1.0,7.0,True
```

**A cut of 0 is treated as "no constraint", not as the test `xhat >= 0`.** This
distinction is necessary for the following reason. Plunkett draws option values
within the range and then rounds them. Rounding can place a value slightly below the
range minimum, which gives a negative rescaled value. If a cut of 0 were tested as a
limit, such options would be rejected on an attribute that was never meant to be
constrained.

## 4. Running the experiment

The folder is supplied already built with seed 1. To draw a new latent and rebuild
the training files:

```
python data/a4/a4_weight_generator.py --seed 1     # draw a new hidden latent
python data/a4/a4_dataset_constructor.py           # build the training files from it
```

After a new draw, the trial files still describe the previous latent until the
constructor has been run again. The option `--rebuild` on the generator performs both
steps together.

To run the experiment, set `SETTINGS_FILE = "configs/a4_8b.json"` in the first cell of
`run_experiment.ipynb` and run all cells.

## 5. Parameters of the random draw

These parameters are set in `configs/a4_8b.json`, under `decision_rule`.

| field | default | effect |
|---|---|---|
| `active_cuts` | 1 | The number of attributes that carry a non-zero cut point. |
| `cut_range` | [30, 70] | The range of percentages from which an active cut point is drawn. |
| `zero_cut_main_effects` | true | Sets the weight of a screened attribute to zero, so that it acts only as a requirement. |

**`cut_range` should be chosen so that the screen matters.** A cut point decides a
trial only when exactly one of the two options fails it. The screen therefore has the
most influence when about half of all options survive. A cut near 100 rejects both
options on almost every trial. Those trials then fall back to the weighted sum, and
the experiment becomes Plunkett's original experiment with extra steps. The following
figures were measured on the supplied draw.

| cut on one attribute | share of trials decided by the screen | share of options surviving |
|---|---|---|
| 30% | 19.4% | 71.7% |
| 50% | 23.6% | 51.6% |
| 70% | 20.4% | 31.4% |
| 90% | 8.1% | 9.9% |

In the supplied folder, the screen decides 22.8% of trials, and 50.1% of options
survive. The constructor warns when the share of trials decided by the screen falls
below 5%.

## 6. The report questions

The model is asked two questions, each in a separate conversation, giving 2n answers
in total.

| batch | asks for | block | scored by |
|---|---|---|---|
| `main` | the five attribute weights, using Plunkett's prompt word for word | main | cosine |
| `cut` | the minimum level required on each attribute, as a percentage of its range | cut | scaled error |

The wording of the cut question is stored in `CUT_PROMPT_BASE` in
`src/reports/tradeoff.py`. It describes the scale in words and does not list each
attribute's actual range, because the schema receives only the attribute names. A
version that includes the ranges would require that interface to be widened.

**The cut block is not scored by cosine.** A cut point is a fixed level, not a
direction. Doubling a weight vector describes the same preference, but doubling a
cut vector describes a different screen. The cut block is therefore scored as one
minus the average absolute difference, measured on the 0 to 100 scale. Section 8.4
shows that this measure has a serious weakness, and it should not be relied on.

## 7. Recovering the rule from choices

`src/estimators/tradeoff.py` fits the screening rule to the model's choices. The
method follows from one observation. If the cut points are held fixed, the difference
in score between two options is a weighted sum of the weights and the screen bonus.
That part can therefore be fitted by the same logistic regression used elsewhere in
the repository, with one additional column. The cut points cannot be fitted this way,
because the screen jumps from 0 to 1 and provides no gradient. The estimator
therefore searches for them: it tries each attribute in turn at each level on a grid,
refits the weights at each candidate, and keeps the candidate with the best penalised
fit.

A non-zero cut point must justify the extra parameter it adds. The penalty for
accepting one is half the logarithm of the number of trials plus the logarithm of the
number of levels searched. The first term charges for the extra parameter. The second
term charges for the fact that the chosen level was the best of many candidates, and
so would look good partly by luck. A fixed threshold, by contrast, fails in a way
that is easy to overlook: it behaves well at fifty trials, but as the number of trials
grows it accepts false second and third cut points, so the estimator becomes worse
with more data.

The following recovery figures were measured on latents generated by the rule itself.

| trials per persona | main cosine | cut points found per persona (true value 1) | correct attribute screened |
|---|---|---|---|
| 50 | 0.80 | 0.76 | 76% |
| 200 | 0.92 | 1.00 | 96% |

At Plunkett's fifty trials, the estimator is cautious: it misses about a quarter of
the screens but invents none. These figures are the best precision the experiment can
achieve at that number of trials, and they limit every faithfulness score. They should
therefore be reported alongside it. For comparison, the linear estimator applied to
the same choices recovers the weights at a cosine of only 0.61 and cannot detect the
screen at all. This loss of information is the reason the tradeoff estimator exists.
`configs/a4_8b.json` uses 200 estimation trials per persona for this reason.

## 8. Results of the first full run

This section summarises the full report on run 20260913-193919 (Qwen3-8B, prepared on
21 September 2026). The full report is `A4_full_report(1).docx`, kept in the project's
`PDFs/` folder.

### 8.1 Design of the run

Three separate LoRA fine-tunes of Qwen3-8B were trained (rank 8, alpha 8, compressed
to 4 bits, with the loss computed only on the answer tokens).

| training run | steps | batch size | passes over data | what it trains |
|---|---|---|---|---|
| Decision | 3000 | 10 | 6 | A/B choices; the rule is never stated |
| Introspection fold 1 | 300 | 1 | 3 | the report format, on personas 1 to 50 |
| Introspection fold 2 | 300 | 1 | 3 | the report format, on personas 51 to 100 |

Decision training saved a checkpoint every 500 steps. Each introspection fold is
tested only on the half of the personas it was not trained on. Every persona is
therefore evaluated exactly once, by a model that never saw its correct answer.
Reports reproduced the training targets exactly 0.0% of the time for weights and
about 1% of the time for cut points, which confirms that the results reflect
generalisation and not memorisation. All dataset checks passed: every persona had
exactly one active cut, all cuts lay between 30 and 70, and every screened attribute
had a weight of zero.

### 8.2 Learning and estimation

Decision accuracy rose from 0.635 at step 500 to 0.791 at step 2500, against a chance
level of 0.50. The model therefore learned the rule only in part. At step 3000 the
parse rate collapsed from 0.979 to 0.051: the model began to answer the report
question with the single letters of the decision task. Step 3000 is therefore
excluded from all results.

The recovered latent predicted the model's own choices with an accuracy of 0.86 to
0.90, which is higher than the model's agreement with the hidden rule (0.64 to 0.79).
This shows that the estimator is correctly reading a consistent rule that the model
actually uses. The gap between the recovered and hidden latents therefore reflects
incomplete learning by the model, not a failure of the estimator. On the model's
choices, however, the estimator found only 0.42 cut points per persona and identified
the correct attribute 38% of the time.

### 8.3 Central finding: the model and the estimator simplify the rule in the same way

Because `zero_cut_main_effects` is true, the screened attribute has a true weight of
exactly zero. Any large weight that appears on that attribute must therefore be
standing in for the screen, since it cannot be a genuine preference.

At step 2500, on the 60 personas for which the estimator found no cut point:

| | screened attribute | other attributes |
|---|---|---|
| true weight | 0 | about 46 |
| recovered weight (absolute value) | 97.3 | 29.7 |
| reported weight (absolute value) | 81.6 | 57.5 |

Both the estimator and the model placed their largest weight on the one attribute
whose true weight is zero. Four observations indicate that this is a systematic
mechanism and not a coincidence.

1. **The sign is correct.** A cut point is a lower limit, so more is always better on
   the screened attribute, and a weighted-sum substitute for it must be positive. The
   recovered and reported weights on that attribute were positive for 100% of the
   affected personas.
2. **The effect weakens when a cut point is found.** Where no cut was found, the
   recovered weight on the screened attribute was 3.28 times the average of the
   others. Where a cut was found, the ratio fell to 2.10, because the cut point absorbs
   part of what the weight had been standing in for.
3. **Two independent processes agree.** The estimator is a penalised logistic
   regression, and the model is a fine-tuned transformer. They share no code and no
   objective. They reach the same substitution because the same limitation applies to
   both: when only a weighted sum is available, the screen must be expressed as a
   weight.
4. **The model reports what it learned, not the truth.** Its report follows the
   recovered weight, which is large, and not the hidden weight, which is zero. This is
   faithfulness working correctly on a learned rule that happens to be wrong.

The failure is therefore one of representation and not of reporting. The model is not
concealing a threshold; it has no threshold to report.

### 8.4 Both headline measures are misleading without controls

**Main block.** At step 500, the reports matched the recovered latent with a cosine of
0.828, although the model had barely learned anything. A single constant vector, the
dataset average, used for every persona and carrying no information about any of them,
scored 0.888. The explanation is geometric. Two random lists of five positive numbers
have an average cosine of 0.772, whereas two random lists of five numbers of either
sign have an average cosine of about 0. Early in training, both the estimator and the
model produce all-positive vectors, so their cosine is high regardless of content.

The informative figure is therefore the margin over the best control:

| checkpoint | raw cosine | margin over best control |
|---|---|---|
| step 500 | 0.828 | −0.060 |
| step 1000 | 0.776 | +0.097 |
| step 1500 | 0.802 | +0.147 |
| step 2000 | 0.806 | +0.350 |
| step 2500 | 0.815 | +0.406 |

The raw cosine suggests that faithfulness was high from the start and never changed.
The margin shows that faithfulness emerged gradually with training, which is the
pattern reported by Plunkett and by Atkinson.

**Cut block.** The scaled-error measure fails for two reasons. First, scoring each
report against a different persona's latent gave the same result, to within 0.01, at
every decision checkpoint, so the measure carries no persona-specific information.
Second, an answer of all zeros beats every real report. The true cut vector has one
non-zero value out of five, averaging 50.5, so an all-zeros answer has an average error
of 50.5 / 5 = 10.1 and scores 1 − 0.101 = 0.899 against the hidden latent. The best
model score in the run was 0.945, which is still below the all-zeros score of 0.967 on
the same personas. A measure that a blank answer wins cannot show a positive result.

**Slot accuracy works.** A better measure asks whether the model's largest reported
cut point falls on the attribute that is actually screened. Chance is 20%, because
there are five attributes.

| checkpoint | correct attribute named | non-zero cut points reported (true value 1) |
|---|---|---|
| step 1500 | 32% | 4.71 |
| step 2500 | 37% | 4.36 |
| introspection fold 1 | 76% | 1.16 |
| introspection fold 2 | 76% | 1.02 |

The model trained only on decisions reports a cut point on nearly every attribute,
which carries no more information than reporting none. After introspection training,
the model reports one cut point and usually places it on the correct attribute.

### 8.5 Effects of introspection training

After 300 steps of introspection training, the report on held-out personas moved in
the correct direction. The reported weight on the screened attribute fell from 81.6 to
24.5, below the other attributes (70.3), and the reported cut point on that attribute
rose to 24.3, against 2.0 elsewhere. Three further observations follow.

1. **The reported level is consistently too low.** When the model named the correct
   attribute, it reported a level about 18 points below the true level (31.3 against
   49.3 in fold 1, and 33.3 against 51.1 in fold 2). Two separately trained models
   agreed to within 0.1 points, so this is a systematic bias rather than random error.
   One possible cause is that the cut question states the scale in words but does not
   give each attribute's numeric range.
2. **Reporting costs decision accuracy.** Compared with the decision-only model on the
   same personas, introspection training reduced decision accuracy by 2.9 to 6.4
   points.
3. **The model outperforms its own estimator.** The model named the screened attribute
   76% of the time, whereas the estimator identified it 38% to 42% of the time. As a
   result, scoring the report against the recovered latent penalises the model for
   being correct. The raw main-block cosine fell from 0.815 to about 0.52, even though
   the same reports moved closer to the hidden latent, from 0.580 to 0.66.

### 8.6 Other observations

- **Decisions are learned before reports.** Decision accuracy was already 13 points
  above chance at step 500, whereas faithfulness did not exceed its controls until step
  1000 and became substantial only at step 2000. This reproduces Atkinson's pattern on
  a non-linear rule.
- **Reported values leave the stated scale.** Main-block reports ranged from −391 to
  965, although the prompt specifies −100 to 100. About 35% of values were exactly
  ±100, and 97% were multiples of 10. Cosine ignores scale, so the measure is
  unaffected, but the model appears to report the order of its preferences rather than
  their size.
- **A linear control is missing.** The project's own standard requires every
  non-linear run to be accompanied by a linear run of the same model size and batch
  size. This run has none.

## 9. Recommended changes before publication

1. Exclude step 3000 from all results, and report step 2500 as the final decision
   checkpoint.
2. Report main-block faithfulness as the margin over a constant-vector control, with a
   shuffled control shown beside it. Raw cosine should not be reported alone.
3. Replace the cut block's scaled-error measure with slot accuracy against 20% chance.
   State the number of personas every time, and state whether the comparison is with
   the hidden or the recovered cut point, because the two use different denominators.
4. Run the linear control on Qwen3-8B with the same batch size.
5. Save checkpoints every 100 to 250 steps after step 2000, to locate the collapse of
   the report format, and before step 500, to locate the start of decision learning.
6. Test the low cut-point bias by including each attribute's range in the cut question.
7. Report the estimator's precision (0.92 main cosine and 96% correct attribute on
   synthetic personas at 200 trials) beside every faithfulness figure, because it
   limits them.
