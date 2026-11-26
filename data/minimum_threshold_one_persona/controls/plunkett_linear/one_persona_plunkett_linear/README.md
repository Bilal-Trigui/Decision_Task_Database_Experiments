# data/minimum_threshold_one_persona/controls/plunkett_linear/one_persona_plunkett_linear

The one-persona Plunkett control: the first of Plunkett's hundred personas, alone. loose_leaf_tea, persona line
"Imagine you are Djedefre.", Plunkett's linear rule (the higher weighted sum of range-normalised values wins),
no screen, two options (A or B, no C).

Settings: `configs/minimum_threshold_one_persona/controls/plunkett_linear/one_persona_plunkett_linear.json` (16 GB), `_24gb`, `_40gb`, `_80gb`.

| attribute | caffeine_content | leaf_size | oxidation_level | steeping_time | resteep_potential |
|---|---|---|---|---|---|
| range | 0 to 70 | 0.1 to 10 | 0 to 100 | 1 to 7 | 1 to 8 |
| weight (Plunkett's) | -100 | 95 | 72 | -67 | -1 |
| persona A's, for comparison | 0 | -83 | -38 | -100 | 31 |

Built by `scripts/make_plunkett_one_persona.py`: Plunkett's four input files (`data/plunkett/`) copied as they are,
so row 1 of `instilled_weights.csv` is his weights for this persona and the other 1099 rows are kept for the
cross-persona null. His trial seed. 5000 training trials, the size of his training set (100 personas x 50), 100 validation.
The training prompts are the same prompts, in the same order, as the A-weights control's (`data/minimum_threshold_one_persona/controls/persona_a/one_persona_linear/`): the trials do not depend on the weights.
The answers differ from that control's on 2772 of 5000 trials.

Training answers: 2546 A, 2454 B. Labels checked against an independent recomputation (`tests/test_plunkett_one_persona.py`): 5000/5000 and 100/100.

## Decidability

271 of 5000 training trials (5%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 114 under 2, 546 under 10; median gap 48.1; exact ties 0.
resteep_potential's weight is -1 of 100: it changes almost no answer, so its recovered weight will be noisy and a
reported weight near 0 for it is the right report.

No introspection files (no question is trained on).

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_plunkett_one_persona.py --rebuild
```
