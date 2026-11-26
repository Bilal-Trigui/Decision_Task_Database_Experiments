# data/minimum_threshold_one_persona

One persona (loose_leaf_tea, "Djedefre"), A/B/C answers under a minimum threshold. Every folder holds the
persona's row in `instilled_weights.csv` (the 1099 other rows are kept for the cross-persona null), the trial
files, `constraint_table.csv` and a README. The layout is the same as the settings under
`configs/minimum_threshold_one_persona/`: a group, then the persona it is built on.

```
base/        persona_a/one_persona_minimum_threshold      persona_b/one_persona_minimum_threshold_b
             persona_c/..._c   persona_d/..._d   persona_e/..._e_positive   persona_f/..._f_oxidation, ..._f_resteep
             persona_g/..._g   persona_h/..._h
inverse/     persona_a/..._a_inverse                      persona_b/..._b_inverse
             persona_c/..._c_inverse   persona_d/..._d_inverse   persona_e/..._e_negative
             persona_f/..._f_oxidation_inverse, ..._f_resteep_inverse   persona_g/..._g_inverse   persona_h/..._h_inverse
rule_tests/  persona_a/..._screen_moved, _cut25, _cut75, _screen_moved_cut25, _screened_weighted
             persona_a_inverse/..._a_inverse_cut75
controls/    persona_a/one_persona_linear, ..._shuffled
cut_sweep/   persona_a/w00_c100 ... w75_c25      persona_b/...   persona_a_inverse/...   persona_b_inverse/...   persona_c/...   persona_c_inverse/...   persona_e_positive/...   persona_e_negative/...
```

The Qwen3-14B and Qwen3-32B settings (`configs/minimum_threshold_one_persona_qwen3_14b/`, `..._qwen3_32b/`) use
these same folders: the trials and their answers do not depend on the model.

## base, inverse, rule_tests, controls

5000 training and 100 validation trials each.

| folder | weights (caffeine, leaf, oxidation, steeping, resteep) | screen | same prompts as A |
|---|---|---|---|
| `base/persona_a/one_persona_minimum_threshold` (A) | 0, -83, -38, -100, 31 | caffeine >= 35 mg (50%) | itself |
| `base/persona_b/one_persona_minimum_threshold_b` (B) | 0, 100, 20, -50, -70 | caffeine >= 35 mg | yes |
| `inverse/persona_a/one_persona_minimum_threshold_a_inverse` | 0, 83, 38, 100, -31 | caffeine >= 35 mg | yes |
| `inverse/persona_b/one_persona_minimum_threshold_b_inverse` | 0, -100, -20, 50, 70 | caffeine >= 35 mg | yes |
| `rule_tests/persona_a/..._screen_moved` | -100, -83, -38, 0, 31 | steeping_time >= 4 min (50%) | no (trials drawn for its own screen) |
| `rule_tests/persona_a/..._cut25` | 0, -83, -38, -100, 31 | caffeine >= 17.5 mg (25%) | no |
| `rule_tests/persona_a/..._cut75` | 0, -83, -38, -100, 31 | caffeine >= 52.5 mg (75%) | no |
| `rule_tests/persona_a_inverse/..._a_inverse_cut75` | 0, 83, 38, 100, -31 | caffeine >= 52.5 mg (75%) | no: cut75's prompts (only weight-decided answers differ) |
| `rule_tests/persona_a/..._screen_moved_cut25` | -100, -83, -38, 0, 31 | steeping_time >= 2.5 min (25%) | no |
| `rule_tests/persona_a/..._screened_weighted` | 60, -83, -38, -100, 31 | caffeine >= 35 mg | yes (only weight-decided answers differ) |
| `controls/persona_a/one_persona_linear` | 0, -83, -38, -100, 31 | none (two options, dot product) | no |
| `controls/persona_a/..._shuffled` | A's (not followed) | A's (not followed) | yes, answers permuted |
| `base/persona_c/..._c` | 0, 100, -30, -100, 30 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `inverse/persona_c/..._c_inverse` | 0, -100, 30, 100, -30 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `base/persona_d/..._d` | 0, 15, -10, 20, -15 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `inverse/persona_d/..._d_inverse` | 0, -15, 10, -20, 15 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `base/persona_e/..._e_positive` | 0, 100, 40, 70, 25 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `inverse/persona_e/..._e_negative` | 0, -100, -40, -70, -25 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `base/persona_f/..._f_oxidation` | 0, 30, -100, 45, -20 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `base/persona_f/..._f_resteep` | 0, -25, 40, -30, 100 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `base/persona_g/..._g` | -83, -38, -100, 0, 31 | steeping_time >= 2.5 min (25%) | no (trials drawn for its own screen) |
| `base/persona_h/..._h` | cutting_board: 0, 100, 20, -50, -70 | board_thickness >= 2 cm (25%) | no: cutting_board, "Imagine you are Antonín Dvořák" |
| `inverse/persona_f/..._f_oxidation_inverse` | 0, -30, 100, -45, 20 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `inverse/persona_f/..._f_resteep_inverse` | 0, 25, -40, 30, -100 | caffeine >= 17.5 mg (25%) | no: cut25's prompts |
| `inverse/persona_g/..._g_inverse` | 83, 38, 100, 0, -31 | steeping_time >= 2.5 min (25%) | no: G's prompts |
| `inverse/persona_h/..._h_inverse` | cutting_board: 0, -100, -20, 50, 70 | board_thickness >= 2 cm (25%) | no: H's prompts |

Every threshold folder: 3750 trials decided by the weights, 625 by the screen, 625 C (both options fail), in
training; 75 / 13 / 12 in validation. `tests/test_rule_test_datasets.py` and `tests/test_new_personas.py` (C to H)
recompute every answer from the prompt's options and the folder's own rule, with none of the pipeline's code. H rates
three attributes in whole numbers 1 to 10, so 7 of its 3750 weight-decided trials are exact ties of the weighted sums
(the rule gives B; float rounding gave one of them A); every other answer matches. H-inverse has the same 7 ties,
and F-resteep-inverse has one.

## cut_sweep (personas A, B, A-inverse, B-inverse, C, C-inverse, E-positive, E-negative)

One folder per persona (`persona_a`, `persona_b`, `persona_a_inverse`, `persona_b_inverse`), each built from that
persona's own data folder. The table below is A's; the other three have the same nine mixes under their own weights.
Persona A has two sets: `persona_a/wXX_cYY` is the first sweep's (from 20261106-152354, run 2026-11-07/08), and
`persona_a/screen_as_drawn/wXX_cYY` the matched one's (from 20261110-211307). B, A-inverse and B-inverse hold the
matched set only, as do `persona_c`, `persona_c_inverse`, `persona_e_positive` and `persona_e_negative` (built the
same way, under those personas' weights and their screen at 17.5 mg, 25% of the caffeine range).

Persona A's weights and screen, 5000 training trials, with the mix of trial kinds swept: `wXX_cYY` has XX% decided
by the weights alone and YY% by the cut, half of those C (both options below the cut) and half where one option is
below it. In the first A sweep's folders those are only trials where the cut changes the answer from what the weights
alone would pick (`cut_overturns_only`), so each is answered against the weights; in the matched four's folders they
are taken as drawn, the weights alone right on about half (`configs/minimum_threshold_one_persona/cut_sweep/README.md`).
The first sweep's runs branch from A's run 20261106-152354 at decision step 1500. `tests/test_cut_sweep.py` checks every folder holds
exactly those shares.

| folder | weights decide | C | one option fails |
|---|---|---|---|
| `w75_c25` | 75% | 12.5% | 12.5% |
| `w65_c35` | 65% | 17.5% | 17.5% |
| `w55_c45` | 55% | 22.5% | 22.5% |
| `w45_c55` | 45% | 27.5% | 27.5% |
| `w35_c65` | 35% | 32.5% | 32.5% |
| `w25_c75` | 25% | 37.5% | 37.5% |
| `w15_c85` | 15% | 42.5% | 42.5% |
| `w05_c95` | 5% | 47.5% | 47.5% |
| `w00_c100` | 0% | 50% | 50% |

## Building

`scripts/make_rule_tests.py` (rule_tests and the shuffled control), `scripts/make_cut75_seed_runs.py` (A-inverse at the 75% cut),
`scripts/make_new_personas.py` (personas C to H),
`scripts/make_cut_sweep.py` (cut_sweep), or
`python data/general/vector_dataset_constructor.py --settings <its config>` for the others. LF line endings, or the
manifest hashes will not match.

Finished runs recorded the folders' older places (`data/one_persona_*`, `data/minimum_threshold_one_persona/<group>/<folder>`,
`data/cut_sweep/<wXX_cYY>`); `src/data_paths.py` maps each to its place here, so they still read their data.
