# data/minimum_threshold_one_persona

One persona (loose_leaf_tea, "Djedefre"), 5000 training and 100 validation trials each, A/B/C answers under a
minimum threshold. Every folder holds the persona's row in `instilled_weights.csv` (the 1099 other rows are kept
for the cross-persona null), the trial files, `constraint_table.csv` and a README. Settings mirror this layout under
`configs/minimum_threshold_one_persona/`.

| folder | weights (caffeine, leaf, oxidation, steeping, resteep) | screen | same prompts as A |
|---|---|---|---|
| `base/one_persona_minimum_threshold` (A) | 0, -83, -38, -100, 31 | caffeine >= 35 mg (50%) | itself |
| `base/one_persona_minimum_threshold_b` (B) | 0, 100, 20, -50, -70 | caffeine >= 35 mg | yes |
| `inverse/one_persona_minimum_threshold_a_inverse` | 0, 83, 38, 100, -31 | caffeine >= 35 mg | yes |
| `inverse/one_persona_minimum_threshold_b_inverse` | 0, -100, -20, 50, 70 | caffeine >= 35 mg | yes |
| `rule_tests/..._screen_moved` | -100, -83, -38, 0, 31 | steeping_time >= 4 min (50%) | no (trials drawn for its own screen) |
| `rule_tests/..._cut25` | 0, -83, -38, -100, 31 | caffeine >= 17.5 mg (25%) | no |
| `rule_tests/..._cut75` | 0, -83, -38, -100, 31 | caffeine >= 52.5 mg (75%) | no |
| `rule_tests/..._screen_moved_cut25` | -100, -83, -38, 0, 31 | steeping_time >= 2.5 min (25%) | no |
| `rule_tests/..._screened_weighted` | 60, -83, -38, -100, 31 | caffeine >= 35 mg | yes (only weight-decided answers differ) |
| `controls/one_persona_linear` | 0, -83, -38, -100, 31 | none (two options, dot product) | no |
| `controls/..._shuffled` | A's (not followed) | A's (not followed) | yes, answers permuted |

Every threshold folder: 3750 trials decided by the weights, 625 by the screen, 625 C (both options fail), in
training; 75 / 13 / 12 in validation. `tests/test_rule_test_datasets.py` recomputes every answer from the prompt's
options and the folder's own rule, with none of the pipeline's code.

Build: `scripts/make_rule_tests.py` (rule_tests and the shuffled control), or
`python data/general/vector_dataset_constructor.py --settings <its config>` for the others. LF line endings, or the
manifest hashes will not match.
