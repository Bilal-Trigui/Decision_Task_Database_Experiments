# configs/minimum_threshold_one_persona

Settings for every one-persona minimum-threshold experiment (loose_leaf_tea, persona "Djedefre", Qwen3-8B,
3000 steps, a checkpoint every 250). Data folders mirror this layout under `data/minimum_threshold_one_persona/`.
Run any of them with

```
python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/<group>/<file>.json
```

`extends` is resolved relative to each file's own folder.

## base/: personas A and B

| file | persona | weights (caffeine, leaf_size, oxidation, steeping, resteep) | screen | report setup | finished run |
|---|---|---|---|---|---|
| `one_persona_minimum_threshold.json` | A | 0, -83, -38, -100, 31 | caffeine >= 35 mg (50%) | original 4 questions | 20261106-152354 (A-old) |
| `one_persona_minimum_threshold_b.json` | B | 0, 100, 20, -50, -70 | caffeine >= 35 mg | original 4 questions | 20261109-214726 (B-old) |
| `one_persona_no_trial_a.json` | A | as A | as A | + no-trial questions | 20261110-211307 (A-new) |
| `one_persona_no_trial_b.json` | B | as B | as B | + no-trial questions | 20261110-112915 (B-new) |
| `one_persona_probes_a.json` | A | as A | as A | every question, `"{"` prefix, C shown | not run |
| `one_persona_probes_b.json` | B | as B | as B | every question, `"{"` prefix, C shown | not run |

## inverse/: every weight's sign flipped, the screen kept

| file | weights | data |
|---|---|---|
| `one_persona_a_inverse.json` | 0, 83, 38, 100, -31 | A's 5000 prompts; the 3750 weight-decided answers flipped |
| `one_persona_b_inverse.json` | 0, -100, -20, 50, 70 | the same prompts; B's weight-decided answers flipped |

## rule_tests/: persona A with one part of the rule changed (`scripts/make_rule_tests.py`)

| file | changed | tests |
|---|---|---|
| `..._screen_moved.json` | screen on steeping_time (4th) at 50% (4 min); caffeine and steeping weights swapped | is "dimension 1" a listing-order default? |
| `..._cut25.json` | caffeine cut at 25% (17.5 mg) | does the stated cut move with the true cut? |
| `..._cut75.json` | caffeine cut at 75% (52.5 mg) | the same, above the middle |
| `..._screen_moved_cut25.json` | steeping_time screened at 25% (2.5 min) | both defaults broken at once |
| `..._screened_weighted.json` | caffeine screened at 50% and weighted +60 | can a report separate screened from weighted? |

All extend `base/one_persona_probes_a.json` (every report question, `"{"` prefix, C in report trials). The
screen-moved ones also ask `cut_probe_steeping` (1 to 7 minutes); cut25 and cut75 also ask `cut_probe_caffeine_full`
(0 to 70 mg). For screened_weighted, the four-weight cosine (`cos_recovered_4`) leaves caffeine out although its
weight is 60: read the five-value cosine as well.

## controls/

| file | what |
|---|---|
| `one_persona_linear.json` | A's weights, no screen, two options: the plain dot-product rule |
| `one_persona_minimum_threshold_shuffled.json` | A's prompts with the answers permuted (seed 11): no rule to learn |

Other experiments (cut sweep, maximum threshold, the multi-persona paper configs) stay in `configs/`.
Finished runs recorded the old flat paths (`configs/one_persona_*.json`, `data/one_persona_*`); `src/data_paths.py`
maps the old data paths to these, so they still read their data.
