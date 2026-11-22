# shuffled

The random-answer control for every persona. E-negative's training prompts, with each answer drawn at random: A or B with probability 0.4375 each, C with 0.125 (random.Random(13)), on its own for every trial. That is the answer mix all five personas train on, and the answers follow no rule, so a model trained on this learns none.

A copy of `data/minimum_threshold_one_persona/inverse/persona_e/one_persona_minimum_threshold_e_negative/` with only `instill_1_prefs.jsonl` and `instill_1_prefs_val.jsonl` changed (and their hashes in `manifest.json`, which records the draw under `answers_random`). The prompts, scenarios and `instilled_weights.csv` are E-negative's.

Settings: `configs/minimum_threshold_one_persona/controls/shuffled/shuffled_40gb.json`. Built by `scripts/make_shuffled_control.py`; checked by `tests/test_shuffled_control.py`.
