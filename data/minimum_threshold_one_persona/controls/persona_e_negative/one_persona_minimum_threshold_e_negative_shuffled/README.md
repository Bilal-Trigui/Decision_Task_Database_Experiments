# one_persona_minimum_threshold_e_negative_shuffled

Persona E-negative (0, -100, -40, -70, -25; caffeine >= 17.5 mg): its prompts with the training answers permuted across trials (random.Random(11), train and validation each on its own). No rule to learn: the shuffled-label control for the trial-shown readouts. Settings: `configs/minimum_threshold_one_persona/controls/persona_e_negative/one_persona_e_negative_shuffled_40gb.json`.

A copy of `data/minimum_threshold_one_persona/inverse/persona_e/one_persona_minimum_threshold_e_negative/` with only `instill_1_prefs.jsonl` and `instill_1_prefs_val.jsonl` changed (and their hashes in `manifest.json`, which records the shuffle under `answers_shuffled`). The prompts, the A/B/C shares, `instilled_weights.csv` and the scenarios are the persona's, so the report and estimation trials are those of the trained run 20261116-221120.

Built by `scripts/make_shuffled_controls.py`; checked by `tests/test_shuffled_controls.py`.
