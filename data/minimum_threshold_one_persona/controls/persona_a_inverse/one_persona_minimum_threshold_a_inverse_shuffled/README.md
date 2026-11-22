# one_persona_minimum_threshold_a_inverse_shuffled

Persona A-inverse (0, 83, 38, 100, -31; caffeine >= 35 mg): its prompts with the training answers permuted across trials (random.Random(11), train and validation each on its own). No rule to learn: the shuffled-label control for the trial-shown readouts. Settings: `configs/minimum_threshold_one_persona/controls/persona_a_inverse/one_persona_a_inverse_shuffled_40gb.json`.

A copy of `data/minimum_threshold_one_persona/inverse/persona_a/one_persona_minimum_threshold_a_inverse/` with only `instill_1_prefs.jsonl` and `instill_1_prefs_val.jsonl` changed (and their hashes in `manifest.json`, which records the shuffle under `answers_shuffled`). The prompts, the A/B/C shares, `instilled_weights.csv` and the scenarios are the persona's, so the report and estimation trials are those of the trained run 20261113-170347.

Built by `scripts/make_shuffled_controls.py`; checked by `tests/test_shuffled_controls.py`.
