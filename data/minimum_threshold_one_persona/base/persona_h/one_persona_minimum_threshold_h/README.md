# one_persona_minimum_threshold_h

Persona H: B's weights in a new domain: cutting_board, Antonín Dvořák, board_thickness screened at 25% (2.0 cm); surface_area 100, knife_mark_resistance 20, grip_stability -50, ease_of_cleaning -70.

Settings: `configs/minimum_threshold_one_persona/base/persona_h/one_persona_h.json` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.

Scenario `cutting_board`, persona line "Imagine you are Antonín Dvořák."

| attribute | board_thickness | surface_area | knife_mark_resistance | grip_stability | ease_of_cleaning |
|---|---|---|---|---|---|
| units, range | centimeters, 1 to 5 | square centimeters, 300 to 1200 | resistance rating, 1 to 10 | stability rating, 1 to 10 | cleaning ease rating, 1 to 10 |
| B's weight | 0 | 100 | 20 | -50 | -70 |
| this weight | 0 | 100 | 20 | -50 | -70 |
| cut | 25% |  |  |  |  |

Screen: board_thickness (attribute 1) at 25% of its range (2 centimeters, of 1 to 5).

Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of `instilled_weights.csv` changed and the `cutting_board` row moved to the top of `scenarios.csv` and `instilled_weights.csv` (swapped with loose_leaf_tea, which keeps A's weights), with A's seeds. The trials are drawn for this screen/scenario, so the prompts are not A's.

Training: 3750 trials decided by the weights, 625 by the screen, 625 C. Validation: 75 / 13 / 12.
Labels checked against an independent recomputation (`tests/test_new_personas.py`): 5000/5000 and 100/100.

## Decidability

243 of 3750 weight-decided training trials (6%) have a utility gap under 5 on the normalised scale (utility = sum of weight x value scaled to 0..1); 92 under 2, 503 under 10; median gap 38.0; exact ties (the rule says B; float rounding can make it A) 7.

These gaps are in the units the weights are written in, so they compare personas only after each is put on one scale: the answers do not change when every weight is multiplied by the same number, and no answer here is noisy, so a small gap is not a hard trial in itself.

## Prediction, recorded before running

As for B, whose cut is at 50% where this one is at 25%: faithful weights, the largest (surface_area, +100) reported positive, the three reply forms, and the cut probe (cut_probe_board_thickness_full, 1.0 to 5.0 cm in steps of 0.1) switching at 2.0. If H behaves like B, the tea results generalise; scenario, persona, attribute names, units and ranges all change at once, so this is a full replication of B and nothing else.
Every report question names the persona "Imagine you are Antonín Dvořák."

Readout after a run: `python tools/persona_readouts.py results/<run_id>`.

Constructor stats: `{"screen_decisive_share": 0.18588235294117647, "screen_decisive_share_median_persona": 0.18588235294117647, "screen_decisive_share_min_persona": 0.18588235294117647, "option_survival_rate": 0.8125490196078431, "weight_decided_share": 0.75, "answer_shares": {"A": 0.4327450980392157, "B": 0.4423529411764706, "C": 0.12490196078431373}}`

Rebuild (LF line endings, or the manifest hashes will not match):

```
python scripts/make_new_personas.py --rebuild h
```
