# configs/paper_minimum_threshold

The paper's minimum threshold design over 100 personas: the 8B run, its 16 GB versions (with revised questions, a resume and an extension), a 0.6B version and its smoke tests, and the 2500-step run with twice the data.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `minimum_threshold_1500_steps` | The paper minimum threshold task on Qwen3-8B (16 GB card), 2500 decision steps over twice the training trials, with introspection at steps 1500 and 2500 toward two targets: run 1's schedule, with twice the data and 60% of it decided by the weights. | Qwen3-8B | `minimum_threshold_1500_steps.json` (16 GB) | 2500 |
| `paper_minimum_threshold_0.6b` | The paper's minimum threshold experiment at Qwen3-0.6B, every setting written out so any of them can be changed here. | Qwen3-0.6B | `paper_minimum_threshold_0.6b.json` (micro-batch 10, eval batch 16) | 1500 |
| `paper_minimum_threshold_0.6b_basics` | The paper's minimum threshold experiment at Qwen3-0.6B, every setting written out so any of them can be changed here. | Qwen3-0.6B | `paper_minimum_threshold_0.6b_basics.json` (micro-batch 10, eval batch 4) | 1500 |
| `paper_minimum_threshold_0.6b_basics_updated` | basics_updated: a barebones smoke test of the whole pipeline, built to finish in about 5 minutes on a 16 GB RTX 5070 Ti. | Qwen3-0.6B | `paper_minimum_threshold_0.6b_basics_updated.json` (micro-batch 2, eval batch 8) | 20 |
| `paper_minimum_threshold_8b` | The paper's minimum threshold design at Qwen3-8B, over data/paper_minimum_threshold/: Plunkett's weights, -100 to 100 with the largest magnitude 100 and none nearer 0 than 20, the screened attribute's weight 0 so it is a pure screen, one cut at 25, 50 or 75 percent of its range. | Qwen3-8B | `paper_minimum_threshold_8b.json` (micro-batch 2, eval batch 16) | 3000 |
| `paper_minimum_threshold_8b_16gb` | The paper's minimum threshold experiment on Qwen3-8B, a full run sized for one 16 GB GPU (an RTX 5070 Ti), every setting written out so any of them can be changed here. | Qwen3-8B | `paper_minimum_threshold_8b_16gb.json` (micro-batch 2, eval batch 4) | 2500 |
| `paper_minimum_threshold_8b_16gb_extend` | paper_minimum_threshold_8b_16gb_resume.json, extended: decision training continues past 2500 to 4000. | Qwen3-8B | `paper_minimum_threshold_8b_16gb_extend.json` (16 GB) | 4000 |
| `paper_minimum_threshold_8b_16gb_resume` | paper_minimum_threshold_8b_16gb.json, for resuming a run of it on the same 16 GB card. | Qwen3-8B | `paper_minimum_threshold_8b_16gb_resume.json` (16 GB) | 2500 |
| `paper_minimum_threshold_8b_16gb_units` | paper_minimum_threshold_8b_16gb.json with the report questions revised after the 20261031-061400 run. | Qwen3-8B | `paper_minimum_threshold_8b_16gb_units.json` (16 GB) | 2500 |

Every file is complete (nothing inherited): open it to see or edit any setting.

```
python scripts/run_one_persona.py --settings configs/paper_minimum_threshold/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
