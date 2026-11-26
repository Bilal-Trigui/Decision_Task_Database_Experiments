# configs/minimum_threshold

The minimum threshold rule over many personas (100), answers A, B or C, on Qwen3-8B: the multi-persona experiment the one-persona work grew out of.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `minimum_threshold_8b` | The minimum threshold rule at Qwen3-8B, 3000 steps, over data/minimum_threshold/. | Qwen3-8B | `minimum_threshold_8b.json` (micro-batch 2, eval batch 16) | 3000 |

Every file is complete (nothing inherited): open it to see or edit any setting.

```
python scripts/run_one_persona.py --settings configs/minimum_threshold/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
