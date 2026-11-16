# configs/one_persona_maximum_threshold

One persona (A's weights) under a maximum threshold instead of a minimum: an option with MORE than 35 mg of caffeine is rejected.

| experiment | what it is for | model | files (card) | steps |
|---|---|---|---|---|
| `one_persona_maximum_threshold` | One persona, trained from scratch on a maximum threshold: the inverse of one_persona_minimum_threshold.json. | Qwen3-8B | `one_persona_maximum_threshold.json` (16 GB) | 2000 |

Every file is complete (nothing inherited): open it to see or edit any setting.

```
python scripts/run_one_persona.py --settings configs/one_persona_maximum_threshold/<file>.json
```

Written by `scripts/write_config_readmes.py`; rerun it after adding a file.
