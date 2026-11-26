# configs/ask_only

Question sets for `ask_only.ipynb` (`QUESTIONS_FROM`), not runs to train. ask_only takes `report_schema` alone from
the file and keeps every other setting of the run being asked.

| file | what it asks | written by |
|---|---|---|
| `nine_q_order53412.json` | The listing-order control: the nine report questions with the attributes listed 5, 3, 4, 1, 2 instead of 1 to 5. Each attribute keeps its own number, so the replies are scored as the nine's are; only the order of the listing changes. `main` lists no attributes and is asked as it is; the other eight are `questions/paper_minimum_threshold/<name>_order53412/`. Prefix "{" and C shown. Asked on A seed 5 (20261123-180745) and A-inverse seed 5 (20261123-180741) as `nine-q-order53412-prefix-c-perkind50`, beside their `nine-q-prefix-c-perkind50`. | `scripts/make_listing_order_control.py` |

The rest of the file is A's seed-5 nine-question settings (40 GB) so that it loads as a complete settings file.
