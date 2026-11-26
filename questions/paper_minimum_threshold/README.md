# Paper minimum threshold questions

One folder per question. A settings file picks which are asked and which are trained on in
introspection:

```json
"report_schema": {
  "questions": {
    "main":              {"ask": true,  "train": true},
    "screen_index":      {"ask": true,  "train": true},
    "screen_cut_units":  {"ask": true,  "train": false},
    "screen_weight":     {"ask": false, "train": false}
  }
}
```

`ask` without `train` is a held-out probe: measured at every checkpoint, never trained on. `train`
without `ask` is refused. Questions are asked in the order listed.

| folder | kind | what it asks |
|---|---|---|
| main | weights | Plunkett's question: the five weights, -100 to 100 |
| screen_index | screen_index | which attribute, 1 to 5, carried the minimum threshold |
| screen_cut_units | screen_cut_units | that attribute and its minimum in its own units ("35" mg per cup) |
| main_report_units | combined_units | the weights, the attribute and the minimum in units, in one reply |
| screen_cut | screen_cut_fraction | the minimum as a fraction 0 to 1 of the range (the 20261031 run's wording) |
| screen_weight | screen_weight | the screened attribute's own weight; always 0, so it carries no self-knowledge |
| main_report | combined_fraction | the 20261031 run's all-in-one question |
| cut | cut | a cut point per attribute (the minimum threshold schema's report) |
| cut_probe | cut_probe | yes/no, no trial: would an option with attribute 1 at X be rejected? X = 10 to 50 (caffeine mg), one question per level |
| cut_probe_caffeine_full | cut_probe | the same over caffeine's whole range, 0 to 70 mg by 1 (71 questions) |
| cut_probe_steeping | cut_probe | steeping_time (attribute 4), 1 to 7 minutes by 0.25 (screen_moved datasets) |
| cut_probe_steeping_full | cut_probe | steeping_time, 1.0 to 7.0 minutes by 0.1 (61 questions; persona G, cut at 4.0) |
| cut_probe_board_thickness_full | cut_probe | attribute 1 over 1.0 to 5.0 by 0.1 (41 questions): board_thickness in cm for persona H (cutting_board, cut at 3.0) |
| `<name>_order53412` (eight) | as `<name>` | the listing-order control: `<name>` word for word with the attributes listed 5, 3, 4, 1, 2, each under its own number (`"listing_order"` in question.json), for the eight of the nine that list the attributes; asked by ask_only with `configs/ask_only/nine_q_order53412.json` (`scripts/make_listing_order_control.py`) |

A cut probe's attribute name, units and the persona line come from the scenario, so the same folder asks about
whatever that scenario's attribute is; only the attribute's position and the levels are fixed in `question.json`.

Each folder holds:

- `prompt.txt`: the wording, exactly as the model sees it before the A/B scenario. Placeholders:
  `{word}` (the attribute count as a word), `{count}`, `{listing}` ("1 caffeine_content, 2 leaf_size, ..."),
  `{keys}` and `{units_keys}` (key counts of the combined questions). Write a literal brace as `{{` or `}}`.
- `question.json`: the `kind` (how the answer is written and read; see the README's "Questions"), the
  answer `keys` where the kind takes them, and a `description`.

To reword a question, edit its `prompt.txt` (and `keys`, if you rename them, in both files). The data
folder's introspection file for it is rebuilt on the next run, so training always uses the current
wording. To compare two wordings, copy a folder under a new name and select both. Do not edit the
folders a finished run used if you may still resume it: a resume would ask the new wording.
