# Paper maximum threshold questions

The paper minimum threshold questions (`questions/paper_minimum_threshold/`), reworded for a maximum:
an option that went above the threshold was rejected. Same folders, kinds and placeholders, so that
README covers how the folders work. Answer keys are `maximum_value` and `maximum_level`. `main` asks
about the weights only, so its wording is unchanged.

| folder | kind | what it asks |
|---|---|---|
| main | weights | Plunkett's question: the five weights, -100 to 100 |
| screen_index | screen_index | which attribute, 1 to 5, carried the maximum threshold |
| screen_cut_units | screen_cut_units | that attribute and its maximum in its own units ("35" mg per cup) |
| main_report_units | combined_units | the weights, the attribute and the maximum in units, in one reply |
| screen_cut | screen_cut_fraction | the maximum as a fraction 0 to 1 of the range |
| screen_weight | screen_weight | the screened attribute's own weight; always 0, so it carries no self-knowledge |
| main_report | combined_fraction | the all-in-one question, with the fraction wording |
| cut | cut | a maximum per attribute; 0 means no maximum |
