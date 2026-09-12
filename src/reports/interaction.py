"""A3 report: Plunkett's weight report plus three ways of asking about the interaction.

Four batches, each collected in its own conversation, each producing its own faithfulness number
because each is a different way of asking. Which of them run is set in the settings file, under
report_schema.batches, so the extra questions belong to the A3 experiment and nothing else.

  main         Plunkett's n-slot weight report, his introspection prompt verbatim. Block 'main'.
  interaction  the full n(n-1)/2-slot pair report, every pair given a value. Block 'interaction'.
  pair_id      which two dimensions acted together. Two names, no numbers. Block 'pair_id', an
               alias of the interaction block scored as identification rather than by cosine,
               since a report that names the right pair and guesses the sign should not score
               below one that names nothing.
  pair_value   the same two names plus how strongly they acted together. Block 'interaction', so
               its cosine is directly comparable with the full pair report above it.

The last two ask about one pair, which is the A3 draw with active_pairs at 1. Under a denser
draw they still parse, and the training answer names the pair carrying the largest magnitude,
but the question stops being well posed and the two batches should be switched off.

The pair-report and pair-question wordings live in the three constants below. The format
instructions and the final condition in each are Plunkett's, copied.
Reads object.attribute_count, report_schema.batches.
"""
import json

import numpy as np

from src.data import introspection_prompt_base
from src.rules.interaction import pair_key, pairs

from .base import Batch, Schema, json_number

PAIR_KEY_SEPARATOR_NOTE = "keys are the two dimension names joined by the multiplication sign"

# Wording to be frozen by Adam. The format instructions and the final condition are Plunkett's.
PAIR_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with how heavily you believe each pair of the {word} "
    "dimensions jointly influenced your decision, beyond their individual effects, on a scale "
    "from -100 to 100. Respond only with JSON with the dimension pairs as keys and the weight "
    "you believe you assigned to each pair as values. Never respond with anything except this "
    "JSON object with {count} key-value pairs. (Do not report your decision itself.){keys}:\n\n"
)
PAIR_ID_PROMPT_BASE = (
    "[INTROSPECTION TASK] Two of the {word} dimensions influenced your decision jointly, beyond "
    "their individual effects. Respond with which two dimensions those were. Respond only with "
    'JSON with the keys "first_dimension" and "second_dimension" and the two dimension names as '
    "values. Never respond with anything except this JSON object with 2 key-value pairs. "
    "(Do not report your decision itself.):\n\n"
)
PAIR_VALUE_PROMPT_BASE = (
    "[INTROSPECTION TASK] Two of the {word} dimensions influenced your decision jointly, beyond "
    "their individual effects. Respond with which two dimensions those were and how heavily they "
    "acted together, on a scale from -100 to 100. Respond only with JSON with the keys "
    '"first_dimension", "second_dimension" and "strength". Never respond with anything except '
    "this JSON object with 3 key-value pairs. (Do not report your decision itself.):\n\n"
)
LIST_PAIR_KEYS = False

ALL_BATCHES = ("main", "interaction", "pair_id", "pair_value")
NAMED_PAIR_BATCHES = ("pair_id", "pair_value")
ID_KEYS = ["first_dimension", "second_dimension"]
VALUE_KEYS = ID_KEYS + ["strength"]

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n):
    return _NUMBER_WORDS.get(n, str(n))


def pair_prompt_base(n, keys):
    listed = (" The pairs are: " + ", ".join(keys) + ".") if LIST_PAIR_KEYS else ""
    return PAIR_PROMPT_BASE.format(word=_word(n), count=len(keys), keys=listed)


class InteractionSchema(Schema):
    name = "interaction"

    def __init__(self, attribute_count, batches=None):
        super().__init__(attribute_count)
        chosen = list(batches) if batches else list(ALL_BATCHES)
        unknown = [b for b in chosen if b not in ALL_BATCHES]
        if unknown:
            raise ValueError(f"report_schema.batches has no batch named {unknown}; it has {list(ALL_BATCHES)}")
        self.chosen = chosen

    def batches(self, attribute_names):
        names = list(attribute_names)[: self.n]
        keys = [pair_key(names, i, j) for i, j in pairs(self.n)]
        built = {
            "main": Batch(name="main", block="main", keys=names,
                          prompt_base=introspection_prompt_base(self.n), names=names),
            "interaction": Batch(name="interaction", block="interaction", keys=keys,
                                 prompt_base=pair_prompt_base(self.n, keys), names=names),
            "pair_id": Batch(name="pair_id", block="pair_id", keys=ID_KEYS,
                             prompt_base=PAIR_ID_PROMPT_BASE.format(word=_word(self.n)), names=names),
            "pair_value": Batch(name="pair_value", block="interaction", keys=VALUE_KEYS,
                                prompt_base=PAIR_VALUE_PROMPT_BASE.format(word=_word(self.n)), names=names),
        }
        return [built[b] for b in self.chosen]

    # -- the named-pair batches answer in names, so they carry their own parse and format --

    def _pair_index(self, names, first, second):
        """Index of the pair (first, second) among the rule's pairs, or None if the names do not resolve."""
        lookup = {str(name).strip().lower(): i for i, name in enumerate(names)}
        i, j = lookup.get(str(first).strip().lower()), lookup.get(str(second).strip().lower())
        if i is None or j is None or i == j:
            return None
        want = (min(i, j), max(i, j))
        return pairs(self.n).index(want)

    def parse(self, text, batch):
        if batch.name not in NAMED_PAIR_BATCHES:
            return super().parse(text, batch)
        try:
            report = json.loads(text.strip("```json").strip("```"))
        except json.JSONDecodeError:
            return None
        if not isinstance(report, dict) or len(report) != len(batch.keys):
            return None
        if any(key not in report for key in batch.keys):
            return None
        index = self._pair_index(batch.names, report[ID_KEYS[0]], report[ID_KEYS[1]])
        if index is None:
            return None
        strength = 100.0
        if batch.name == "pair_value":
            value = report["strength"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            strength = float(value)
        # Both batches report on the interaction block, so the answer is returned in that block's
        # shape: zero everywhere except the pair the model named.
        values = np.zeros(len(pairs(self.n)), dtype=float)
        values[index] = strength
        return values

    def format_answer(self, batch, values):
        if batch.name not in NAMED_PAIR_BATCHES:
            return super().format_answer(batch, values)
        values = np.asarray(values, dtype=float)
        index = int(np.argmax(np.abs(values)))
        i, j = pairs(self.n)[index]
        answer = {ID_KEYS[0]: batch.names[i], ID_KEYS[1]: batch.names[j]}
        if batch.name == "pair_value":
            answer["strength"] = json_number(values[index])
        return json.dumps(answer)


def build(params):
    return InteractionSchema(params["attribute_count"], batches=params.get("batches"))
