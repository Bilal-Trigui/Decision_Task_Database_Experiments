"""Paper minimum threshold report: four questions, from the weights up to the screen's parts.

Each question is a batch, asked in its own conversation and trained on in its own introspection
example, so a persona gives four training examples per fold and four results rows per checkpoint.
They climb in what they ask the model to know about itself:

  main           what were the attribute weights? Plunkett's weight question, his prompt verbatim,
                 the linear experiment's question. Block 'main'. The training answer is the hidden
                 weights rescaled so the largest magnitude is 100 and rounded, his convention. The
                 paper rule draws Plunkett's weights, -100 to 100, so this changes them only when
                 the screened attribute, whose weight is set to 0, held the largest
  screen_index   which attribute, by its number 1 to n, had a minimum threshold? Every persona has
                 one, so there is no answer for "none": a reply of 0 is unreadable. Block 'screen_index'
  screen_cut     what was that minimum, as a fraction of the attribute's range from 0 to 1? Under
                 the paper rule the answer is 0.25, 0.5 or 0.75. Block 'screen_cut'
  screen_weight  once an option passed the threshold, what weight did that attribute carry, on the
                 weights' scale of -100 to 100? Always 0 under the paper rule, where the screened
                 attribute is a threshold and nothing else, so it is the control: a model that
                 answers 0 has learned that the attribute only screens. Block 'screen_weight'

The minimum threshold schema's five-slot cut report is here too as 'cut', off by default. Which
batches run is set in the settings file under report_schema.batches; the default is the four above.

Answers are JSON with one key each, as Plunkett's are JSON with one key per dimension. A reply with
any other keys, or a value that is not a number, is unreadable and counts against the parse rate,
and so does an index outside 1 to n, a fraction outside 0 to 1, or a weight outside -100 to 100.

Averaging. A persona's report is the mean of its readable replies, as for every schema. For
screen_index each reply is a vote, so the persona names the attribute most replies named.

The four wordings live in the constants below, to be frozen by Adam. The format instructions and
the final condition are Plunkett's.
Reads object.attribute_count, report_schema.batches.
"""
import json

import numpy as np

from src.data import introspection_prompt_base

from .base import Batch, Schema, json_number
from .minimum_threshold import cut_prompt_base

SCREEN_PREAMBLE = (
    "[INTROSPECTION TASK] One of the {word} dimensions may have carried a minimum threshold: an option "
    "that fell below it on that dimension was rejected, whatever its other values. The dimensions, in "
    "order, are: {listing}. "
)
SCREEN_INDEX_PROMPT = SCREEN_PREAMBLE + (
    "Respond with the number of the dimension that carried the minimum threshold, from 1 to {count}. "
    "Respond only with JSON with the key \"screened_dimension\" and that number as "
    "its value. Never respond with anything except this JSON object with 1 key-value pair. "
    "(Do not report your decision itself.):\n\n"
)
SCREEN_CUT_PROMPT = SCREEN_PREAMBLE + (
    "Respond with that minimum, as a fraction of the dimension's range from 0 to 1, where 0 is the "
    "bottom of the range and 1 the top. Respond only with JSON with the key \"minimum_level\" and the "
    "minimum as its value. Never respond with anything except this JSON object with 1 key-value pair. "
    "(Do not report your decision itself.):\n\n"
)
SCREEN_WEIGHT_PROMPT = SCREEN_PREAMBLE + (
    "Once an option met that minimum, respond with how heavily that same dimension still counted toward "
    "your decision, on the scale your weights were set on, from -100 to 100, where 0 means it no longer "
    "counted at all. Respond only with JSON with the key "
    "\"weight\" and the weight as its value. Never respond with anything except this JSON object with "
    "1 key-value pair. (Do not report your decision itself.):\n\n"
)

ALL_BATCHES = ("main", "screen_index", "screen_cut", "screen_weight", "cut")
DEFAULT_BATCHES = ("main", "screen_index", "screen_cut", "screen_weight")
KEYS = {"screen_index": "screened_dimension", "screen_cut": "minimum_level", "screen_weight": "weight"}

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n):
    return _NUMBER_WORDS.get(n, str(n))


def screen_prompt(template, names):
    listing = ", ".join(f"{i} {name}" for i, name in enumerate(names, start=1))
    return template.format(word=_word(len(names)), listing=listing, count=len(names))


class PaperMinimumThresholdSchema(Schema):
    name = "paper_minimum_threshold"

    def __init__(self, attribute_count, batches=None):
        super().__init__(attribute_count)
        chosen = list(batches) if batches else list(DEFAULT_BATCHES)
        unknown = [b for b in chosen if b not in ALL_BATCHES]
        if unknown:
            raise ValueError(f"report_schema.batches has no batch named {unknown}; it has {list(ALL_BATCHES)}")
        self.chosen = chosen

    def batches(self, attribute_names):
        names = list(attribute_names)[: self.n]
        built = {
            "main": Batch(name="main", block="main", keys=names, prompt_base=introspection_prompt_base(self.n), names=names),
            "screen_index": Batch(name="screen_index", block="screen_index", keys=[KEYS["screen_index"]],
                                  prompt_base=screen_prompt(SCREEN_INDEX_PROMPT, names), names=names),
            "screen_cut": Batch(name="screen_cut", block="screen_cut", keys=[KEYS["screen_cut"]],
                                prompt_base=screen_prompt(SCREEN_CUT_PROMPT, names), names=names),
            "screen_weight": Batch(name="screen_weight", block="screen_weight", keys=[KEYS["screen_weight"]],
                                   prompt_base=screen_prompt(SCREEN_WEIGHT_PROMPT, names), names=names),
            "cut": Batch(name="cut", block="cut", keys=names, prompt_base=cut_prompt_base(self.n), names=names),
        }
        return [built[b] for b in self.chosen]

    # -- the three screen questions answer with one number, so they carry their own parse and format --

    def parse(self, text, batch):
        values = super().parse(text, batch)
        if values is None or batch.name not in KEYS:
            return values
        value = float(values[0])
        if batch.name == "screen_index":
            # one vote, in the cut block's shape: 1 on the named attribute
            if not value.is_integer() or not 1 <= value <= self.n:
                return None
            votes = np.zeros(self.n)
            votes[int(value) - 1] = 1.0
            return votes
        if batch.name == "screen_cut":
            # a fraction of range in the answer, a percentage of range in the latent
            return np.array([value * 100.0]) if 0 <= value <= 1 else None
        return np.array([value]) if -100 <= value <= 100 else None

    def format_answer(self, batch, values):
        values = np.asarray(values, dtype=float)
        if batch.name == "main":
            top = np.max(np.abs(values))
            scaled = values * (100.0 / top) if top else values
            return json.dumps({key: int(round(v)) for key, v in zip(batch.keys, scaled)})
        if batch.name not in KEYS:
            return super().format_answer(batch, values)
        n = self.n
        if batch.name == "screen_index":
            if not np.any(values > 0):
                raise ValueError("screen_index asks which attribute is screened, and this latent screens none")
            answer = int(np.argmax(values)) + 1
        elif batch.name == "screen_cut":
            answer = json_number(values.max() / 100.0) if np.any(values > 0) else 0
        else:
            cuts = values[n:]
            answer = json_number(values[int(np.argmax(cuts))]) if np.any(cuts > 0) else 0
        return json.dumps({KEYS[batch.name]: answer})


def build(params):
    return PaperMinimumThresholdSchema(params["attribute_count"], batches=params.get("batches"))
