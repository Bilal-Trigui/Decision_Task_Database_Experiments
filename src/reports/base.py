"""

Reports:
A report schema decides how you ask the model what its weights were. This file defines what a schema is. 
While rules (base.py and build.py) is how a persona picks between A and B.

Two pieces:
Batch. One question is asked in its own fresh conversation. It holds a name, a block, keys, prompt_base.

There is also the Schema class.
Thisis the base class every schema inherits. A subclass only has to write batches(). Everything else is shared.
This holds slot_count, prompt, parse and format answer. 

The three subclasses (interaction, linear, minimum_threshold) only differ in their batches. Linear.py has one batch, minimum_threshold.py has two batches, and interaction.py has up to 4 batches. 

Blocks connects both reports/base.py and rules/base.py and rules/build.py

"""
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    name: str
    block: str
    keys: list
    prompt_base: str
    names: list = None      # the scenario's attribute names, for a batch answered in names


def json_number(value):
    """Integers stay integers in the JSON answer, as in Plunkett's training targets; nothing is rounded."""
    value = float(value)
    return int(value) if value.is_integer() else value


class Schema:
    """
    The schema is the questionaire. This decides what the true answer is. 
    It decides how the question is worded, and what a valid reply looks like, and how a reply becomes numbers that can be compared with the truth.

    How a schema is used:
    1. created from the setitngs. config.resolve (config.py) reads report_schema.type. For example, "interaction." it calls src/reports/interaction.py:build(), which returns
    an interaction schema. 

    2. Then the model is asked. Happens in ealuation.py 
    
    3. Then it is scored. Each batch's block says which slice of the true latent is to be compared against. rule.block_slices turns that name into indices, and every batch gets
    its own faithfulness row.

    4. Introspection training (run in the other direction). In data.py, for each persona and batch, the code looks up the true values for batch.block from the latent. Schema.format_answer
    then writes them as the JSON answer. Them odel is trained on (prompt -> correct JSON). This is also where a mismatch fails: if a schema asks for a block the rule doesn't define, data
    format error is raised. 

    """


    name = "base"

    def __init__(self, attribute_count):
        self.n = attribute_count

    def batches(self, attribute_names):
        """The batches for one scenario's attribute names, in collection order."""
        raise NotImplementedError

    def slot_count(self):
        """Total number of reported numbers. Reads object.attribute_count."""
        return sum(len(b.keys) for b in self.batches([f"attribute_{i}" for i in range(1, self.n + 1)]))

    def prompt(self, attribute_names, batch_index=0):
        """The prompt text before the trial for one batch."""
        return self.batches(attribute_names)[batch_index].prompt_base

    def parse(self, text, batch):
        """Plunkett's save_reported_weights rule: fenced JSON allowed, must be a dict with exactly the batch's keys."""
        r_string = text.strip("```json").strip("```")
        try:
            report = json.loads(r_string)
        except json.JSONDecodeError:
            return None
        if not isinstance(report, dict) or len(report) != len(batch.keys):
            return None
        values = []
        for key in batch.keys:
            if key not in report:
                return None
            value = report[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            values.append(float(value))
        return np.array(values, dtype=float)

    def format_answer(self, batch, values):
        """The JSON answer for an introspection-training example, in Plunkett's json.dumps format."""
        return json.dumps({key: json_number(v) for key, v in zip(batch.keys, values)})
