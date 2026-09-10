"""Base class for report schemas: the exact format and wording a report is collected under.

A schema splits its slots into one or more batches, each collected in its own
prompt (Plunkett's five-slot report is one batch; A3 uses two). A batch names
the block of the latent it reports on, the JSON keys in order, and the prompt
text placed before the trial. `parse` applies Plunkett's acceptance rule and
returns the values in key order or None.

Reads: object.attribute_count (through `attribute_count`); report_schema
(type only; temperature and sample counts are used by evaluate.py).
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


def json_number(value):
    """Integers stay integers in the JSON answer, as in Plunkett's training targets; nothing is rounded."""
    value = float(value)
    return int(value) if value.is_integer() else value


class Schema:
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
