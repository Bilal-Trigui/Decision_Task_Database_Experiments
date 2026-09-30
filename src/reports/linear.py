"""Plunkett's report: one JSON object with the attribute names as keys, weights from -100 to 100.

One batch, 'main', over the 'main' block, with Plunkett's introspection prompt
verbatim. Reads object.attribute_count.
"""
from src.data import introspection_prompt_base

from .base import Batch, Schema


class LinearSchema(Schema):
    name = "linear"

    def batches(self, attribute_names):
        names = list(attribute_names)[: self.n]
        return [Batch(name="main", block="main", keys=names, prompt_base=introspection_prompt_base(self.n))]


def build(params):
    return LinearSchema(params["attribute_count"])
