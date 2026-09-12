"""A4 report: two batches of n slots each, Plunkett's weight report and a cut-point report.

The main batch uses Plunkett's introspection prompt verbatim. The cut batch asks for the minimum
level the model required of each dimension before it would accept an option at all, on the same
0 to 100 percentage-of-range scale the rule holds cut points in, with the surrounding format
instructions kept word for word from Plunkett. Both batches key on the attribute names, so the
report is 2n numbers collected in two conversations. CUT_PROMPT_BASE is the one place the cut
wording lives.

The paper's scaffolding dial for this schema would be stating each attribute's range in the
prompt, so that a percentage has an explicit referent. That is not offered here because the
schema interface is handed attribute names only, never ranges, and widening it would change
every schema. The prompt names the scale instead.

Reads object.attribute_count.
"""
from src.data import introspection_prompt_base

from .base import Batch, Schema

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

# Wording to be frozen by Adam. The format instructions and the final condition are Plunkett's.
CUT_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with the minimum level you believe you required of each of the "
    "{word} dimensions before you would accept an option at all, as a percentage of that "
    "dimension's range from 0 to 100, where 0 means you required no minimum. Respond only with "
    "JSON with the dimension names as keys and the minimum you believe you required of each of "
    "them as values. Never respond with anything except this JSON object with {count} key-value "
    "pairs. (Do not report your decision itself.):\n\n"
)


def cut_prompt_base(n):
    return CUT_PROMPT_BASE.format(word=_NUMBER_WORDS.get(n, str(n)), count=n)


class TradeoffSchema(Schema):
    name = "tradeoff"

    def batches(self, attribute_names):
        names = list(attribute_names)[: self.n]
        return [
            Batch(name="main", block="main", keys=names, prompt_base=introspection_prompt_base(self.n)),
            Batch(name="cut", block="cut", keys=names, prompt_base=cut_prompt_base(self.n)),
        ]


def build(params):
    return TradeoffSchema(params["attribute_count"])
