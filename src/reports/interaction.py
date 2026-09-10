"""A3 report: two batches, Plunkett's n-slot main-effect report and an n(n-1)/2-slot pair report.

Batching follows Doc-26: the main-effect batch uses Plunkett's introspection
prompt verbatim, the pair batch asks for each pair of dimensions jointly with
the surrounding format instructions kept word for word. The pair batch's keys
are 'first_name × second_name' in itertools.combinations order, matching the
rule's latent columns. PAIR_PROMPT_BASE is the one place the pair wording
lives; LIST_PAIR_KEYS appends the exact keys to the prompt when true.
Reads object.attribute_count.
"""
from src.data import introspection_prompt_base
from src.rules.interaction import pair_key, pairs

from .base import Batch, Schema

# Wording to be frozen by Adam. The format instructions and the final condition are Plunkett's.
PAIR_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with how heavily you believe each pair of the {word} "
    "dimensions jointly influenced your decision, beyond their individual effects, on a scale "
    "from -100 to 100. Respond only with JSON with the dimension pairs as keys and the weight "
    "you believe you assigned to each pair as values. Never respond with anything except this "
    "JSON object with {count} key-value pairs. (Do not report your decision itself.){keys}:\n\n"
)
LIST_PAIR_KEYS = False

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def pair_prompt_base(n, keys):
    listed = (" The pairs are: " + ", ".join(keys) + ".") if LIST_PAIR_KEYS else ""
    return PAIR_PROMPT_BASE.format(word=_NUMBER_WORDS.get(n, str(n)), count=len(keys), keys=listed)


class InteractionSchema(Schema):
    name = "interaction"

    def batches(self, attribute_names):
        names = list(attribute_names)[: self.n]
        keys = [pair_key(names, i, j) for i, j in pairs(self.n)]
        return [
            Batch(name="main", block="main", keys=names, prompt_base=introspection_prompt_base(self.n)),
            Batch(name="interaction", block="interaction", keys=keys, prompt_base=pair_prompt_base(self.n, keys)),
        ]


def build(params):
    return InteractionSchema(params["attribute_count"])
