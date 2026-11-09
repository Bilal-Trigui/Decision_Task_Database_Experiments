"""Paper maximum threshold report: the paper minimum threshold schema, asking about a maximum.

The kinds, the parsing, the scoring and the training answers are paper_minimum_threshold.py's,
unchanged: a threshold is one number on the attribute's range whichever side of it passes. Only the
wording differs, so this schema reads its questions from questions/paper_maximum_threshold/ (or
report_schema.questions_dir), where prompts ask about a maximum and answer keys are maximum_value
and maximum_level.

Reads object.attribute_count, report_schema (batches, questions, questions_dir).
"""
from pathlib import Path

from .paper_minimum_threshold import PaperMinimumThresholdSchema

QUESTIONS_DIR = Path(__file__).resolve().parents[2] / "questions" / "paper_maximum_threshold"


class PaperMaximumThresholdSchema(PaperMinimumThresholdSchema):
    name = "paper_maximum_threshold"

    def __init__(self, attribute_count, batches=None, questions=None, questions_dir=None):
        super().__init__(attribute_count, batches=batches, questions=questions,
                         questions_dir=questions_dir if questions_dir is not None else QUESTIONS_DIR)


def build(params):
    return PaperMaximumThresholdSchema(params["attribute_count"], batches=params.get("batches"),
                                       questions=params.get("questions"), questions_dir=params.get("questions_dir"))
