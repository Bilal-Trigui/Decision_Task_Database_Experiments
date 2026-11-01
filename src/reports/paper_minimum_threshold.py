"""Paper minimum threshold report: the questions asked about a screened rule, each one a folder.

THE QUESTIONS LIVE IN FOLDERS
Every question is a folder under questions/paper_minimum_threshold/ (or report_schema.questions_dir):

  questions/paper_minimum_threshold/<name>/prompt.txt     the wording, word for word, with placeholders
  questions/paper_minimum_threshold/<name>/question.json  its kind, its answer keys, a description

The folder's name is the question's name: the name a settings file selects it by, the report_batch
column of its results rows, and the name of its introspection training file. To reword a question,
edit its prompt.txt; to try a second wording beside the first, copy the folder under a new name and
select both. A data folder's introspection files are checked against the wording and rebuilt when it
changes (data.load_introspection_examples).

prompt.txt placeholders, filled per scenario:
  {word}        the attribute count as a word ("five")
  {count}       the attribute count as a number
  {listing}     the attributes in order with their numbers: "1 caffeine_content, 2 leaf_size, ..."
  {keys}        the key count of a combined_fraction reply (n + 3)
  {units_keys}  the key count of a combined_units reply (n + 2)

WHAT A KIND DOES
The kind says how a question's training answer is written and how a reply is read and scored. The
wording and the key names are the folder's; the kinds are code, so a new kind needs code here.

  weights              n weights under the attribute names, -100 to 100. Block main. The training
                       answer is the weights rescaled so the largest magnitude is 100 and rounded,
                       Plunkett's convention
  screen_index         which attribute, 1 to n, carries the screen. Block screen_index. Every persona
                       has one, so there is no answer for "none": 0 is unreadable
  screen_cut_fraction  the screen's minimum as a fraction 0 to 1 of the attribute's range (0.25, 0.5
                       or 0.75 under the paper rule). Block screen_cut
  screen_cut_units     the screened attribute's number and its minimum in that attribute's own units,
                       as the options list them ("35 mg per cup"). The value is put on the named
                       attribute's range and scored on block screen_cut, so the reply is not told
                       which attribute is screened. Needs the scenario's ranges (batches_for)
  screen_weight        the screened attribute's own weight once passed, -100 to 100. Always 0 under
                       the paper rule, so it can be answered with no self-knowledge. Block screen_weight
  cut                  the minimum threshold schema's per-attribute cut report. Block cut
  combined_fraction    weights + screened_dimension + minimum (fraction) + screen weight in one reply,
                       split into main, screen_index, screen_cut and screen_weight
  combined_units       weights + screened_dimension + minimum in units, split into main, screen_index
                       and screen_cut. Needs the scenario's ranges

A combined reply counts only when all its keys are present and every part is readable. Answers are
JSON. A reply with any other keys, or a value that is not a number, is unreadable and counts against
the parse rate, and so does an index outside 1 to n, a fraction outside 0 to 1, a value outside the
attribute's range, or a weight outside -100 to 100.

CHOOSING QUESTIONS IN A SETTINGS FILE
  "report_schema": {"questions": {"main": {"ask": true, "train": true},
                                  "screen_cut_units": {"ask": true, "train": false}, ...}}
ask: asked at every checkpoint and scored. train: an introspection training example too. A question
asked but not trained on is a held-out probe; one trained on but never asked is refused. Questions
come in the order listed. The older "batches": [...] list still works and means ask and train each.
With neither, the default is main, screen_index, screen_cut and screen_weight.

Averaging. A persona's report is the mean of its readable replies, as for every schema, with two
exceptions. For screen_index each reply is a vote, so the persona names the attribute most replies
named. For screen_cut and screen_weight, single numbers scored by their distance from one level, the
persona's report is its most common reply (the earliest of those tied): a mean of replies scattered
around a level lands near it, so ten random screen weights averaged score about 0.87 against a
hidden 0, while one random reply scores about 0.5.

Reads object.attribute_count, report_schema (batches, questions, questions_dir).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .base import Batch, Schema, json_number

REPO = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = REPO / "questions" / "paper_minimum_threshold"

# kind -> (block a reply is scored on, or the parts a combined reply splits into, and how many keys it
# answers beyond the n attribute names when the attribute names are its keys)
KINDS = {
    "weights": {"block": "main", "keys": None},
    "cut": {"block": "cut", "keys": None},
    "screen_index": {"block": "screen_index", "keys": 1},
    "screen_cut_fraction": {"block": "screen_cut", "keys": 1},
    "screen_cut_units": {"block": "screen_cut", "keys": 2},
    "screen_weight": {"block": "screen_weight", "keys": 1},
    "combined_fraction": {"block": "main", "keys": 3, "parts": ("main", "screen_index", "screen_cut", "screen_weight")},
    "combined_units": {"block": "main", "keys": 2, "parts": ("main", "screen_index", "screen_cut")},
}
UNITS_KINDS = ("screen_cut_units", "combined_units")
FULL_LATENT_KINDS = ("screen_weight", "combined_fraction", "combined_units")   # given weights and cuts, not one block
DEFAULT_BATCHES = ("main", "screen_index", "screen_cut", "screen_weight")
SINGLE_LEVELS = ("screen_cut", "screen_weight")     # one number each, aggregated as the most common reply
PLACEHOLDERS = ("word", "count", "listing", "keys", "units_keys")

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n):
    return _NUMBER_WORDS.get(n, str(n))


def screen_prompt(template, names):
    """A prompt.txt filled in for one scenario's attribute names."""
    n = len(names)
    listing = ", ".join(f"{i} {name}" for i, name in enumerate(names, start=1))
    return template.format(word=_word(n), listing=listing, count=n,
                           keys=n + KINDS["combined_fraction"]["keys"], units_keys=n + KINDS["combined_units"]["keys"])


def _decimals(low, high):
    """Decimals for a minimum in units: one more than Plunkett's rounding of option values (data.rounding_precision),
    so a level a quarter of the way along a short range (2.575 on 0.1 to 10) is not rounded a fifth of a step off."""
    from src.data import rounding_precision
    return rounding_precision({"range": [low, high]}) + 1


def resolve_questions_dir(path=None):
    """The folder the questions are read from: report_schema.questions_dir, relative to the working folder
    or to the repository, else questions/paper_minimum_threshold."""
    if path is None:
        return QUESTIONS_DIR
    path = Path(path)
    return path if path.is_absolute() or path.exists() else REPO / path


def load_question(folder):
    """One question folder as a dict: name, kind, keys (or None), description, template. Checks it."""
    folder = Path(folder)
    spec_path, prompt_path = folder / "question.json", folder / "prompt.txt"
    if not spec_path.exists() or not prompt_path.exists():
        raise FileNotFoundError(f"question {folder.name} needs question.json and prompt.txt in {folder}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{spec_path}: kind {kind!r} is not one of {sorted(KINDS)}")
    keys = spec.get("keys")
    want = KINDS[kind]["keys"]
    if want is None and keys is not None:
        raise ValueError(f"{spec_path}: a {kind} question answers under the attribute names; it takes no keys")
    if want is not None and (not isinstance(keys, list) or len(keys) != want or len(set(keys)) != want
                             or not all(isinstance(k, str) and k for k in keys)):
        raise ValueError(f"{spec_path}: a {kind} question needs {want} distinct key names, got {keys!r}")
    template = prompt_path.read_bytes().decode("utf-8")
    try:
        screen_prompt(template, ["a", "b"])
    except (KeyError, IndexError, ValueError) as err:
        raise ValueError(f"{prompt_path}: placeholder {err} is not one of {{{'}, {'.join(PLACEHOLDERS)}}}; "
                         "write a literal brace as {{ or }}") from None
    return {"name": folder.name, "kind": kind, "keys": keys, "description": spec.get("description", ""),
            "template": template}


def available_questions(path=None):
    """Every question folder's name, sorted."""
    root = resolve_questions_dir(path)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "question.json").exists())


def choose(batches=None, questions=None):
    """Which questions are asked and which are trained on, from report_schema: [(name, trained)] in order."""
    if batches is not None and questions is not None:
        raise ValueError("report_schema takes either batches (the older list) or questions, not both")
    if questions is not None:
        if not isinstance(questions, dict) or not questions:
            raise ValueError('report_schema.questions must map question names to {"ask": ..., "train": ...}')
        chosen = []
        for name, flags in questions.items():
            if not isinstance(flags, dict) or set(flags) - {"ask", "train"}:
                raise ValueError(f'report_schema.questions.{name} must be {{"ask": true/false, "train": true/false}}')
            ask, train = flags.get("ask", True), flags.get("train", True)
            if not isinstance(ask, bool) or not isinstance(train, bool):
                raise ValueError(f"report_schema.questions.{name}: ask and train must be true or false")
            if train and not ask:
                raise ValueError(f"report_schema.questions.{name} is trained on but never asked, so its training "
                                 "could never be measured; set ask to true, or train to false")
            if ask:
                chosen.append((name, train))
        if not chosen:
            raise ValueError("report_schema.questions asks nothing; set ask to true on at least one question")
        return chosen
    return [(name, True) for name in (list(batches) if batches else list(DEFAULT_BATCHES))]


class PaperMinimumThresholdSchema(Schema):
    name = "paper_minimum_threshold"

    def __init__(self, attribute_count, batches=None, questions=None, questions_dir=None):
        super().__init__(attribute_count)
        root = resolve_questions_dir(questions_dir)
        known = available_questions(root)
        chosen = choose(batches, questions)
        unknown = [name for name, _ in chosen if name not in known]
        if unknown:
            raise ValueError(f"report_schema names {unknown}, which {root} has no folder for; it has {known}")
        self.root = root
        self.chosen = [name for name, _ in chosen]
        self.trained = {name: trained for name, trained in chosen}
        self.specs = {name: load_question(root / name) for name in self.chosen}

    def batches_for(self, scenario):
        mins, maxs = getattr(scenario, "mins", None), getattr(scenario, "maxs", None)
        return self.batches(scenario.names, mins, maxs)

    def batches(self, attribute_names, mins=None, maxs=None):
        names = list(attribute_names)[: self.n]
        ranges = dict(mins=None if mins is None else tuple(float(v) for v in mins[: self.n]),
                      maxs=None if maxs is None else tuple(float(v) for v in maxs[: self.n]))
        out = []
        for question in self.chosen:
            spec = self.specs[question]
            kind = KINDS[spec["kind"]]
            keys = names + list(spec["keys"]) if kind.get("parts") else (spec["keys"] or names)
            out.append(Batch(name=question, block=kind["block"], keys=list(keys),
                             prompt_base=screen_prompt(spec["template"], names), names=names,
                             trained=self.trained[question], parts=tuple(kind.get("parts", ())),
                             kind=spec["kind"], **ranges))
        return out

    # -- reading a single screen answer --

    def _screen_value(self, kind, value):
        """One screen answer in its block's shape, or None where it is out of range."""
        if kind == "screen_index":
            # one vote, in the cut block's shape: 1 on the named attribute
            if not value.is_integer() or not 1 <= value <= self.n:
                return None
            votes = np.zeros(self.n)
            votes[int(value) - 1] = 1.0
            return votes
        if kind == "screen_cut_fraction":
            # a fraction of range in the answer, a percentage of range in the latent
            return np.array([value * 100.0]) if 0 <= value <= 1 else None
        return np.array([value]) if -100 <= value <= 100 else None      # screen_weight

    @staticmethod
    def _ranges(batch):
        if batch.mins is None or batch.maxs is None:
            raise ValueError(f"question '{batch.name}' answers in the attributes' own units and needs their ranges: "
                             "build it with schema.batches_for(scenario)")
        return np.asarray(batch.mins, float), np.asarray(batch.maxs, float)

    def _units_to_percent(self, batch, index, value):
        """(screened_dimension, minimum in units) as (votes, [percent of that attribute's range]), or None."""
        votes = self._screen_value("screen_index", index)
        if votes is None:
            return None
        mins, maxs = self._ranges(batch)
        j = int(index) - 1
        percent = (value - mins[j]) / (maxs[j] - mins[j]) * 100.0
        if not 0 <= percent <= 100:
            return None     # outside the attribute's range, as a fraction outside 0 to 1 is for screen_cut
        return votes, np.array([percent])

    def parse(self, text, batch):
        values = super().parse(text, batch)
        if values is None:
            return None
        kind, n = batch.kind or batch.name, self.n
        if kind in UNITS_KINDS:
            read = self._units_to_percent(batch, float(values[-2]), float(values[-1]))
            if read is None:
                return None
            votes, percent = read
            # screen_cut_units is scored on screen_cut alone; its index only says which range the value is on
            return percent if kind == "screen_cut_units" else np.concatenate([values[:n], votes, percent])
        if kind == "combined_fraction":
            # the n weights as they are, then each screen answer read as its own question reads it
            screen = [self._screen_value(k, float(v)) for k, v in
                      zip(("screen_index", "screen_cut_fraction", "screen_weight"), values[n:])]
            if any(part is None for part in screen):
                return None
            return np.concatenate([values[:n], *screen])
        if kind in ("screen_index", "screen_cut_fraction", "screen_weight"):
            return self._screen_value(kind, float(values[0]))
        return values       # weights, cut

    def split(self, batch, values):
        """A combined reply as its blocks; every other question answers its own block."""
        kind, n = batch.kind or batch.name, self.n
        if kind == "combined_units":
            return {"main": values[:n], "screen_index": values[n:2 * n], "screen_cut": values[2 * n:2 * n + 1]}
        if kind == "combined_fraction":
            return {"main": values[:n], "screen_index": values[n:2 * n],
                    "screen_cut": values[2 * n:2 * n + 1], "screen_weight": values[2 * n + 1:2 * n + 2]}
        return super().split(batch, values)

    def aggregate(self, block, replies):
        """screen_cut and screen_weight take the most common reply, the earliest of any tied; see Averaging."""
        if block not in SINGLE_LEVELS:
            return super().aggregate(block, replies)
        counts = Counter(float(np.ravel(r)[0]) for r in replies)
        return np.array([counts.most_common(1)[0][0]])

    # -- writing a training answer --

    @staticmethod
    def _screened(cuts, name):
        if not np.any(np.asarray(cuts) > 0):
            raise ValueError(f"{name} asks which attribute is screened, and this latent screens none")
        return int(np.argmax(cuts))

    def _screen_answers(self, batch, latent):
        """The screen answers from the full latent (weights then cuts), in the kind's key order."""
        kind, n = batch.kind or batch.name, self.n
        weights, cuts = latent[:n], latent[n:]
        j = self._screened(cuts, batch.name)
        if kind in UNITS_KINDS:
            mins, maxs = self._ranges(batch)
            value = mins[j] + cuts[j] / 100.0 * (maxs[j] - mins[j])
            return [j + 1, json_number(round(value, _decimals(mins[j], maxs[j])))]
        answers = {"screen_index": j + 1, "screen_cut_fraction": json_number(cuts.max() / 100.0),
                   "screen_weight": json_number(weights[j])}
        if kind == "combined_fraction":
            return [answers["screen_index"], answers["screen_cut_fraction"], answers["screen_weight"]]
        return [answers[kind]]

    @staticmethod
    def _scaled_weights(weights):
        top = np.max(np.abs(weights))
        scaled = weights * (100.0 / top) if top else weights
        return [int(round(v)) for v in scaled]

    def format_answer(self, batch, values):
        """The JSON training answer. A screen or combined question is given the full latent (weights then
        cuts), screen_index, screen_cut_fraction and screen_cut_units the cut block alone, as the pipeline does."""
        values = np.asarray(values, dtype=float)
        kind, n = batch.kind or batch.name, self.n
        if kind == "cut":
            return super().format_answer(batch, values)
        if kind == "weights":
            return json.dumps(dict(zip(batch.keys, self._scaled_weights(values[:n]))))
        latent = values if kind in FULL_LATENT_KINDS else np.concatenate([np.zeros(n), values])
        if kind in ("combined_fraction", "combined_units"):
            answer = dict(zip(batch.keys[:n], self._scaled_weights(values[:n])))
            answer.update(zip(batch.keys[n:], self._screen_answers(batch, latent)))
            return json.dumps(answer)
        return json.dumps(dict(zip(batch.keys, self._screen_answers(batch, latent))))


def build(params):
    return PaperMinimumThresholdSchema(params["attribute_count"], batches=params.get("batches"),
                                       questions=params.get("questions"), questions_dir=params.get("questions_dir"))
