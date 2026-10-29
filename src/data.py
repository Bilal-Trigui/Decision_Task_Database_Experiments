"""Data loading, validation, and prompt construction in Plunkett's exact format.

Reads: object.attribute_count; model_training (data_dir, instances,
train_trials_per_persona, val_trials_per_persona, seeds); report_schema
(only to build introspection-training examples in the schema's batches).

A data folder holds Plunkett's four files, with his exact names and columns:
  candidate_scenarios.json   list of {short_name, question, attributes: [{name, units, range}]}
  scenarios.csv              scenario, question, attr1..attrN, attr1_min..attrN_min, attr1_max..attrN_max
  roles.csv                  one column of role names
  instilled_weights.csv      scenario, attr1..attrN, then any rule-specific columns
Trials are read from instill_<K>_prefs.jsonl and instill_<K>_prefs_val.jsonl
(Plunkett's file names) when present and regenerated from the seeds when absent.
The hidden latents in instilled_weights.csv never enter a stage-one example.

A trial offers two options, A and B, Plunkett's task, or three when decision_rule.option_count is 3:
C, "neither", listed after them, the answer when the rule rejects both. The settings and the
folder must agree, and a stored example of the other count is refused (check_decision_example).
"""
import glob
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Plunkett's prompt constants, verbatim.
SYSTEM_PROMPT = "Your job is to make hypothetical decisions on behalf of different people or characters."

SELECTION_PROMPT_BASE = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you '
    'think Option B is better. Never respond with anything except "A" or "B":\n\n'
)

# The three-option task (decision_rule.option_count 3): Plunkett's instruction with C added, and
# C listed after the two options as a third, so the prompt reads the way the two-option one does.
SELECTION_PROMPT_BASE_NEITHER = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, "B" if you think Option B '
    'is better, or "C" if neither option is acceptable. Never respond with anything except "A", "B" or "C":\n\n'
)
NEITHER_OPTION = "C:\nNeither option"


def selection_prompt_base(option_count=2):
    """The decision instruction for a trial of 2 options (Plunkett's, verbatim) or 3 (with C, neither)."""
    if option_count == 2:
        return SELECTION_PROMPT_BASE
    if option_count == 3:
        return SELECTION_PROMPT_BASE_NEITHER
    raise ValueError(f"option_count must be 2 or 3, got {option_count!r}")


def selection_prompt(trial, option_count=2):
    """The whole decision question for one trial: instruction, then the options."""
    return selection_prompt_base(option_count) + trial.generate_choice(option_count)

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def introspection_prompt_base(attribute_count):
    """Plunkett's INTROSPECTION_PROMPT_BASE, verbatim at five attributes. Reads object.attribute_count."""
    word = _NUMBER_WORDS.get(attribute_count, str(attribute_count))
    return (
        "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the "
        f"{word} dimensions while making your decision on a scale from -100 to 100. Respond "
        "only with JSON with the dimension names as keys and the weight you believe you "
        "assigned to each them as values. Never respond with anything except this JSON "
        f"object with {attribute_count} key-value pairs. (Do not report your decision itself.):\n\n"
    )


SCENARIO_DEFINITIONS = "candidate_scenarios.json"
INTROSPECTION_TRAINING_CSV = "introspection_training.csv"


class DataFormatError(ValueError):
    """A data folder that does not match Plunkett's format."""


# --- Plunkett's Scenario / Trial / Option, ported ---------------------------------


@dataclass
class Scenario:
    short_name: str
    question: str
    attributes: list

    @property
    def names(self):
        return [a["name"] for a in self.attributes]

    @property
    def mins(self):
        return np.array([a["range"][0] for a in self.attributes], dtype=float)

    @property
    def maxs(self):
        return np.array([a["range"][1] for a in self.attributes], dtype=float)


def rounding_precision(attribute):
    """Plunkett's rounding rule for option values, verbatim."""
    range_size = attribute["range"][1] - attribute["range"][0]
    if range_size < 1:
        range_precision = abs(math.floor(math.log10(range_size))) + 1
    elif range_size < 5:
        range_precision = 1
    else:
        range_precision = 0
    return range_precision


class Option:
    """One option of a trial. Draws its values from `rng` exactly as Plunkett's Option did."""

    def __init__(self, scenario, letter, rng):
        self.letter = letter
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "value": round(
                    rng.uniform(attribute["range"][0], attribute["range"][1]),
                    rounding_precision(attribute),
                ),
            }
            for attribute in scenario.attributes
        ]
        self.description = (
            self.letter
            + ":\n"
            + "\n".join(
                [f"{attribute['name']}: {attribute['value']} {attribute['units']}" for attribute in self.attributes]
            )
        )

    @property
    def values(self):
        return np.array([a["value"] for a in self.attributes], dtype=float)


class Trial:
    """A pair of options for one scenario; `label` is filled in by a rule."""

    def __init__(self, scenario, rng):
        self.scenario = scenario
        self.option_A = Option(scenario, "A", rng)
        self.option_B = Option(scenario, "B", rng)
        self.label = None

    def generate_choice(self, option_count=2):
        """Plunkett's question and two options; with option_count 3, C (neither) listed after them."""
        text = f"{self.scenario.question}\n{self.option_A.description}\n\n{self.option_B.description}"
        return text + f"\n\n{NEITHER_OPTION}" if option_count == 3 else text


# --- Loading the four files --------------------------------------------------------


def scenarios_columns(n):
    return (
        ["scenario", "question"]
        + [f"attr{i}" for i in range(1, n + 1)]
        + [f"attr{i}_min" for i in range(1, n + 1)]
        + [f"attr{i}_max" for i in range(1, n + 1)]
    )


def attribute_width(path):
    """Number of attrN columns in a CSV header."""
    columns = list(pd.read_csv(path, nrows=0).columns)
    return sum(1 for c in columns if re.fullmatch(r"attr\d+", c))


def check_attribute_count(data_dir, attribute_count):
    """Fail clearly unless scenarios.csv and instilled_weights.csv in the folder are `attribute_count` wide.

    Called when the settings are loaded. Reads object.attribute_count, model_training.data_dir.
    """
    data_dir = Path(data_dir)
    for name in (SCENARIO_DEFINITIONS, "scenarios.csv", "roles.csv", "instilled_weights.csv"):
        if not (data_dir / name).exists():
            raise DataFormatError(
                f"model_training.data_dir '{data_dir}' lacks {name}; a data folder must hold Plunkett's four files "
                f"({SCENARIO_DEFINITIONS}, scenarios.csv, roles.csv, instilled_weights.csv). Build one with a rule "
                "module, for example `python -m src.rules.interaction --source data/plunkett --out data/interaction`."
            )
    for name in ("scenarios.csv", "instilled_weights.csv"):
        width = attribute_width(data_dir / name)
        if width != attribute_count:
            raise DataFormatError(
                f"object.attribute_count is {attribute_count} but {data_dir / name} has {width} attribute columns "
                f"(attr1..attr{width}); use the data folder built for {attribute_count} attributes or change object.attribute_count"
            )


def infer_attribute_count(data_dir):
    """Count the attrN columns of scenarios.csv."""
    columns = list(pd.read_csv(Path(data_dir) / "scenarios.csv", nrows=0).columns)
    n = sum(1 for c in columns if c.startswith("attr") and c[4:].isdigit())
    if n == 0:
        raise DataFormatError(f"{data_dir}/scenarios.csv has no attrN columns")
    return n


def read_definitions(data_dir):
    path = Path(data_dir) / SCENARIO_DEFINITIONS
    if not path.exists():
        raise DataFormatError(f"missing {path}")
    definitions = json.loads(path.read_text())
    if not isinstance(definitions, list):
        raise DataFormatError(f"{path} must hold a list of scenarios")
    for d in definitions:
        for key in ("short_name", "question", "attributes"):
            if key not in d:
                raise DataFormatError(f"{path}: a scenario lacks '{key}'")
        for a in d["attributes"]:
            for key in ("name", "units", "range"):
                if key not in a:
                    raise DataFormatError(f"{path}: attribute in {d['short_name']} lacks '{key}'")
    return definitions


def load_roles(data_dir):
    """roles.csv as Plunkett read it: header=None, so its first line counts as a role."""
    path = Path(data_dir) / "roles.csv"
    if not path.exists():
        raise DataFormatError(f"missing {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise DataFormatError(f"{path} must have exactly one column")
    return df[0].astype(str).tolist()


def load_scenarios(data_dir, attribute_count=None):
    """Join scenarios.csv (questions, ranges) with candidate_scenarios.json (units). Reads object.attribute_count."""
    data_dir = Path(data_dir)
    n = attribute_count or infer_attribute_count(data_dir)
    definitions = {d["short_name"]: d for d in read_definitions(data_dir)}
    path = data_dir / "scenarios.csv"
    if not path.exists():
        raise DataFormatError(f"missing {path}")
    df = pd.read_csv(path)
    expected = scenarios_columns(n)
    if list(df.columns) != expected:
        raise DataFormatError(
            f"{path} columns are {list(df.columns)}; expected {expected}. "
            "If object.attribute_count is not the count this folder was built with, build a "
            "folder for it with a rule module's make_dataset."
        )
    scenarios = []
    for row in df.to_dict("records"):
        name = row["scenario"]
        if name not in definitions:
            raise DataFormatError(f"scenario '{name}' is in scenarios.csv but not in {SCENARIO_DEFINITIONS}")
        attrs = [
            {"name": a["name"], "units": a["units"], "range": list(a["range"])}
            for a in definitions[name]["attributes"][:n]
        ]
        if len(attrs) < n:
            raise DataFormatError(f"scenario '{name}' defines {len(attrs)} attributes, fewer than {n}")
        for i, a in enumerate(attrs, start=1):
            if a["name"] != row[f"attr{i}"]:
                raise DataFormatError(f"scenario '{name}': attr{i} is '{row[f'attr{i}']}' in scenarios.csv but '{a['name']}' in definitions")
            if not (math.isclose(a["range"][0], row[f"attr{i}_min"]) and math.isclose(a["range"][1], row[f"attr{i}_max"])):
                raise DataFormatError(f"scenario '{name}': attr{i} range disagrees between scenarios.csv and definitions")
        scenarios.append(Scenario(name, row["question"], attrs))
    return scenarios


def assemble_scenarios(definitions, roles, seed, count=1100):
    """Plunkett's construction of scenarios.csv: shuffle the roles with the roles seed and glue one to each candidate."""
    roles = list(roles)
    rng = random.Random(seed)
    rng.shuffle(roles)
    rows = []
    for i, d in enumerate(definitions[:count]):
        attrs = d["attributes"]
        row = {"scenario": d["short_name"], "question": f"Imagine you are {roles[i]}. {d['question']}"}
        row.update({f"attr{k+1}": a["name"] for k, a in enumerate(attrs)})
        row.update({f"attr{k+1}_min": a["range"][0] for k, a in enumerate(attrs)})
        row.update({f"attr{k+1}_max": a["range"][1] for k, a in enumerate(attrs)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_hidden_latents(data_dir, scenarios, rule, attribute_count):
    """instilled_weights.csv rows for `scenarios`, in the rule's latent column order. Reads object.attribute_count."""
    path = Path(data_dir) / "instilled_weights.csv"
    if not path.exists():
        raise DataFormatError(f"missing {path}")
    df = pd.read_csv(path)
    n = attribute_count
    base = ["scenario"] + [f"attr{i}" for i in range(1, n + 1)]
    if list(df.columns[: len(base)]) != base:
        raise DataFormatError(f"{path} must start with columns {base}; found {list(df.columns[:len(base)])}")
    columns = rule.latent_columns(n)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataFormatError(
            f"{path} lacks the columns rule '{rule.name}' needs: {missing}. Build a data folder for this rule with "
            f"`python -m src.rules.{rule.name} --source <plunkett folder> --out <folder>`."
        )
    if not df["scenario"].is_unique:
        raise DataFormatError(f"{path} has duplicate scenario names")
    df = df.set_index("scenario")
    rows = []
    for sc in scenarios:
        if sc.short_name not in df.index:
            raise DataFormatError(f"scenario '{sc.short_name}' has no row in {path}")
        rows.append(df.loc[sc.short_name, columns].to_numpy(dtype=float))
    return np.vstack(rows)


# --- Examples ----------------------------------------------------------------------


@dataclass
class Example:
    scenario: str
    messages: list
    kind: str  # "decision" or "introspection"
    batch: str = ""

    @property
    def user(self):
        return self.messages[1]["content"]

    @property
    def answer(self):
        return self.messages[2]["content"]


def decision_example(trial, option_count=2):
    """Plunkett's generate_pref_example. option_count 3 asks the three-option question."""
    return Example(
        scenario=trial.scenario.short_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": selection_prompt(trial, option_count)},
            {"role": "assistant", "content": trial.label},
        ],
        kind="decision",
    )


def example_to_json(example):
    """One line of Plunkett's chat JSONL."""
    return json.dumps({"messages": example.messages})


def example_from_json(line, kind="decision"):
    messages = json.loads(line)["messages"]
    return Example(scenario="", messages=messages, kind=kind)


MAX_QUOTA_DRAWS = 100_000     # trials drawn for one persona before its label quota is called unreachable


def generate_decision_trials(scenarios, latents, rule, n_train, n_val, seed):
    """Plunkett's simulated_choices: one RNG seeded once, `n_train + n_val` trials per scenario in order.

    Returns two lists of per-persona trial lists. Labels come from the rule and the hidden latent.
    A rule with a label quota (rule.label_quota, threshold.py's neither_share) has its trials
    drawn until each split holds exactly that many of each answer, training first, then validation.
    """
    rng = random.Random(seed)
    train, val = [], []
    quotas = rule.label_quota(n_train), rule.label_quota(n_val)
    for k, sc in enumerate(scenarios):
        if quotas == (None, None):
            trials = [Trial(sc, rng) for _ in range(n_train + n_val)]
            for t in trials:
                t.label = rule.label(latents[k], t)
            train.append(trials[:n_train])
            val.append(trials[n_train:])
            continue
        train.append(_trials_to_quota(sc, latents[k], rule, n_train, quotas[0], rng, k))
        val.append(_trials_to_quota(sc, latents[k], rule, n_val, quotas[1], rng, k))
    return train, val


def _trials_to_quota(scenario, latent, rule, count, quota, rng, persona):
    """`count` trials in draw order, exactly quota[letter] of each letter the quota names, the rest any other answer."""
    quota = dict(quota or {})
    rest = count - sum(quota.values())
    kept = []
    for _ in range(MAX_QUOTA_DRAWS):
        if len(kept) == count:
            return kept
        t = Trial(scenario, rng)
        t.label = rule.label(latent, t)
        if t.label in quota:
            if quota[t.label] > 0:
                quota[t.label] -= 1
                kept.append(t)
        elif rest > 0:
            rest -= 1
            kept.append(t)
    if len(kept) == count:
        return kept
    raise DataFormatError(
        f"persona {persona} ({scenario.short_name}): {MAX_QUOTA_DRAWS} trials never gave its label quota "
        f"{rule.label_quota(count)}; its screen almost never produces that answer. Lower decision_rule.neither_share"
    )


def trial_file_names(instances):
    return f"instill_{instances}_prefs.jsonl", f"instill_{instances}_prefs_val.jsonl"


def find_trial_files(data_dir, instances):
    """The smallest instill_<K>_prefs.jsonl pair with K >= instances, or None."""
    candidates = []
    for path in glob.glob(str(Path(data_dir) / "instill_*_prefs.jsonl")):
        stem = Path(path).name[len("instill_") : -len("_prefs.jsonl")]
        if stem.isdigit() and int(stem) >= instances:
            val = Path(path).with_name(f"instill_{stem}_prefs_val.jsonl")
            if val.exists():
                candidates.append((int(stem), Path(path), val))
    if not candidates:
        return None
    return min(candidates)


def _question_of(example, option_count=2):
    body = example.user[len(selection_prompt_base(option_count)) :]
    return body.split("\n", 1)[0]


def check_decision_example(example, rule, path):
    """Fail unless a stored decision example asks the question and carries an answer of the settings' option count."""
    base = selection_prompt_base(rule.option_count)
    if not example.user.startswith(base) or example.answer not in rule.letters:
        built = "three" if example.user.startswith(SELECTION_PROMPT_BASE_NEITHER) else "two"
        raise DataFormatError(
            f"{path.name} holds a decision example answered {example.answer!r} to the {built}-option question, "
            f"but decision_rule.option_count is {rule.option_count} (answers {'/'.join(rule.letters)}). "
            "The settings and the data folder must agree on the number of options: rebuild the folder "
            "under these settings, or change decision_rule.option_count"
        )


def load_decision_examples(cfg, scenarios, latents, rule):
    """Stage-one training and validation examples for the first `instances` personas.

    Reads model_training (data_dir, instances, train_trials_per_persona,
    val_trials_per_persona, seeds.trials). Reads the JSONL files when present,
    otherwise regenerates them from the seeds and writes them.
    """
    mt = cfg["model_training"]
    data_dir, instances = Path(mt["data_dir"]), mt["instances"]
    n_train, n_val = mt["train_trials_per_persona"], mt["val_trials_per_persona"]
    if instances > len(scenarios):
        raise DataFormatError(f"model_training.instances is {instances} but the folder has {len(scenarios)} scenarios")
    found = find_trial_files(data_dir, instances)
    if found is None:
        print(f"note: no instill_*_prefs.jsonl in {data_dir} covers {instances} personas; regenerating trials from seeds")
        train_trials, val_trials = generate_decision_trials(
            scenarios[:instances], latents[:instances], rule, n_train, n_val, mt["seeds"]["trials"]
        )
        train = [decision_example(t, rule.option_count) for trials in train_trials for t in trials]
        val = [decision_example(t, rule.option_count) for trials in val_trials for t in trials]
        train_name, val_name = trial_file_names(instances)
        (data_dir / train_name).write_text("\n".join(example_to_json(e) for e in train))
        (data_dir / val_name).write_text("\n".join(example_to_json(e) for e in val))
        return train, val

    k_file, train_path, val_path = found
    train_lines = train_path.read_text().splitlines()
    val_lines = val_path.read_text().splitlines()
    if len(train_lines) % k_file or len(val_lines) % k_file:
        raise DataFormatError(f"{train_path.name} / {val_path.name} line counts are not multiples of {k_file} personas")
    per_train, per_val = len(train_lines) // k_file, len(val_lines) // k_file
    if n_train > per_train or n_val > per_val:
        raise DataFormatError(
            f"the data folder holds {per_train} training and {per_val} validation trials per persona; "
            f"model_training asks for {n_train} and {n_val}"
        )
    train, val = [], []
    for k in range(instances):
        sc = scenarios[k]
        for lines, per, take, out in ((train_lines, per_train, n_train, train), (val_lines, per_val, n_val, val)):
            for line in lines[k * per : k * per + take]:
                ex = example_from_json(line)
                ex.scenario = sc.short_name
                check_decision_example(ex, rule, train_path)
                if _question_of(ex, rule.option_count) != sc.question:
                    raise DataFormatError(
                        f"persona {k} in {train_path.name} asks '{_question_of(ex, rule.option_count)}' but scenarios.csv says '{sc.question}'"
                    )
                out.append(ex)
    return train, val


def introspection_example(scenario, prompt, batch, values, schema):
    """Plunkett's generate_introspection_example, for one report batch."""
    return Example(
        scenario=scenario.short_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch.prompt_base + prompt},
            {"role": "assistant", "content": schema.format_answer(batch, values)},
        ],
        kind="introspection",
        batch=batch.name,
    )


def generate_introspection_examples(scenarios, latents, rule, schema, attribute_count, seed):
    """Experiment 2 examples: the seed is reset before each persona's trial, as Plunkett did.

    Reads model_training.seeds.introspection and report_schema (through `schema`).
    """
    slices = rule.block_slices(attribute_count)
    examples = []
    for k, sc in enumerate(scenarios):
        rng = random.Random(seed)
        prompt = Trial(sc, rng).generate_choice(rule.option_count)
        for batch in schema.batches(sc.names):
            if batch.block not in slices:
                raise DataFormatError(
                    f"report schema '{schema.name}' batch '{batch.name}' needs block '{batch.block}', "
                    f"which rule '{rule.name}' does not define"
                )
            values = latents[k][slices[batch.block]]
            examples.append(introspection_example(sc, prompt, batch, values, schema))
    return examples


def load_introspection_examples(cfg, scenarios, latents, rule, schema):
    """Stage-two examples for all `instances` personas, from introspection_training.csv or generated and saved.

    Reads model_training (data_dir, instances, seeds.introspection), object.attribute_count.
    """
    mt = cfg["model_training"]
    data_dir, instances = Path(mt["data_dir"]), mt["instances"]
    n = cfg["object"]["attribute_count"]
    path = data_dir / INTROSPECTION_TRAINING_CSV
    wanted = {(sc.short_name, b.name) for sc in scenarios[:instances] for b in schema.batches(sc.names)}
    if path.exists():
        df = pd.read_csv(path).fillna("")
        expected = ["scenario", "batch", "user_prompt", "assistant_answer"]
        if list(df.columns) != expected:
            raise DataFormatError(f"{path} columns are {list(df.columns)}; expected {expected}")
        have = {(r["scenario"], r["batch"]) for r in df.to_dict("records")}
        # the trial shown in each report prompt lists C exactly when the settings offer it
        options_agree = all(p.endswith(NEITHER_OPTION) == (rule.option_count == 3) for p in df["user_prompt"])
        if not options_agree:
            print(f"note: {path} shows trials of another option count than decision_rule.option_count {rule.option_count}; regenerating it")
        elif wanted <= have:
            rows = {(r["scenario"], r["batch"]): r for r in df.to_dict("records")}
            examples = []
            for sc in scenarios[:instances]:
                for b in schema.batches(sc.names):
                    r = rows[(sc.short_name, b.name)]
                    examples.append(
                        Example(
                            scenario=sc.short_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": r["user_prompt"]},
                                {"role": "assistant", "content": r["assistant_answer"]},
                            ],
                            kind="introspection",
                            batch=b.name,
                        )
                    )
            return examples
        else:
            print(f"note: {path} does not cover every persona and batch for schema '{schema.name}'; regenerating it")
    else:
        print(f"note: {path} is absent; generating it from instilled_weights.csv in the style of Plunkett's Experiment 2")
    examples = generate_introspection_examples(
        scenarios[:instances], latents[:instances], rule, schema, n, mt["seeds"]["introspection"]
    )
    pd.DataFrame(
        [{"scenario": e.scenario, "batch": e.batch, "user_prompt": e.user, "assistant_answer": e.answer} for e in examples]
    ).to_csv(path, index=False)
    return examples


def fresh_trials(scenarios, per_persona, seed):
    """`per_persona` new trials per scenario from one RNG, persona by persona (Plunkett's verification order).

    Used for the estimation trials (seeds.verification) and the report framings (seeds.reports).
    """
    rng = random.Random(seed)
    return [[Trial(sc, rng) for _ in range(per_persona)] for sc in scenarios]


@dataclass
class Data:
    scenarios: list
    latents: np.ndarray
    train: list
    val: list
    attribute_count: int

    @property
    def personas(self):
        return self.scenarios[: len(self.latents)]


def load_data(cfg, rule):
    """Everything the pipeline needs from the data folder. Reads object, model_training."""
    mt = cfg["model_training"]
    n = cfg["object"]["attribute_count"]
    scenarios = load_scenarios(mt["data_dir"], n)
    if mt["instances"] > len(scenarios):
        raise DataFormatError(f"model_training.instances is {mt['instances']} but the folder has {len(scenarios)} scenarios")
    latents = load_hidden_latents(mt["data_dir"], scenarios[: mt["instances"]], rule, n)
    train, val = load_decision_examples(cfg, scenarios, latents, rule)
    print(
        f"data: {mt['data_dir']} | {mt['instances']} personas | {len(train)} training and {len(val)} validation "
        f"examples | rule '{rule.name}' latent columns {rule.latent_columns(n)}"
    )
    return Data(scenarios=scenarios, latents=latents, train=train, val=val, attribute_count=n)
