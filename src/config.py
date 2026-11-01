"""Settings loading, validation, and component resolution.

The settings file has six blocks named after the environment frame (object,
decision_rule, model_training, model_estimating, report_schema,
model_hyperparameters) plus `gates` and `compute`. This module reads all of
them once, validates them, and resolves the interchangeable parts by name:
`src.rules.<decision_rule.type>`, `src.estimators.<model_estimating.type>`,
`src.reports.<report_schema.type>`, `src.compute.<compute.backend>`. Each of
those modules exposes `build(...)`. Adding a new rule, estimator, report
schema or backend means adding one file in the matching package; nothing here
needs editing.
"""
import copy
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

BLOCKS = [
    "object",
    "decision_rule",
    "model_training",
    "model_estimating",
    "report_schema",
    "model_hyperparameters",
    "gates",
    "compute",
]

DEFAULT_SEEDS = {
    "roles": 0,
    "weights": 1,
    "trials": 2,
    "verification": 5,
    "introspection": 6,
    "reports": 7,
}

BACKENDS = ("local", "colab", "api")
DEVICES = ("cpu", "cuda")
FINETUNE = ("lora", "full")
QUANTIZATION = ("none", "4bit")
PRECISION = ("no", "fp16", "bf16")
NUMBER = (int, float)

REQUIRED = {
    "object": {"type": str, "attribute_count": int},
    "decision_rule": {"type": str},
    "model_training": {
        "data_dir": str,
        "instances": int,
        "train_trials_per_persona": int,
        "val_trials_per_persona": int,
    },
    "model_estimating": {
        "type": str,
        "decision_temperature": NUMBER,
        "samples_per_trial": int,
        "estimation_trials_per_persona": (int, list),
        "chance_draws": int,
    },
    "report_schema": {
        "type": (str, list),
        "report_temperature": NUMBER,
        "samples_per_report": int,
        "max_new_tokens": int,
    },
    "model_hyperparameters": {
        "model_name": str,
        "quantization": str,
        "finetune": str,
        "lora_rank": int,
        "lora_alpha": NUMBER,
        "learning_rate": NUMBER,
        "batch_size": int,
        "loss_mask_answer_only": bool,
        "training_steps": int,
        "checkpoint_every": int,
        "introspection_training": bool,
        "introspection_steps": int,
        "introspection_batch_size": int,
        "seed": int,
    },
    "gates": {"min_decision_accuracy": NUMBER, "min_parse_rate": NUMBER},
    "compute": {"backend": str, "device": str, "mixed_precision": str, "eval_batch_size": int},
}


# Fields a block may carry beyond REQUIRED, because apply_defaults fills them. The three
# component blocks (decision_rule, model_estimating, report_schema) are deliberately absent:
# they also carry their component's own parameters, and the component validates those when it
# is built.
OPTIONAL = {
    "object": set(),
    "model_training": {"seeds"},
    "model_hyperparameters": {"gradient_checkpointing", "micro_batch_size", "introspection_at_steps",
                              "baseline_before_training", "lr_schedule", "introspection_targets",
                              "introspection_remeasure_decisions"},
    "gates": set(),
    "compute": {"mount_drive", "decision_eval_batch_size", "introspection_eval_batch_size"},
}

# The stages whose evaluation batch size can be set apart; each falls back to compute.eval_batch_size.
EVAL_STAGES = ("decision", "introspection")

# What introspection training can train toward, each its own branch (model_hyperparameters.introspection_targets):
#   hidden     the persona's own rule, from instilled_weights.csv (Plunkett's Experiment 2)
#   recovered  the rule read back from the model's own choices at the checkpoint it branched from
#   shuffled   another training persona's own rule, no persona keeping its own: the format, not the content
#   random     a rule drawn fresh from the decision rule, for no persona at all
INTROSPECTION_TARGETS = ("hidden", "recovered", "shuffled", "random")
LR_SCHEDULES = ("constant", "linear", "cosine")


class SettingsError(ValueError):
    """A settings file that the pipeline refuses to run."""


def apply_defaults(cfg):
    """Fill the few fields that may be omitted, with configs/default.json's values. Reads model_training.seeds, compute, decision_rule.option_count."""
    cfg.setdefault("model_training", {}).setdefault("seeds", {})
    for key, value in DEFAULT_SEEDS.items():
        cfg["model_training"]["seeds"].setdefault(key, value)
    compute = cfg.setdefault("compute", {})
    compute.setdefault("backend", "colab")
    compute.setdefault("device", "cpu" if compute["backend"] == "local" else "cuda")
    compute.setdefault("mixed_precision", "no" if compute["device"] == "cpu" else "fp16")
    compute.setdefault("eval_batch_size", 16)
    compute.setdefault("mount_drive", False)
    cfg.setdefault("decision_rule", {}).setdefault("option_count", 2)
    cfg.setdefault("report_schema", {}).setdefault("max_new_tokens", 200)
    cfg.setdefault("model_estimating", {}).setdefault("chance_draws", 1000)
    cfg.setdefault("model_hyperparameters", {}).setdefault("gradient_checkpointing", False)
    cfg["model_hyperparameters"].setdefault("micro_batch_size", None)
    cfg["model_hyperparameters"].setdefault("introspection_at_steps", None)
    cfg["model_hyperparameters"].setdefault("baseline_before_training", True)
    cfg["model_hyperparameters"].setdefault("lr_schedule", None)
    cfg["model_hyperparameters"].setdefault("introspection_targets", ["hidden"])
    cfg["model_hyperparameters"].setdefault("introspection_remeasure_decisions", False)
    cfg["model_estimating"].setdefault("shuffle_draws", 1000)
    return cfg


def _check_type(block, field, value, expected):
    if expected is int and isinstance(value, bool):
        raise SettingsError(f"{block}.{field} must be an integer, got a boolean")
    if expected == NUMBER and isinstance(value, bool):
        raise SettingsError(f"{block}.{field} must be a number, got a boolean")
    if not isinstance(value, expected):
        name = getattr(expected, "__name__", str(expected))
        raise SettingsError(f"{block}.{field} must be {name}, got {type(value).__name__}: {value!r}")


def validate(cfg):
    """Check every block, field, type and enumeration. Reads all eight blocks."""
    for block in BLOCKS:
        if block not in cfg or not isinstance(cfg[block], dict):
            raise SettingsError(f"settings are missing the '{block}' block")
        for field, expected in REQUIRED[block].items():
            if field not in cfg[block]:
                raise SettingsError(f"settings are missing {block}.{field}")
            _check_type(block, field, cfg[block][field], expected)

    for block, optional in OPTIONAL.items():
        known = set(REQUIRED[block]) | optional
        unknown = set(cfg[block]) - known
        if unknown:
            raise SettingsError(
                f"{block} has no field {sorted(unknown)}; it takes {sorted(known)}. "
                "A misspelled field would otherwise be ignored, and in a file that extends "
                "another it would silently leave the parent's value in place."
            )

    obj, mt, me, rs, mh, gates, comp = (
        cfg["object"],
        cfg["model_training"],
        cfg["model_estimating"],
        cfg["report_schema"],
        cfg["model_hyperparameters"],
        cfg["gates"],
        cfg["compute"],
    )
    if obj["type"] != "vector":
        raise SettingsError("object.type must be 'vector'; non-vector objects are out of scope")
    if obj["attribute_count"] < 2:
        raise SettingsError("object.attribute_count must be at least 2")
    options = cfg["decision_rule"]["option_count"]
    if isinstance(options, bool) or options not in (2, 3):
        raise SettingsError(
            "decision_rule.option_count must be 2 (A or B, Plunkett's task) or 3 (A, B, or C for neither); "
            f"got {options!r}"
        )
    for key in DEFAULT_SEEDS:
        if key not in mt["seeds"] or isinstance(mt["seeds"][key], bool) or not isinstance(mt["seeds"][key], int):
            raise SettingsError(f"model_training.seeds.{key} must be an integer")
    if mt["instances"] < 2:
        raise SettingsError("model_training.instances must be at least 2 (introspection training uses two folds)")
    for field in ("train_trials_per_persona", "val_trials_per_persona"):
        if mt[field] < 1:
            raise SettingsError(f"model_training.{field} must be at least 1")
    if me["decision_temperature"] < 0 or rs["report_temperature"] < 0:
        raise SettingsError("temperatures must be non-negative")
    for block, field in (
        ("model_estimating", "samples_per_trial"),
        ("model_estimating", "chance_draws"),
        ("report_schema", "samples_per_report"),
        ("report_schema", "max_new_tokens"),
        ("model_hyperparameters", "batch_size"),
        ("model_hyperparameters", "training_steps"),
        ("model_hyperparameters", "checkpoint_every"),
        ("model_hyperparameters", "introspection_steps"),
        ("model_hyperparameters", "introspection_batch_size"),
        ("model_hyperparameters", "lora_rank"),
        ("compute", "eval_batch_size"),
    ):
        if cfg[block][field] < 1:
            raise SettingsError(f"{block}.{field} must be at least 1")
    if isinstance(rs["type"], list) and not all(isinstance(t, str) for t in rs["type"]):
        raise SettingsError("report_schema.type must be a name or a list of names")
    if mh["finetune"] not in FINETUNE:
        raise SettingsError(f"model_hyperparameters.finetune must be one of {FINETUNE}")
    if mh["quantization"] not in QUANTIZATION:
        raise SettingsError(f"model_hyperparameters.quantization must be one of {QUANTIZATION}")
    if not isinstance(mh["gradient_checkpointing"], bool):
        raise SettingsError("model_hyperparameters.gradient_checkpointing must be true or false")
    micro = mh["micro_batch_size"]
    if micro is not None and (isinstance(micro, bool) or not isinstance(micro, int) or micro < 1):
        raise SettingsError("model_hyperparameters.micro_batch_size must be a positive integer or null")
    if not isinstance(mh["baseline_before_training"], bool):
        raise SettingsError("model_hyperparameters.baseline_before_training must be true or false")
    lr_schedule(cfg)    # refuses a malformed schedule
    targets = mh["introspection_targets"]
    if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
            or any(t not in INTROSPECTION_TARGETS for t in targets)):
        raise SettingsError(f"model_hyperparameters.introspection_targets must be a non-empty list of distinct names "
                            f"from {list(INTROSPECTION_TARGETS)}, got {targets!r}")
    if not isinstance(mh["introspection_remeasure_decisions"], bool):
        raise SettingsError("model_hyperparameters.introspection_remeasure_decisions must be true or false")
    draws = me.get("shuffle_draws")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise SettingsError("model_estimating.shuffle_draws must be a positive integer")
    trials = me["estimation_trials_per_persona"]
    counts = trials if isinstance(trials, list) else [trials]
    if not counts or any(isinstance(t, bool) or not isinstance(t, int) or t < 1 for t in counts):
        raise SettingsError(
            "model_estimating.estimation_trials_per_persona must be a positive integer, "
            "or a non-empty list of them to recover the rule from several trial counts"
        )
    if len(set(counts)) != len(counts):
        raise SettingsError(f"model_estimating.estimation_trials_per_persona lists a count twice: {trials}")
    at = mh["introspection_at_steps"]
    if at is not None:
        at = _introspection_steps(at)
        if not mh["introspection_training"]:
            raise SettingsError(
                "model_hyperparameters.introspection_at_steps is set but introspection_training is false; "
                "set introspection_training to true, or remove introspection_at_steps"
            )
        if len(set(at)) != len(at):
            raise SettingsError(f"model_hyperparameters.introspection_at_steps lists a step twice: {at}")
        outside = [s for s in at if not 1 <= s <= mh["training_steps"]]
        if outside:
            raise SettingsError(
                f"model_hyperparameters.introspection_at_steps {outside} lie outside decision training "
                f"(steps 1 to training_steps, {mh['training_steps']})"
            )
    if mh["finetune"] == "full" and mh["quantization"] != "none":
        raise SettingsError("model_hyperparameters.finetune 'full' needs quantization 'none'; a quantized base cannot be fully trained")
    for field in ("min_decision_accuracy", "min_parse_rate"):
        if not 0 <= gates[field] <= 1:
            raise SettingsError(f"gates.{field} must be between 0 and 1")
    for stage in EVAL_STAGES:
        size = comp.get(f"{stage}_eval_batch_size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise SettingsError(f"compute.{stage}_eval_batch_size must be a positive integer or null")
    if comp["backend"] not in BACKENDS:
        raise SettingsError(f"compute.backend must be one of {BACKENDS}")
    if comp["device"] not in DEVICES:
        raise SettingsError(f"compute.device must be one of {DEVICES}")
    if comp["mixed_precision"] not in PRECISION:
        raise SettingsError(f"compute.mixed_precision must be one of {PRECISION}")
    return cfg


def lr_schedule(cfg):
    """Decision training's learning-rate schedule, filled in: {"type", "warmup_steps", "final_fraction"}.

    model_hyperparameters.lr_schedule null (the default) is a constant rate, as every run before it.
    Otherwise {"type": "linear" or "cosine" (or "constant"), "warmup_steps": W, "final_fraction": f}:
    the rate climbs linearly from 0 over the first W steps (default 0), then falls from
    learning_rate to f x learning_rate (default 0) at training_steps, along a line or a half cosine.
    Introspection training keeps the constant rate. Refuses a malformed schedule.
    """
    mh = cfg["model_hyperparameters"]
    given = mh.get("lr_schedule")
    if given is None:
        return {"type": "constant", "warmup_steps": 0, "final_fraction": 1.0}
    if not isinstance(given, dict) or set(given) - {"type", "warmup_steps", "final_fraction"} or "type" not in given:
        raise SettingsError('model_hyperparameters.lr_schedule must be null or {"type": ..., "warmup_steps": ..., '
                            f'"final_fraction": ...}}, got {given!r}')
    kind = given["type"]
    if kind not in LR_SCHEDULES:
        raise SettingsError(f"model_hyperparameters.lr_schedule.type must be one of {LR_SCHEDULES}, got {kind!r}")
    warmup = given.get("warmup_steps", 0)
    final = given.get("final_fraction", 1.0 if kind == "constant" else 0.0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < mh["training_steps"]:
        raise SettingsError("model_hyperparameters.lr_schedule.warmup_steps must be an integer from 0 to below training_steps")
    if isinstance(final, bool) or not isinstance(final, NUMBER) or not 0 <= final <= 1:
        raise SettingsError("model_hyperparameters.lr_schedule.final_fraction must be a number from 0 to 1")
    return {"type": kind, "warmup_steps": warmup, "final_fraction": float(final)}


def learning_rate_at(cfg, step, stage="decision"):
    """The learning rate for optimizer step `step` (1-based) of `stage`, under lr_schedule."""
    import math

    base = cfg["model_hyperparameters"]["learning_rate"]
    sched = lr_schedule(cfg)
    if stage != "decision" or (sched["type"] == "constant" and sched["warmup_steps"] == 0):
        return base
    total, warmup = cfg["model_hyperparameters"]["training_steps"], sched["warmup_steps"]
    if step <= warmup:
        return base * step / warmup
    if sched["type"] == "constant":
        return base
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    shape = 1.0 - progress if sched["type"] == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
    final = sched["final_fraction"]
    return base * (final + (1.0 - final) * shape)


def decays(cfg):
    """True when the learning rate depends on where training ends (a schedule other than constant)."""
    return lr_schedule(cfg)["type"] != "constant"


def eval_batch_size(cfg, stage):
    """The evaluation batch size for `stage`, 'decision' or 'introspection'. Reads compute.

    compute.decision_eval_batch_size covers everything asked of a decision checkpoint (the base
    model included): the validation pass during decision training, the A/B decisions and the reports.
    compute.introspection_eval_batch_size covers the validation pass during introspection training
    and the reports asked after it, whose prompts and answers are far longer. Either one left out or
    null falls back to compute.eval_batch_size.
    """
    if stage not in EVAL_STAGES:
        raise ValueError(f"eval_batch_size stage must be one of {EVAL_STAGES}, got {stage!r}")
    comp = cfg["compute"]
    return comp.get(f"{stage}_eval_batch_size") or comp["eval_batch_size"]


# What a resumed run may set differently from the run it continues. Everything else changes the
# experiment, so a resume refuses it: the continued run must be the same run.
RESUME_FREE_BLOCKS = ("compute", "gates")
RESUME_FREE_HYPERPARAMETERS = ("training_steps", "introspection_at_steps")


def check_resume(saved, cfg, from_step):
    """Refuse a resume whose settings would change the experiment. Returns the fields that differ.

    `saved` is the settings.json the run wrote when it started, `cfg` the settings for the resume.
    compute and gates may differ freely. training_steps may only grow, and must lie beyond from_step.
    introspection_at_steps may differ only beyond from_step: the points up to and including it are
    the ones already run (or, for from_step itself, about to be finished), so they must stay as they were.
    """
    def strip(c):
        c = copy.deepcopy({k: v for k, v in c.items() if not k.startswith("_")})
        for block in RESUME_FREE_BLOCKS:
            c.pop(block, None)
        for field in RESUME_FREE_HYPERPARAMETERS:
            c.get("model_hyperparameters", {}).pop(field, None)
        return c

    def diff(a, b, path=""):
        out = []
        for key in sorted(set(a) | set(b)):
            here = f"{path}.{key}" if path else key
            if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                out += diff(a[key], b[key], here)
            elif a.get(key) != b.get(key):
                out.append(f"{here}: {a.get(key)!r} -> {b.get(key)!r}")
        return out

    # fields added to the settings after a run started are absent from its settings.json; they are read
    # as their defaults, which is what that run ran with
    saved = apply_defaults(copy.deepcopy(saved))
    changed = diff(strip(saved), strip(cfg))
    if changed:
        raise SettingsError(
            "a resume continues the same experiment, so only compute, gates, training_steps and later "
            "introspection points may change; these differ from the run's settings.json:\n  " + "\n  ".join(changed)
        )
    old, new = saved["model_hyperparameters"], cfg["model_hyperparameters"]
    if new["training_steps"] < old["training_steps"]:
        raise SettingsError(f"training_steps may only grow on a resume: {old['training_steps']} -> {new['training_steps']}")
    if new["training_steps"] <= from_step:
        raise SettingsError(f"training_steps {new['training_steps']} leaves nothing to train after step {from_step}")
    if new["training_steps"] != old["training_steps"] and decays(saved):
        raise SettingsError(
            f"training_steps cannot change on a resume under lr_schedule {old['lr_schedule']}: the learning "
            "rate at every step depends on where training ends, so the steps already taken were set for the "
            "old end. Start a new run with the longer schedule instead")
    before = lambda c: [s for s in introspection_points(c) if s <= from_step]
    if before(saved) != before(cfg):
        raise SettingsError(
            f"introspection points up to step {from_step} must stay as they ran: {before(saved)} -> {before(cfg)}"
        )
    notes = []
    for block in RESUME_FREE_BLOCKS:
        notes += diff({block: saved.get(block, {})}, {block: cfg.get(block, {})})
    for field in RESUME_FREE_HYPERPARAMETERS:
        if old.get(field) != new.get(field):
            notes.append(f"model_hyperparameters.{field}: {old.get(field)!r} -> {new.get(field)!r}")
    return notes


def merge(base, over):
    """`over` wins field by field. A block merges into a block; anything else replaces outright.

    Merging per field rather than per block is what lets a file name one hyperparameter without
    restating the other thirteen. Lists replace whole, so a range like [50, 100] behaves the way
    it reads.
    """
    out = dict(base)
    for key, value in over.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def read_settings(path, _seen=()):
    """Read a settings file, merged over whatever it extends. Returns the settings and the chain.

    A file names its parent with "extends", resolved relative to its own folder, so each file
    holds only the fields it changes and a shared default is written once. The chain comes back
    child first, and goes into `_source` so a results row still says where its settings came from.
    """
    path = Path(path)
    if not path.exists():
        with_json = path.with_suffix(".json")
        if with_json.exists():
            raise SettingsError(f"no settings file at '{path}'; did you mean '{with_json}'?")
        folder = path.parent if path.parent.is_dir() else Path("configs")
        listed = ", ".join(sorted(q.name for q in folder.glob("*.json"))) or "none"
        raise SettingsError(f"no settings file at '{path}'. Files in {folder}/: {listed}")
    here = path.resolve()
    if here in _seen:
        names = " -> ".join(p.name for p in _seen + (here,))
        raise SettingsError(f"settings files extend each other in a cycle: {names}")
    cfg = json.loads(path.read_text())
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg, [str(path)]
    base, chain = read_settings(path.parent / parent, _seen + (here,))
    return merge(base, cfg), [str(path)] + chain


def load(path=None, use_test=False):
    """Load settings from a JSON file, or the hardcoded TEST dict when use_test is true.

    Also checks that the data folder exists and is object.attribute_count wide.
    """
    if use_test:
        from src.configs.test import TEST

        cfg, source = copy.deepcopy(TEST), "src/configs/test.py:TEST"
    else:
        if path is None:
            raise SettingsError("no settings file given and use_test is false")
        cfg, chain = read_settings(path)
        source = " <- ".join(chain)
    apply_defaults(cfg)
    validate(cfg)
    from src.data import check_attribute_count

    check_attribute_count(cfg["model_training"]["data_dir"], cfg["object"]["attribute_count"])
    cfg["_source"] = source
    return cfg


def _available(kind):
    package = Path(__file__).parent / kind
    return sorted(p.stem for p in package.glob("*.py") if p.stem not in ("__init__", "base"))


def load_component(kind, name, *args):
    """Import src.<kind>.<name> and call its build(). kind is rules, estimators, reports or compute."""
    try:
        module = importlib.import_module(f"src.{kind}.{name}")
    except ModuleNotFoundError as err:
        if err.name == f"src.{kind}.{name}":
            raise SettingsError(
                f"no {kind} module named '{name}'; available: {', '.join(_available(kind))}. "
                f"Add src/{kind}/{name}.py to define it."
            ) from None
        raise
    if not hasattr(module, "build"):
        raise SettingsError(f"src/{kind}/{name}.py must define build()")
    return module.build(*args)


@dataclass
class Components:
    rule: object
    estimator: object
    schemas: list
    backend: object


def resolve(cfg):
    """Turn the names in the settings into objects. Reads decision_rule, model_estimating, report_schema, object, compute."""
    n = cfg["object"]["attribute_count"]
    rule_params = {k: v for k, v in cfg["decision_rule"].items() if k != "type"}
    rule = load_component("rules", cfg["decision_rule"]["type"], rule_params)
    estimator_params = {k: v for k, v in cfg["model_estimating"].items() if k != "type"}
    # the estimator is told how many options a trial offers, so a threshold fit knows whether both-fail means C
    estimator = load_component("estimators", cfg["model_estimating"]["type"],
                               {**estimator_params, "attribute_count": n, "option_count": rule.option_count})
    active_cuts = getattr(rule, "params", {}).get("active_cuts", 1)
    if getattr(estimator, "single_cut", False) and active_cuts != 1:
        raise SettingsError(
            f"model_estimating.type '{estimator.name}' fits exactly one screened attribute, but "
            f"decision_rule.active_cuts is {active_cuts}"
        )
    schema_params = {k: v for k, v in cfg["report_schema"].items() if k != "type"}
    names = cfg["report_schema"]["type"]
    names = [names] if isinstance(names, str) else list(names)
    names = [rule.name if name == "auto" else name for name in names]
    seen, schemas = set(), []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        schemas.append(load_component("reports", name, {**schema_params, "attribute_count": n}))
    backend = load_component("compute", cfg["compute"]["backend"], cfg)
    return Components(rule=rule, estimator=estimator, schemas=schemas, backend=backend)


RANGE_KEYS = ("start", "end", "every")


def _is_step(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _introspection_steps(at):
    """introspection_at_steps as the list of decision steps it names, refusing anything else.

    Two forms: a list of steps, [1500, 2500, 3000], or a range, {"start": 1000, "end": 3000,
    "every": 500}, which names start, start + every, ... up to and including end. "every" may be
    left out of a range to branch at start and end only.
    """
    if isinstance(at, dict):
        unknown = set(at) - set(RANGE_KEYS)
        if unknown or "start" not in at or "end" not in at:
            raise SettingsError(
                "model_hyperparameters.introspection_at_steps as a range takes "
                '{"start": <step>, "end": <step>, "every": <steps>}, "every" optional; '
                f"got {at}"
            )
        start, end, every = at["start"], at["end"], at.get("every", at["end"] - at["start"] or 1)
        if not (_is_step(start) and _is_step(end) and _is_step(every)) or every < 1 or end < start:
            raise SettingsError(
                f"model_hyperparameters.introspection_at_steps range {at} needs whole numbers with start <= end "
                "and every at least 1"
            )
        steps = list(range(start, end + 1, every))
        return steps if steps[-1] == end else steps + [end]
    if not isinstance(at, list) or not at or any(not _is_step(s) for s in at):
        raise SettingsError(
            "model_hyperparameters.introspection_at_steps must be a non-empty list of decision steps, "
            '{"start": ..., "end": ..., "every": ...}, or null'
        )
    return list(at)


def introspection_points(cfg):
    """The decision steps that introspection training branches from, in order. Reads model_hyperparameters.

    Empty when introspection_training is off. introspection_at_steps null means the final step
    only, which is the original design: stage two once, after all of stage one. A list names the
    steps; a range {"start", "end", "every"} names start, start + every, ... and always end.
    """
    mh = cfg["model_hyperparameters"]
    if not mh["introspection_training"]:
        return []
    at = mh.get("introspection_at_steps")
    return sorted(_introspection_steps(at)) if at is not None else [mh["training_steps"]]


def estimation_counts(cfg):
    """The trial counts the rule is recovered from, smallest first. Reads model_estimating.

    estimation_trials_per_persona is one number, or a list to sweep: the model makes its A/B
    decisions once, on the largest count, and the rule is recovered again from the first N of
    them for every N listed, so how many trials it takes to find the rule costs no model time.
    """
    trials = cfg["model_estimating"]["estimation_trials_per_persona"]
    return sorted(trials) if isinstance(trials, list) else [trials]


def hyperparameter_columns(cfg):
    """The settings fields written first in every results row. Reads all eight blocks."""
    mt, me, rs, mh, comp = (
        cfg["model_training"],
        cfg["model_estimating"],
        cfg["report_schema"],
        cfg["model_hyperparameters"],
        cfg["compute"],
    )
    rule_params = {k: v for k, v in cfg["decision_rule"].items() if k != "type"}
    return {
        "config_file": cfg.get("_source", ""),
        "data_dir": mt["data_dir"],
        "decision_rule": cfg["decision_rule"]["type"],
        "decision_rule_params": json.dumps(rule_params, sort_keys=True),
        "option_count": cfg["decision_rule"].get("option_count", 2),
        "estimator": me["type"],
        "attribute_count": cfg["object"]["attribute_count"],
        "instances": mt["instances"],
        "train_trials_per_persona": mt["train_trials_per_persona"],
        "val_trials_per_persona": mt["val_trials_per_persona"],
        "model_name": mh["model_name"],
        "quantization": mh["quantization"],
        "finetune": mh["finetune"],
        "lora_rank": mh["lora_rank"],
        "lora_alpha": mh["lora_alpha"],
        "learning_rate": mh["learning_rate"],
        "batch_size": mh["batch_size"],
        "training_steps": mh["training_steps"],
        "introspection_training": mh["introspection_training"],
        "introspection_steps": mh["introspection_steps"],
        "introspection_at_steps": json.dumps(introspection_points(cfg)),
        "gradient_checkpointing": mh["gradient_checkpointing"],
        "micro_batch_size": mh["micro_batch_size"],
        "lr_schedule": "" if mh.get("lr_schedule") is None else json.dumps(mh["lr_schedule"], sort_keys=True),
        "introspection_targets": json.dumps(mh.get("introspection_targets", ["hidden"])),
        "seed": mh["seed"],
        "decision_temperature": me["decision_temperature"],
        "samples_per_trial": me["samples_per_trial"],
        # a single count as it was always written; a sweep as its list, smallest first
        "estimation_trials_per_persona": (json.dumps(estimation_counts(cfg)) if isinstance(me["estimation_trials_per_persona"], list)
                                          else me["estimation_trials_per_persona"]),
        "chance_draws": me["chance_draws"],
        "report_temperature": rs["report_temperature"],
        "samples_per_report": rs["samples_per_report"],
        "compute_backend": comp["backend"],
        # the decision step a resumed run continued from; blank for a run that ran straight through
        "resumed_from_step": cfg.get("_resumed_from_step", ""),
    }
