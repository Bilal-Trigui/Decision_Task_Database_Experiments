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

BACKENDS = ("local", "colab", "azure", "api")
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
        "estimation_trials_per_persona": int,
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


class SettingsError(ValueError):
    """A settings file that the pipeline refuses to run."""


def apply_defaults(cfg):
    """Fill the few fields that may be omitted. Reads model_training.seeds, compute."""
    cfg.setdefault("model_training", {}).setdefault("seeds", {})
    for key, value in DEFAULT_SEEDS.items():
        cfg["model_training"]["seeds"].setdefault(key, value)
    compute = cfg.setdefault("compute", {})
    compute.setdefault("backend", "colab")
    compute.setdefault("device", "cpu" if compute["backend"] == "local" else "cuda")
    compute.setdefault("mixed_precision", "no" if compute["device"] == "cpu" else "fp16")
    compute.setdefault("eval_batch_size", 16)
    compute.setdefault("mount_drive", False)
    cfg.setdefault("report_schema", {}).setdefault("max_new_tokens", 200)
    cfg.setdefault("model_estimating", {}).setdefault("chance_draws", 1000)
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
        ("model_estimating", "estimation_trials_per_persona"),
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
    if mh["finetune"] == "full" and mh["quantization"] != "none":
        raise SettingsError("model_hyperparameters.finetune 'full' needs quantization 'none'; a quantized base cannot be fully trained")
    for field in ("min_decision_accuracy", "min_parse_rate"):
        if not 0 <= gates[field] <= 1:
            raise SettingsError(f"gates.{field} must be between 0 and 1")
    if comp["backend"] not in BACKENDS:
        raise SettingsError(f"compute.backend must be one of {BACKENDS}")
    if comp["device"] not in DEVICES:
        raise SettingsError(f"compute.device must be one of {DEVICES}")
    if comp["mixed_precision"] not in PRECISION:
        raise SettingsError(f"compute.mixed_precision must be one of {PRECISION}")
    return cfg


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
        cfg, source = json.loads(Path(path).read_text()), str(path)
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
    estimator = load_component("estimators", cfg["model_estimating"]["type"], {"attribute_count": n})
    names = cfg["report_schema"]["type"]
    names = [names] if isinstance(names, str) else list(names)
    names = [rule.name if name == "auto" else name for name in names]
    seen, schemas = set(), []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        schemas.append(load_component("reports", name, {"attribute_count": n}))
    backend = load_component("compute", cfg["compute"]["backend"], cfg)
    return Components(rule=rule, estimator=estimator, schemas=schemas, backend=backend)


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
        "seed": mh["seed"],
        "decision_temperature": me["decision_temperature"],
        "samples_per_trial": me["samples_per_trial"],
        "estimation_trials_per_persona": me["estimation_trials_per_persona"],
        "chance_draws": me["chance_draws"],
        "report_temperature": rs["report_temperature"],
        "samples_per_report": rs["samples_per_report"],
        "compute_backend": comp["backend"],
    }
