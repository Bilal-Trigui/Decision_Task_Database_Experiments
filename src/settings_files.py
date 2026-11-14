"""Writing settings files: every file holds every setting, nothing inherited.

A settings file under configs/ is complete on its own: open it and every value the run uses is there,
so any of them can be edited in that one file. (read_settings still follows "extends" for an older or
hand-written file, but no file in configs/ uses it.)

    flat(path)                       the full settings a file stands for, every field filled in
    write(path, cfg, comment)        write settings as one complete file, blocks in default.json's order
    with_gpu(cfg, model, gpu)        the same experiment sized for another model and card (GPU_PROFILES)

GPU sizing lives in each file's compute block and model_hyperparameters.micro_batch_size; GPU_PROFILES
below is only what the scripts write into a new file. Change a file's values directly to resize it.
"""
import copy
import json
from pathlib import Path

from src.config import apply_defaults, read_settings

BLOCK_ORDER = ["object", "decision_rule", "model_training", "model_estimating", "report_schema",
               "model_hyperparameters", "gates", "compute"]

MODELS = {"8b": "Qwen/Qwen3-8B", "14b": "Qwen/Qwen3-14B", "32b": "Qwen/Qwen3-32B"}

# (model, card): micro-batch, eval batch sizes, and why. All keep 4-bit, bf16, gradient checkpointing on.
# `cards` names cards of that size; `why` says why these sizes, then what to lower first if memory runs out.
GPU_PROFILES = {
    ("8b", "16gb"): dict(micro_batch_size=2, eval_batch_size=4, decision_eval_batch_size=4, introspection_eval_batch_size=1,
                         cards="RTX 5070 Ti, RTX 4080",
                         why="Qwen3-8B in 4-bit is about 6 GB, which leaves room for the 152k-token logits. Out of memory: "
                             "micro_batch_size 1, then report_schema.max_new_tokens 200"),
    ("8b", "24gb"): dict(micro_batch_size=5, eval_batch_size=16, decision_eval_batch_size=16, introspection_eval_batch_size=4,
                         cards="L4, A10, RTX 4090, RTX 3090",
                         why="Out of memory: eval batches 8 first, then micro_batch_size 2"),
    ("14b", "24gb"): dict(micro_batch_size=2, eval_batch_size=4, decision_eval_batch_size=4, introspection_eval_batch_size=1,
                          cards="L4, A10, RTX 4090, RTX 3090",
                          why="Qwen3-14B in 4-bit is about 10 GB, as tight on 24 GB as 8B is on 16 GB. Out of memory: "
                              "micro_batch_size 1, then report_schema.max_new_tokens 200"),
    ("14b", "80gb"): dict(micro_batch_size=10, eval_batch_size=32, decision_eval_batch_size=32, introspection_eval_batch_size=8,
                          cards="A100 80GB, H100",
                          why="The whole batch of 10 in one pass. Out of memory: eval batches 16"),
    ("32b", "48gb"): dict(micro_batch_size=2, eval_batch_size=4, decision_eval_batch_size=4, introspection_eval_batch_size=1,
                          cards="A6000, L40S, A40",
                          why="Qwen3-32B in 4-bit is about 19 GB; it does not fit a 24 GB card. Out of memory: "
                              "micro_batch_size 1"),
    ("32b", "80gb"): dict(micro_batch_size=5, eval_batch_size=16, decision_eval_batch_size=16, introspection_eval_batch_size=4,
                          cards="A100 80GB, H100",
                          why="Out of memory: eval batches 8"),
}


def flat(path):
    """Every setting `path` stands for (its extends chain merged, defaults filled), without its comment."""
    cfg, _ = read_settings(path)
    cfg.pop("_comment", None)
    return apply_defaults(cfg)


def ordered(cfg):
    """Blocks in default.json's order, anything else after them; fields keep their order."""
    out = {k: cfg[k] for k in BLOCK_ORDER if k in cfg}
    out.update({k: v for k, v in cfg.items() if k not in out})
    return out


def write(path, cfg, comment):
    """Write `cfg` as one complete settings file (LF line endings), `comment` (a list of lines) first."""
    if "extends" in cfg:
        raise ValueError("a settings file in configs/ holds every setting itself: no extends")
    body = {"_comment": list(comment), **ordered({k: v for k, v in cfg.items() if k != "_comment"})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(body, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def with_gpu(cfg, model, gpu):
    """A copy of `cfg` on MODELS[model], with GPU_PROFILES[(model, gpu)]'s batch sizes. Returns (cfg, why)."""
    p = GPU_PROFILES[(model, gpu)]
    out = copy.deepcopy(cfg)
    mh, comp = out["model_hyperparameters"], out["compute"]
    mh["model_name"] = MODELS[model]
    mh["micro_batch_size"] = p["micro_batch_size"]
    for key in ("eval_batch_size", "decision_eval_batch_size", "introspection_eval_batch_size"):
        comp[key] = p[key]
    return out, p["why"]


def gpu_line(model, gpu):
    """The comment line that says which card a file is sized for, and what to lower if memory runs out."""
    p = GPU_PROFILES[(model, gpu)]
    return (f"Sized for {MODELS[model]} on a {gpu[:-2]} GB card ({p['cards']}): micro_batch_size {p['micro_batch_size']}, "
            f"eval batches {p['eval_batch_size']} (decision {p['decision_eval_batch_size']}, introspection "
            f"{p['introspection_eval_batch_size']}). {p['why']}.")
