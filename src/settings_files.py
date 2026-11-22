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

# (model, card): micro-batch, eval batch sizes, and why. All keep 4-bit and bf16, and gradient checkpointing on
# unless the profile sets gradient_checkpointing (8B on 40 GB turns it off: the memory is there, and it costs time).
# `cards` names cards of that size; `why` says why these sizes, then what to lower first if memory runs out.
GPU_PROFILES = {
    ("8b", "16gb"): dict(micro_batch_size=2, eval_batch_size=4, decision_eval_batch_size=4, introspection_eval_batch_size=1,
                         cards="RTX 5070 Ti, RTX 4080",
                         why="Qwen3-8B in 4-bit is about 6 GB, which leaves room for the 152k-token logits. Out of memory: "
                             "micro_batch_size 1, then report_schema.max_new_tokens 200"),
    ("8b", "24gb"): dict(micro_batch_size=5, gradient_checkpointing=False, eval_batch_size=48, decision_eval_batch_size=48, introspection_eval_batch_size=4,
                         cards="L4, A10, RTX 4090, RTX 3090",
                         why="Gradient checkpointing off and eval batches of 48 since 2026-11-22 (they were on and 16): the 40 GB "
                             "sizes scaled by the memory left after the model, not measured on a 24 GB card. Out of memory in "
                             "training: gradient_checkpointing true. Out of memory in evaluation: eval batches 24, then 16"),
    ("8b", "40gb"): dict(micro_batch_size=5, gradient_checkpointing=False, eval_batch_size=96, decision_eval_batch_size=96, introspection_eval_batch_size=8,
                         cards="A100 40GB",
                         why="Gradient checkpointing off: it recomputes every layer's activations in the backward pass, "
                             "about 30% of training time, and only saves memory. The batch of 10 in two passes of 5, whose "
                             "activations should fit without it (same effective batch, LoRA and 4-bit as the 16 and 24 GB "
                             "files, so runs compare). Out of memory in training: gradient_checkpointing true (then "
                             "micro_batch_size 10 fits in one pass). Out of memory in evaluation: eval batches 48, then 32"),
    ("8b", "80gb"): dict(micro_batch_size=10, gradient_checkpointing=False, eval_batch_size=192, decision_eval_batch_size=192, introspection_eval_batch_size=16,
                         cards="A100 80GB, H100",
                         why="The 40 GB sizes doubled: the whole batch of 10 in one pass, gradient checkpointing off, eval batches "
                             "of 192 (same effective batch, LoRA and 4-bit as the other cards' files, so runs compare). Not "
                             "measured on an 80 GB card. Out of memory in training: micro_batch_size 5. Out of memory in "
                             "evaluation: eval batches 96"),
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
    if "gradient_checkpointing" in p:
        mh["gradient_checkpointing"] = p["gradient_checkpointing"]
    for key in ("eval_batch_size", "decision_eval_batch_size", "introspection_eval_batch_size"):
        comp[key] = p[key]
    return out, p["why"]


def gpu_line(model, gpu):
    """The comment line that says which card a file is sized for, and what to lower if memory runs out."""
    p = GPU_PROFILES[(model, gpu)]
    extra = "" if "gradient_checkpointing" not in p else f"gradient_checkpointing {str(p['gradient_checkpointing']).lower()}, "
    return (f"Sized for {MODELS[model]} on a {gpu[:-2]} GB card ({p['cards']}): micro_batch_size {p['micro_batch_size']}, {extra}"
            f"eval batches {p['eval_batch_size']} (decision {p['decision_eval_batch_size']}, introspection "
            f"{p['introspection_eval_batch_size']}). {p['why']}.")
