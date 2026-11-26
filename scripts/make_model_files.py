"""Persona A and A-inverse on the three smaller Qwen3 models: 0.6B, 1.7B and 4B.

    python scripts/make_model_files.py            # write the settings files and each tree's README
    python scripts/make_model_files.py --check    # change nothing; say what would be written

    configs/minimum_threshold_one_persona_qwen3_<model>/base/persona_a/one_persona_minimum_threshold_a_<model>_<card>.json
    configs/minimum_threshold_one_persona_qwen3_<model>/inverse/persona_a/one_persona_a_inverse_<model>_<card>.json

with <model> 0.6b, 1.7b or 4b and <card> 24gb, 40gb or 80gb. Qwen3 has no 0.5B or 2B model: 0.6B and 1.7B are the
sizes nearest those.

Each file is its Qwen3-8B namesake (the file with no card suffix) with three things changed:
  - the model
  - the card's sizes (src.settings_files.GPU_PROFILES): micro_batch_size, gradient checkpointing, the eval batches
  - baseline_before_training true: the untrained model is scored first as step 0, since none of these models has a
    base-model score for these prompts (as the 14B and 32B files; the 8B files skip it since 2026-11-23)
The persona, data folder, trials, seeds, 4-bit quantization, LoRA, learning rate, batch of 10, 3000 training steps
and the report questions are the 8B file's, so a run differs from the 8B run in the model alone.

To add a model or an experiment, add it to MODELS or EXPERIMENTS below (a new model also needs its sizes in
GPU_PROFILES), run this, then scripts/write_config_readmes.py.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

SOURCE = ROOT / "configs" / "minimum_threshold_one_persona"
MODELS = ("0.6b", "1.7b", "4b")
CARDS = ("24gb", "40gb", "80gb")
# <group>/<persona folder>: the experiment's file name on Qwen3-8B, without a card suffix
EXPERIMENTS = {"base/persona_a": "one_persona_minimum_threshold_a",
               "inverse/persona_a": "one_persona_a_inverse"}
DATA_LINE = "The data folder is the Qwen3-8B experiment's: trials and answers do not depend on the model."
WEIGHTS_GB = {"0.6b": "under 1 GB", "1.7b": "about 1.5 GB", "4b": "about 3.5 GB"}      # in 4-bit, a rough figure


def tree(model):
    return ROOT / "configs" / f"minimum_threshold_one_persona_qwen3_{model}"


def target(model, folder, name, card):
    return tree(model) / folder / f"{name}_{model}_{card}.json"


def rel(path):
    return path.relative_to(ROOT).as_posix()


def one_line(comment, start):
    at = [i for i, text in enumerate(comment) if text.startswith(start)]
    if len(at) != 1:
        raise ValueError(f"{len(at)} comment lines start with {start!r}")
    return at[0]


def sized(source, model, card, path):
    """(settings, comment) for `source` (an 8B settings file as read) on `model`, sized for `card`."""
    name = SF.MODELS[model]
    short = name.split("/")[-1]
    if source["model_hyperparameters"]["model_name"] != SF.MODELS["8b"]:
        raise ValueError(f"{path}: written from a file that is not {SF.MODELS['8b']}")
    cfg, _ = SF.with_gpu({k: v for k, v in source.items() if k != "_comment"}, model, card)
    cfg["model_hyperparameters"]["baseline_before_training"] = True
    comment = list(source["_comment"])
    comment[0] = f"{comment[0]} On {name}."
    comment[one_line(comment, "Sized for ")] = SF.gpu_line(model, card)
    comment[one_line(comment, "Run: ")] = f"Run: python scripts/run_one_persona.py --settings {rel(path)}"
    at = one_line(comment, "Every setting is in this file")
    comment[at:at] = [
        DATA_LINE,
        f"The untrained model is scored first, as step 0 (baseline_before_training true): {short} has no base-model "
        f"score for these prompts. Where a note below names Qwen3-8B or a 16 GB card, or says the untrained model is "
        f"not scored first, it is the Qwen3-8B file this one was written from: {rel(path.parent).split('/', 2)[2]}/ in "
        f"configs/minimum_threshold_one_persona/.",
    ]
    return cfg, comment


def body(cfg, comment):
    """The bytes SF.write puts in the file."""
    return (json.dumps({"_comment": comment, **SF.ordered(cfg)}, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def tree_readme(model):
    name = SF.MODELS[model]
    folder = f"configs/minimum_threshold_one_persona_qwen3_{model}"
    lines = [
        f"# {folder}", "",
        f"Persona A and A-inverse of `../minimum_threshold_one_persona/` on **{name}**. Each file is its Qwen3-8B namesake",
        "with only the model, the batch sizes and `baseline_before_training` changed: the same persona, data folder, trials,",
        "seeds, 4-bit quantization, learning rate, LoRA rank, **3000 training steps** (12 checkpoints, one every 250) and the",
        "same report questions. The untrained model is scored first (step 0, `base_model` rows), since this model has no",
        "base-model score for these prompts. The 8B README describes both experiments and the persona.", "",
        "```", f"python scripts/run_one_persona.py --settings {folder}/<group>/persona_a/<file>.json", "```", "",
        "## Cards", "",
        f"{name} in 4-bit is {WEIGHTS_GB[model]} of weights, so the card is filled by the batch, not the model. bf16 needs an "
        "Ampere or newer card.", "",
        "| file ends in | card | micro_batch_size | gradient checkpointing | eval batches |", "|---|---|---|---|---|",
    ]
    for card in CARDS:
        p = SF.GPU_PROFILES[(model, card)]
        lines.append(f"| `_{model}_{card}.json` | {card[:-2]} GB ({p['cards']}) | {p['micro_batch_size']} | "
                     f"{'on' if p.get('gradient_checkpointing', True) else 'off'} | {p['eval_batch_size']} |")
    lines += [
        "", "Every file is complete (nothing inherited), so resize a run by editing `model_hyperparameters.micro_batch_size`",
        "and the `compute` eval batch sizes in the file itself. These sizes are worked out from the Qwen3-8B runs",
        f"(`src/settings_files.py`), not measured on {name}: the first training steps and the first checkpoint's evaluation",
        "show whether they fit. Out of memory in training: halve `micro_batch_size`, then `gradient_checkpointing` true. Out",
        "of memory in evaluation: halve the eval batches.", "",
        "## Layout", "", "```",
    ]
    lines += [f"{folder_ + '/':<22}{name_}" for folder_, name_ in EXPERIMENTS.items()]
    lines += [
        "```", "",
        "Data comes from `data/minimum_threshold_one_persona/`, the 8B experiments' folders: the trials and answers do not",
        "depend on the model.", "",
        f"Run ids carry the model (`..._{name.split('/')[-1]}_<tag>`), so results never mix with another model's.", "",
        "Written by `scripts/make_model_files.py` (this page and the settings files); the per-folder pages by",
        "`scripts/write_config_readmes.py`.", "",
    ]
    return "\n".join(lines).encode("utf-8")


def build(check=False):
    wanted = {}
    for model in MODELS:
        for folder, name in EXPERIMENTS.items():
            source = json.loads((SOURCE / folder / f"{name}.json").read_text(encoding="utf-8"))
            for card in CARDS:
                path = target(model, folder, name, card)
                wanted[path] = body(*sized(source, model, card, path))
        wanted[tree(model) / "README.md"] = tree_readme(model)
    changed = [p for p, data in wanted.items() if not p.exists() or p.read_bytes() != data]
    if not check:
        for path in changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wanted[path])
    print(f"{len(changed)} of {len(wanted)} files {'would be ' if check else ''}written")
    for path in changed:
        print(f"  {rel(path)}")
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    build(ap.parse_args().check)


if __name__ == "__main__":
    main()
