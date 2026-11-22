"""Every one-persona experiment, sized for every card: 16 GB, 24 GB, 40 GB and 80 GB.

    python scripts/make_card_files.py            # write the missing card files, bring the 24 GB files up to date
    python scripts/make_card_files.py --check    # change nothing; say what would be written

Each experiment in configs/minimum_threshold_one_persona/ is one settings file per card, and the files of one
experiment differ only in how they are sized for the card (src.settings_files.GPU_PROFILES):

    <name>.json        16 GB   micro-batch 2,  gradient checkpointing on,  eval batches 4
    <name>_24gb.json   24 GB   micro-batch 2,  gradient checkpointing off, eval batches 48
    <name>_40gb.json   40 GB   micro-batch 5,  gradient checkpointing off, eval batches 96
    <name>_80gb.json   80 GB   micro-batch 10, gradient checkpointing off, eval batches 192

The optimizer's batch (model_hyperparameters.batch_size, 10), the adapter, the quantization, the data and the
questions are the same in all four, so runs on different cards compare. A larger card passes the batch through the
model in fewer pieces and evaluates more prompts at once; it does not train on a larger batch.

A card file that is missing is written from one the experiment has (the 40 GB file if there is one, else the 24 GB,
else the 16 GB), with the five sizing fields and the two comment lines that name the card and the file changed.
A 16 GB or 40 GB file that exists is left as it is. Every 24 GB file is brought to the current 24 GB sizes.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

TREE = ROOT / "configs" / "minimum_threshold_one_persona"
SUFFIX = {"16gb": "", "24gb": "_24gb", "40gb": "_40gb", "80gb": "_80gb"}
CARD = re.compile(r"_(24gb|40gb|80gb)$")
PREFERRED = ("40gb", "24gb", "80gb", "16gb")    # which existing file a missing one is written from
REFRESH = ("24gb",)                             # cards whose existing files are brought to the current profile


def experiments():
    """(folder, name) -> {card: path} for every settings file in the tree."""
    found = {}
    for path in sorted(TREE.rglob("*.json")):
        m = CARD.search(path.stem)
        found.setdefault((path.parent, CARD.sub("", path.stem)), {})[m.group(1) if m else "16gb"] = path
    return found


def sized(body, card, path):
    """`body` (a settings file as read) sized for `card`, its comment naming that card and `path`."""
    p = SF.GPU_PROFILES[("8b", card)]
    out = json.loads(json.dumps(body))
    mh, comp = out["model_hyperparameters"], out["compute"]
    if mh["model_name"] != SF.MODELS["8b"]:
        raise ValueError(f"{path}: {mh['model_name']} is not {SF.MODELS['8b']}")
    mh["micro_batch_size"] = p["micro_batch_size"]
    mh["gradient_checkpointing"] = p.get("gradient_checkpointing", True)    # on unless the card's profile turns it off
    for key in ("eval_batch_size", "decision_eval_batch_size", "introspection_eval_batch_size"):
        comp[key] = p[key]
    rel = path.relative_to(ROOT).as_posix()
    lines = {"Sized for ": SF.gpu_line("8b", card), "Run: ": f"Run: python scripts/run_one_persona.py --settings {rel}"}
    comment = list(out["_comment"])
    for start, line in lines.items():
        at = [i for i, text in enumerate(comment) if text.startswith(start)]
        if len(at) != 1:
            raise ValueError(f"{path}: {len(at)} comment lines start with {start!r}")
        comment[at[0]] = line
    out["_comment"] = comment
    return (json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(check=False):
    written, refreshed, kept = [], [], 0
    for (folder, name), have in experiments().items():
        source = json.loads(have[next(c for c in PREFERRED if c in have)].read_bytes())
        for card, suffix in SUFFIX.items():
            path = folder / f"{name}{suffix}.json"
            if path.exists() and card not in REFRESH:
                kept += 1
                continue
            data = sized(json.loads(path.read_bytes()) if path.exists() else source, card, path)
            if path.exists() and path.read_bytes() == data:
                kept += 1
                continue
            (refreshed if path.exists() else written).append(path)
            if not check:
                path.write_bytes(data)
    verb = "would be " if check else ""
    print(f"{len(written)} card files {verb}written, {len(refreshed)} 24 GB files {verb}brought up to date, {kept} left as they are")
    by_card = {card: sum(p.stem.endswith(s) if s else not CARD.search(p.stem) for p in written) for card, s in SUFFIX.items()}
    print("  new files by card: " + ", ".join(f"{card} {n}" for card, n in by_card.items()))
    return written, refreshed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    build(ap.parse_args().check)


if __name__ == "__main__":
    main()
