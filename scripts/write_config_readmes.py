"""Write a README.md into every configs/ folder that holds settings files, saying what the folder is for and what
each experiment in it is for, with the card sizes available.

    python scripts/write_config_readmes.py

The folder purposes are written below (PURPOSE); each experiment's purpose is the first line of its file's `_comment`
(the line after "Notes, written when..." for older files). Size twins (`_24gb`, `_40gb`, `_14b_80gb`, ...) are one row.
The hand-written READMEs one level up (configs/README.md, configs/minimum_threshold_one_persona/README.md, its cut_sweep/
README and the 14B and 32B ones) are left alone. Rerun after adding or changing a settings file.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

CONFIGS = ROOT / "configs"
HANDWRITTEN = {CONFIGS, CONFIGS / "minimum_threshold_one_persona", CONFIGS / "minimum_threshold_one_persona" / "cut_sweep",
               CONFIGS / "minimum_threshold_one_persona_qwen3_14b", CONFIGS / "minimum_threshold_one_persona_qwen3_32b"}
SIZE = re.compile(r"_(?:(?:14b|32b)_)?(?:16gb|24gb|40gb|48gb|80gb)$")

# purpose of each one-persona folder, keyed by <group>/<persona>; the 14B and 32B trees reuse them
PERSONA_DIRS = {
    "base/persona_a": "Persona A, the original one persona: loose_leaf_tea (\"Imagine you are Djedefre\"), weights caffeine 0, "
                      "leaf -83, oxidation -38, steeping -100, resteep 31, caffeine screened at 35 mg (50%). Its runs, from the "
                      "first 10000-step run to the 3000-step run that asks every report question.",
    "base/persona_b": "Persona B, the cross-persona control for A: the same prompts with weights 0, 100, 20, -50, -70 (summing "
                      "to 0, cosine -0.34 with A's). If B's reports follow B's weights better than A's, they carry what "
                      "training put there, not attribute-name priors.",
    "base/persona_c": "Persona C: two weights tied for largest at opposite signs (leaf +100, steeping -100; oxidation -30, "
                      "resteep +30). Which of the two is reported positive, if either? A sizes-only reply scores cos4 = 0 here.",
    "base/persona_d": "Persona D: four weights close in size (15, -10, 20, -15). Are the signs and the order reported right? The size "
                      "itself is not in the choices: the same answers follow from 75, -50, 100, -75.",
    "base/persona_e": "Persona E-positive: every weight positive (leaf 100 > steeping 70 > oxidation 40 > resteep 25), close to "
                      "the untrained model's own reply, so the readout is the order (Spearman), not the cosine.",
    "base/persona_f": "Persona F: the largest weight on an attribute that has never carried it. F-oxidation (-100, predicted "
                      "reported positive: wrong) and F-resteep (+100, predicted positive: right) pair as a sign test.",
    "base/persona_g": "Persona G: the screen on steeping_time (attribute 4) at 2.5 minutes (25%) instead of caffeine, A's four weight "
                      "values on the other slots. Is \"dimension 4\" named for the right reason, and does the cut probe switch "
                      "between 2 and 3 minutes on steeping?",
    "base/persona_h": "Persona H: B's weights in a new domain, cutting_board (\"Imagine you are Antonín Dvořák\"), "
                      "board_thickness screened at 2.0 cm (25%). A full replication of B with scenario, persona, names, units and "
                      "ranges all changed: if H behaves like B, the tea results generalise.",
    "inverse/persona_a": "A-inverse: A's weights with every sign flipped (0, 83, 38, 100, -31), the screen kept. The same 5000 "
                         "prompts as A; only the 3750 weight-decided answers change.",
    "inverse/persona_b": "B-inverse: B's weights with every sign flipped (0, -100, -20, 50, 70), the screen kept.",
    "inverse/persona_c": "C-inverse: C with every sign flipped (leaf -100, steeping +100). With C, tells a rule that picks an "
                         "attribute by name from one that tracks sign.",
    "inverse/persona_d": "D-inverse: D with every sign flipped (-15, 10, -20, 15).",
    "inverse/persona_e": "E-negative: every weight negative (-100, -40, -70, -25), the clean sign test: the untrained model writes "
                         "negative numbers only about 25% of the time. Readout: the share of reported numbers that are negative.",
    "controls/persona_a": "Controls for persona A: the plain linear rule with A's weights and no screen, and A's prompts with the "
                          "answers shuffled (no rule to learn; the floor for every readout).",
    "rule_tests/persona_a": "Persona A with one part of the rule changed (the cut level, the screened attribute, or the screened "
                            "attribute also weighted), to test whether the reports move with the true rule.",
    "cut_sweep/persona_a": "Persona A's cut sweeps: the mix of training trials swept from 75% decided by the weights (w75_c25) "
                           "to none (w00_c100). `cut_sweep_wXX` branch from A's first run 20261106-152354 at step 1500 (all "
                           "nine ran); `cut_sweep_a_wXX` branch from A's 3000-step run 20261110-211307 at step 1500 and ask "
                           "every question, matched with the B, A-inverse and B-inverse sweeps. See ../README.md.",
    "cut_sweep/persona_b": "Persona B's cut sweep, matched with A's, A-inverse's and B-inverse's: the mix of training trials "
                           "swept from 75% decided by the weights to none, branching from B's run 20261110-112915 at step "
                           "1500. See ../README.md.",
    "cut_sweep/persona_a_inverse": "Persona A-inverse's cut sweep, branching from 20261113-170347 at step 1500. See ../README.md.",
    "cut_sweep/persona_b_inverse": "Persona B-inverse's cut sweep, branching from 20261112-235923 at step 1500. See ../README.md.",
    "cut_sweep/persona_c": "Persona C's cut sweep (cut at 25% of range), branching from 20261116-214255 at step 1500. "
                         "See ../README.md.",
    "cut_sweep/persona_c_inverse": "Persona C-inverse's cut sweep (cut at 25% of range), branching from 20261116-220846 at step 1500. "
                         "See ../README.md.",
    "cut_sweep/persona_e_positive": "Persona E-positive's cut sweep (cut at 25% of range), branching from 20261116-221042 at step 1500. "
                         "See ../README.md.",
    "cut_sweep/persona_e_negative": "Persona E-negative's cut sweep (cut at 25% of range), branching from 20261116-221120 at step 1500. "
                         "See ../README.md.",
}
OTHER_DIRS = {
    "minimum_threshold": "The minimum threshold rule over many personas (100), answers A, B or C, on Qwen3-8B: the multi-persona "
                         "experiment the one-persona work grew out of.",
    "one_persona_maximum_threshold": "One persona (A's weights) under a maximum threshold instead of a minimum: an option with "
                                     "MORE than 35 mg of caffeine is rejected.",
    "paper_minimum_threshold": "The paper's minimum threshold design over 100 personas: the 8B run, its 16 GB versions (with "
                               "revised questions, a resume and an extension), a 0.6B version and its smoke tests, and the "
                               "2500-step run with twice the data.",
    "plunkett_linear": "Plunkett's linear task (no threshold): the Atkinson variant and the 4B and 8B versions sized for 20 GB.",
}
TREES = {"minimum_threshold_one_persona": "Qwen3-8B",
         "minimum_threshold_one_persona_qwen3_14b": "Qwen3-14B",
         "minimum_threshold_one_persona_qwen3_32b": "Qwen3-32B"}


def purpose_of(raw):
    """The comment's first sentence about the experiment, its wrapped lines joined."""
    lines = [l.strip() for l in raw.get("_comment", []) if l.strip()]
    lines = [l for l in lines if not l.startswith(("Every setting", "Sized for", "Run:"))]
    start = next((i + 1 for i, l in enumerate(lines) if l.startswith("Notes, written when")), 0)
    text = ""
    for l in lines[start:start + 4]:
        text = f"{text} {l}".strip()
        if re.search(r"[.:]$", l) or re.search(r"\. [A-Z]", l):
            break
    m = re.match(r"(.+?\.)(?:\s+[A-Z(]|$)", text)
    return (m.group(1) if m else text).strip()


def card_of(raw):
    mh, comp = raw["model_hyperparameters"], raw["compute"]
    for (model, gpu), p in SF.GPU_PROFILES.items():
        if SF.MODELS[model] == mh["model_name"] and mh["micro_batch_size"] == p["micro_batch_size"] and \
                comp.get("decision_eval_batch_size") == p["decision_eval_batch_size"]:
            return f"{gpu[:-2]} GB"
    micro = mh.get("micro_batch_size") or mh.get("batch_size")
    return f"micro-batch {micro}, eval batch {comp.get('decision_eval_batch_size') or comp.get('eval_batch_size')}"


def folder_purpose(d):
    rel = d.relative_to(CONFIGS).as_posix()
    tree, _, rest = rel.partition("/")
    if tree in TREES and rest in PERSONA_DIRS:
        return f"On **{TREES[tree]}**. " + PERSONA_DIRS[rest]
    return OTHER_DIRS.get(rel, "")


def readme(d):
    files = sorted(d.glob("*.json"))
    one_persona = d.relative_to(CONFIGS).parts[0] in TREES
    groups = {}
    for f in files:   # size twins share a row in the one-persona trees; elsewhere a _16gb file is its own experiment
        groups.setdefault(SIZE.sub("", f.stem) if one_persona else f.stem, []).append(f)
    rel = d.relative_to(ROOT).as_posix()
    lines = [f"# {rel}", "", folder_purpose(d), "",
             "| experiment | what it is for | model | files (card) | steps |", "|---|---|---|---|---|"]
    for stem, fs in groups.items():
        raws = [json.loads(f.read_text(encoding="utf-8")) for f in fs]
        mh = raws[0]["model_hyperparameters"]
        purpose = purpose_of(raws[0]).replace("|", "/")
        sizes = ", ".join(f"`{f.name}` ({card_of(r)})" for f, r in zip(fs, raws))
        lines.append(f"| `{stem}` | {purpose} | {mh['model_name'].split('/')[-1]} | {sizes} | {mh['training_steps']} |")
    lines += ["", "Every file is complete (nothing inherited): open it to see or edit any setting."
              + (" The files of one experiment differ only in how they are sized for the card "
                 "(`model_hyperparameters.micro_batch_size`, the `compute` eval batch sizes, and gradient checkpointing, "
                 "off in the 40 GB files); the effective batch, LoRA, quantization and everything else match."
                 if one_persona else ""), "",
              "```", f"python scripts/run_one_persona.py --settings {rel}/<file>.json", "```", "",
              "Written by `scripts/write_config_readmes.py`; rerun it after adding a file.", ""]
    return "\n".join(lines)


def main():
    dirs = sorted({p.parent for p in CONFIGS.rglob("*.json")} - HANDWRITTEN)
    missing = [d for d in dirs if not folder_purpose(d)]
    if missing:
        sys.exit("no purpose written for: " + ", ".join(d.relative_to(ROOT).as_posix() for d in missing))
    for d in dirs:
        (d / "README.md").write_bytes(readme(d).encode("utf-8"))
    print(f"{len(dirs)} READMEs written")


if __name__ == "__main__":
    main()
