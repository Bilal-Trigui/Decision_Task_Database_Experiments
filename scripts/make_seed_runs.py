"""Seed repeats: A, A-inverse, B, E-positive and E-negative, each trained again with new seeds, asking the nine report
questions of A's 3000-step run (base/persona_a/one_persona_no_trial_a.json), on an A100 40GB.

    python scripts/make_seed_runs.py                 # seeds 5 and 6
    python scripts/make_seed_runs.py --seeds 5 6 7   # writes these; files of other seeds are left as they are

Writes configs/minimum_threshold_one_persona/seeds/persona_<x>/one_persona_<x>_nine_q_seed<N>_40gb.json.

Each file is its persona's settings with three things set:
  - report_schema: no_trial_a's, the nine questions (main, screen_index, screen_cut_units, main_report_units,
    main_threshold_met, main_threshold_not_met, screen_index_no_trial, screen_cut_units_no_trial,
    screen_cut_percent_no_trial), no reply prefix, C not shown in report trials
  - model_hyperparameters.seed: the new seed, and run_tag "<persona>-nine-q-seed<N>"
  - model_hyperparameters.baseline_before_training false: no evaluation of the untrained base model; training starts
    at step 0 from a fresh adapter as before, and the first evaluation is the step-250 checkpoint. The base model's
    replies (temperature 0, the same 50 trials) do not depend on the training seed, and each persona's seed-4 run
    already holds them (base_model_step-0). The pipeline puts the random state back after that evaluation, so
    training is the same with or without it.
  - the 40 GB batch sizes (src/settings_files.py GPU_PROFILES)
Everything else (data folder, learning rate, clipping, 3000 steps, a checkpoint every 250, the data seeds in
model_training.seeds) is the persona's own, so every seed trains on the same 5000 trials and is asked the same 50
report trials. model_hyperparameters.seed sets the LoRA starting point, the training order and the evaluation draws.

The seed-4 runs: A 20261110-211307 and B 20261110-112915 asked these nine questions; A-inverse 20261113-170347, E-positive 20261116-221042
and E-negative 20261116-221120 asked every question with the "{" prefix and C shown, so for those three the nine-
question seeds are compared with each other, not with seed 4.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
OUT = CONFIGS / "seeds"
QUESTIONS_FROM = CONFIGS / "base" / "persona_a" / "one_persona_no_trial_a.json"
# persona: (folder, settings file its other settings come from, what it is, seed-4 run)
PERSONAS = {
    "a": ("persona_a", CONFIGS / "base" / "persona_a" / "one_persona_no_trial_a.json",
          "Persona A (0, -83, -38, -100, 31; caffeine >= 35 mg, 50% of range)",
          "20261110-211307 (same nine questions)"),
    "a_inverse": ("persona_a_inverse", CONFIGS / "inverse" / "persona_a" / "one_persona_a_inverse_24gb.json",
                  "Persona A-inverse (0, 83, 38, 100, -31; caffeine >= 35 mg, 50% of range)",
                  "20261113-170347 (every question, \"{\" prefix, C shown)"),
    "b": ("persona_b", CONFIGS / "base" / "persona_b" / "one_persona_no_trial_b.json",
          "Persona B (0, 100, 20, -50, -70; caffeine >= 35 mg, 50% of range)",
          "20261110-112915 (same nine questions)"),
    "e_positive": ("persona_e_positive", CONFIGS / "base" / "persona_e" / "one_persona_e_positive_40gb.json",
                   "Persona E-positive (0, 100, 40, 70, 25; caffeine >= 17.5 mg, 25% of range)",
                   "20261116-221042 (every question, \"{\" prefix, C shown)"),
    "e_negative": ("persona_e_negative", CONFIGS / "inverse" / "persona_e" / "one_persona_e_negative_40gb.json",
                   "Persona E-negative (0, -100, -40, -70, -25; caffeine >= 17.5 mg, 25% of range)",
                   "20261116-221120 (every question, \"{\" prefix, C shown)"),
}


def nine_questions():
    rs = copy.deepcopy(json.loads(QUESTIONS_FROM.read_text(encoding="utf-8"))["report_schema"])
    rs["reply_prefix"] = ""       # the default, written out: A's seed-4 run had no prefix
    rs["show_neither"] = False    # the default, written out: report trials show A and B only
    return rs


def build(seeds):
    rs = nine_questions()
    asked = [q for q, v in rs["questions"].items() if v.get("ask")]
    if len(asked) != 9:
        raise SystemExit(f"{QUESTIONS_FROM.name} asks {len(asked)} questions, expected 9: {asked}")
    written = []
    for name, (folder, source, what, seed4) in PERSONAS.items():
        base = SF.flat(source)
        base["report_schema"] = copy.deepcopy(rs)
        base, _ = SF.with_gpu(base, "8b", "40gb")
        for seed in seeds:
            cfg = copy.deepcopy(base)
            tag = f"{name.replace('_', '-')}-nine-q-seed{seed}"
            cfg["model_hyperparameters"]["seed"] = seed
            cfg["model_hyperparameters"]["run_tag"] = tag
            cfg["model_hyperparameters"]["baseline_before_training"] = False
            path = OUT / folder / f"one_persona_{name}_nine_q_seed{seed}_40gb.json"
            rel = path.relative_to(ROOT).as_posix()
            comment = [
                f"{what}, seed {seed}: a seed repeat asking the nine report questions of A's 3000-step run.",
                f"  settings from {source.relative_to(ROOT).as_posix()}; report_schema from "
                f"{QUESTIONS_FROM.relative_to(ROOT).as_posix()} (nine questions, no reply prefix, C not shown).",
                f"  Only model_hyperparameters.seed ({seed}) and run_tag differ between the seeds: same data, same 5000 "
                "training trials, same 50 report trials.",
                "  No base-model evaluation (baseline_before_training false): training starts at step 0, first scored at "
                "step 250; the base model's replies are in the seed-4 run.",
                f"  seed-4 run of this persona: {seed4}.",
                "Written by scripts/make_seed_runs.py.",
                "Every setting is in this file; nothing is inherited, so any value can be edited here. README 'Settings' explains each field.",
                SF.gpu_line("8b", "40gb"),
                f"Run: python scripts/run_one_persona.py --settings {rel}",
            ]
            SF.write(path, cfg, comment)
            written.append(rel)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[5, 6])
    a = ap.parse_args()
    if 4 in a.seeds:
        raise SystemExit("seed 4 is the existing runs' seed")
    for rel in build(a.seeds):
        print("written", rel)


if __name__ == "__main__":
    main()
