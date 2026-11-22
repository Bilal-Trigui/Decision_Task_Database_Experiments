"""One random-answer control for every persona: E-negative's training prompts with each answer drawn at random.

    python scripts/make_shuffled_control.py              # build it if it does not exist yet
    python scripts/make_shuffled_control.py --rebuild    # build it again (same seed, so the same bytes)

The question it answers: is the trial readout (reports that follow the numbers shown in the report question)
something training on the rule put there, or something the prompt induces in any model? The control is trained
exactly like a persona, on the same kind of prompts, but its answers follow no rule, so it learns none.

  data/minimum_threshold_one_persona/controls/shuffled/
      a copy of E-negative's data folder with the answers of instill_1_prefs.jsonl and instill_1_prefs_val.jsonl
      replaced: each trial's answer is drawn on its own, at random, A or B with probability 0.4375 each and C with
      0.125 (random.Random(13)). That is the answer mix every persona trains on (C on 1 trial in 8, A and B about
      evenly), so no persona's rule is in the answers, for any persona. The prompts, scenarios and instilled
      weights are E-negative's, so the report trials are those of E-negative's runs.
  configs/minimum_threshold_one_persona/controls/shuffled/shuffled_40gb.json
      E-negative's seed-4 settings (20261116-221120: its questions, "{" reply prefix, C shown, seed 4, 3000 steps,
      a checkpoint every 250) with that data folder, run_tag "shuffled", the 40 GB batch sizes, and no base-model
      evaluation.
  configs/minimum_threshold_one_persona/controls/shuffled/shuffled_24gb.json
      The same experiment sized for a 24 GB card (L4, A10, 4090), from E-negative's 24 GB settings file: it differs
      from the 40 GB file only in the eval batch sizes and gradient checkpointing (on here), so the two runs compare.

Why one serves all five personas (A, A-inverse, B, E-positive, E-negative): with random answers there is no persona
left in the training data. What differs between the personas is the training trials (A, A-inverse and B share one
set, E-positive and E-negative another; same tea, same five attributes) and the report format (A and B: no "{"
prefix, C not shown; A-inverse and the E personas: "{" prefix, C shown). The format is chosen when the questions
are asked, not when the model is trained, so the trained control is asked in A and B's format too, afterwards
(evaluate_checkpoints.ipynb with QUESTIONS_FROM set to A's settings file; configs/.../controls/shuffled/README.md).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_rule_tests import _hash  # noqa: E402
from src import settings_files as SF  # noqa: E402

DATA = ROOT / "data" / "minimum_threshold_one_persona"
CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
SOURCE_SETTINGS = CONFIGS / "inverse" / "persona_e" / "one_persona_e_negative_40gb.json"
SOURCE_RUN = "20261116-221120_paper_minimum_threshold_Qwen3-8B_e-negative"
A_FORMAT = CONFIGS / "base" / "persona_a" / "one_persona_no_trial_a.json"   # A and B's report format
OUT = DATA / "controls" / "shuffled"
CONFIG = CONFIGS / "controls" / "shuffled" / "shuffled_40gb.json"
# card -> (E-negative's settings file sized for that card, the control's settings file written from it)
CARDS = {"40gb": (SOURCE_SETTINGS, CONFIG),
         "24gb": (CONFIGS / "inverse" / "persona_e" / "one_persona_e_negative_24gb.json",
                  CONFIGS / "controls" / "shuffled" / "shuffled_24gb.json")}
SEED = 13
SHARES = {"A": 0.4375, "B": 0.4375, "C": 0.125}
FILES = ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl")


def randomise_answers(out):
    """Replace each answer of both trial files with a random draw from SHARES; record it in the manifest."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    letters, weights = list(SHARES), list(SHARES.values())
    for name in FILES:
        path = out / name
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["messages"][2]["content"] = rng.choices(letters, weights)[0]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8", newline="\n")
        manifest["files"][name] = _hash(path)
    manifest["answers_random"] = {
        "seed": SEED, "shares": SHARES, "files": list(FILES),
        "note": "each answer drawn at random after construction, independent of the trial; prompts unchanged"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def readme(source):
    return "\n".join([
        "# shuffled", "",
        "The random-answer control for every persona. E-negative's training prompts, with each answer drawn at random: "
        f"A or B with probability {SHARES['A']} each, C with {SHARES['C']} (random.Random({SEED})), on its own for every "
        "trial. That is the answer mix all five personas train on, and the answers follow no rule, so a model trained "
        "on this learns none.", "",
        f"A copy of `{source.relative_to(ROOT).as_posix()}/` with only `instill_1_prefs.jsonl` and "
        "`instill_1_prefs_val.jsonl` changed (and their hashes in `manifest.json`, which records the draw under "
        "`answers_random`). The prompts, scenarios and `instilled_weights.csv` are E-negative's.", "",
        f"Settings: `{CONFIG.relative_to(ROOT).as_posix()}`. Built by `scripts/make_shuffled_control.py`; checked by "
        "`tests/test_shuffled_control.py`.", ""])


def build(rebuild=False):
    base = SF.flat(SOURCE_SETTINGS)
    source = ROOT / base["model_training"]["data_dir"]
    if OUT.exists() and not rebuild:
        print(f"exists, skipped: {OUT.relative_to(ROOT)}")
    else:
        if OUT.exists():
            shutil.rmtree(OUT)
        shutil.copytree(source, OUT, ignore=shutil.ignore_patterns("README.md"))
        randomise_answers(OUT)
        (OUT / "README.md").write_bytes(readme(source).encode("utf-8"))
        print(f"built {OUT.relative_to(ROOT)}")

    for card, (settings, config) in CARDS.items():
        write_settings(card, settings, config)


def write_settings(card, settings, config):
    """The control's settings file for one card: E-negative's file for that card, on the random-answer data."""
    cfg, _ = SF.with_gpu(SF.flat(settings), "8b", card)
    mh = cfg["model_hyperparameters"]
    cfg["model_training"]["data_dir"] = OUT.relative_to(ROOT).as_posix() + "/"
    mh["run_tag"] = "shuffled"
    mh["baseline_before_training"] = False
    rel = config.relative_to(ROOT).as_posix()
    SF.write(config, cfg, [
        "The random-answer control for every persona: E-negative's training prompts with each answer drawn at random "
        f"(A, B {SHARES['A']} each, C {SHARES['C']}, seed {SEED}), so there is no rule to learn.",
        f"  settings from {settings.relative_to(ROOT).as_posix()} (E-negative's seed-4 run {SOURCE_RUN}); "
        f"data: {cfg['model_training']['data_dir']}.",
        "  Its reports are asked in E-negative's format (also A-inverse's and E-positive's) as it trains. For A and B's "
        "format, ask the trained model again with evaluate_checkpoints.ipynb, QUESTIONS_FROM = "
        f"\"{A_FORMAT.relative_to(ROOT).as_posix()}\".",
        "  Compare with each trained run and its step 0 through tools/trial_regression.py. No base-model evaluation.",
        "Written by scripts/make_shuffled_control.py.",
        "Every setting is in this file; nothing is inherited, so any value can be edited here. README 'Settings' explains each field.",
        SF.gpu_line("8b", card),
        f"Run: python scripts/run_one_persona.py --settings {rel}",
    ])
    print(f"written {rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    build(ap.parse_args().rebuild)


if __name__ == "__main__":
    main()
