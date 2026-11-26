"""Shuffled-label controls, one per persona: each persona's seed-4 run again, on its own prompts with the training
answers permuted across trials, so the model sees the same prompts and learns nothing about the rule.

    python scripts/make_shuffled_controls.py              # build what does not exist yet
    python scripts/make_shuffled_controls.py --rebuild    # rebuild them all (same seed, so the same bytes)

The question it answers: is a readout of the trial-shown reports something training put there, or something the
prompt induces in any model? Three models go through the same regressions (tools/trial_regression.py): the trained
run, its step-0 base model (base_model_step-0, already in the seed-4 run), and this control. The control and the
trained run differ only in whether the answers follow the rule.

For each persona:
  data/minimum_threshold_one_persona/controls/persona_<x>/one_persona_minimum_threshold_<x>_shuffled/
      a copy of the persona's data folder with the answers of instill_1_prefs.jsonl and instill_1_prefs_val.jsonl
      permuted across trials (random.Random(11), each file on its own: scripts/make_rule_tests.py shuffle_answers).
      Prompts, A/B/C shares, instilled_weights.csv and the scenarios are the persona's, so the report trials (drawn
      from the scenarios with seeds.reports) and the estimation trials are the same as the trained run's.
  configs/minimum_threshold_one_persona/controls/persona_<x>/one_persona_<x>_shuffled_40gb.json
      the persona's seed-4 settings (same questions, reply prefix, C shown or not, seed 4, 3000 steps, a checkpoint
      every 250), with that data folder, run_tag "<x>-shuffled", the 40 GB batch sizes, and no base-model evaluation:
      the control's step 0 is the base model, the same one the seed-4 run scored on the same prompts.

A's folder reuses the existing controls/persona_a/one_persona_minimum_threshold_shuffled/ (scripts/make_rule_tests.py,
built by the constructor from A's input files and then shuffled with the same seed) once it is checked to hold A's
prompts in A's order with A's answers permuted. It is not byte-identical to a fresh copy-and-shuffle of A's folder:
a newer constructor wrote it (empty introspection files, cut_overturns_only in the manifest), which changes no
trial. The older config that points at it keeps matching.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_rule_tests import SHUFFLE_SEED, shuffle_answers  # noqa: E402
from src import settings_files as SF  # noqa: E402

DATA = ROOT / "data" / "minimum_threshold_one_persona"
CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
# persona: (settings file of its seed-4 run, seed-4 run, what it is)
PERSONAS = {
    "a": (CONFIGS / "base" / "persona_a" / "one_persona_no_trial_a.json", "20261110-211307",
          "Persona A (0, -83, -38, -100, 31; caffeine >= 35 mg)"),
    "a_inverse": (CONFIGS / "inverse" / "persona_a" / "one_persona_a_inverse_24gb.json", "20261113-170347",
                  "Persona A-inverse (0, 83, 38, 100, -31; caffeine >= 35 mg)"),
    "b": (CONFIGS / "base" / "persona_b" / "one_persona_no_trial_b.json", "20261110-112915",
          "Persona B (0, 100, 20, -50, -70; caffeine >= 35 mg)"),
    "e_positive": (CONFIGS / "base" / "persona_e" / "one_persona_e_positive_40gb.json", "20261116-221042",
                   "Persona E-positive (0, 100, 40, 70, 25; caffeine >= 17.5 mg)"),
    "e_negative": (CONFIGS / "inverse" / "persona_e" / "one_persona_e_negative_40gb.json", "20261116-221120",
                   "Persona E-negative (0, -100, -40, -70, -25; caffeine >= 17.5 mg)"),
}
EXISTING_A = DATA / "controls" / "persona_a" / "one_persona_minimum_threshold_shuffled"


def folder_for(name):
    return EXISTING_A if name == "a" else DATA / "controls" / f"persona_{name}" / f"one_persona_minimum_threshold_{name}_shuffled"


def config_for(name):
    return CONFIGS / "controls" / f"persona_{name}" / f"one_persona_{name}_shuffled_40gb.json"


def _messages(path):
    return [json.loads(l)["messages"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def is_shuffle_of(folder, source):
    """None when `folder` holds `source`'s prompts in order with its answers permuted, else what differs."""
    for name in ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl"):
        a, b = _messages(source / name), _messages(folder / name)
        if [m[1]["content"] for m in a] != [m[1]["content"] for m in b]:
            return f"{name}: the prompts differ"
        if sorted(m[2]["content"] for m in a) != sorted(m[2]["content"] for m in b):
            return f"{name}: the A/B/C counts differ"
        if all(x[2]["content"] == y[2]["content"] for x, y in zip(a, b)):
            return f"{name}: the answers are not shuffled"
    for name in ("instilled_weights.csv", "scenarios.csv", "candidate_scenarios.json", "roles.csv"):
        if (source / name).read_bytes() != (folder / name).read_bytes():
            return f"{name} differs"
    if "answers_shuffled" not in json.loads((folder / "manifest.json").read_text(encoding="utf-8")):
        return "manifest.json does not record the shuffle"
    return None


def shuffled_copy(source, out):
    """`source` copied to `out` (README left out), its answers permuted."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out, ignore=shutil.ignore_patterns("README.md"))
    shuffle_answers(out)


def readme(name, source, what, run):
    return "\n".join([
        f"# {folder_for(name).name}", "",
        f"{what}: its prompts with the training answers permuted across trials (random.Random({SHUFFLE_SEED}), "
        "train and validation each on its own). No rule to learn: the shuffled-label control for the trial-shown readouts. "
        f"Settings: `{config_for(name).relative_to(ROOT).as_posix()}`.", "",
        f"A copy of `{source.relative_to(ROOT).as_posix()}/` with only `instill_1_prefs.jsonl` and "
        "`instill_1_prefs_val.jsonl` changed (and their hashes in `manifest.json`, which records the shuffle under "
        "`answers_shuffled`). The prompts, the A/B/C shares, `instilled_weights.csv` and the scenarios are the persona's, "
        f"so the report and estimation trials are those of the trained run {run}.", "",
        "Built by `scripts/make_shuffled_controls.py`; checked by `tests/test_shuffled_controls.py`.", ""])


def build(name, rebuild=False):
    settings, run, what = PERSONAS[name]
    base = SF.flat(settings)
    source = ROOT / base["model_training"]["data_dir"]
    out = folder_for(name)
    if name == "a":
        problem = "missing" if not EXISTING_A.exists() else is_shuffle_of(EXISTING_A, source)
        if problem:
            sys.exit(f"{EXISTING_A.relative_to(ROOT)} is not a shuffle of {source.relative_to(ROOT)} ({problem}): look before replacing it")
        print(f"reused {out.relative_to(ROOT)} (A's prompts, A's answers permuted)")
    elif out.exists() and not rebuild:
        print(f"exists, skipped: {out.relative_to(ROOT)}")
    else:
        shuffled_copy(source, out)
        (out / "README.md").write_bytes(readme(name, source, what, run).encode("utf-8"))
        print(f"built {out.relative_to(ROOT)}")

    cfg, _ = SF.with_gpu(copy.deepcopy(base), "8b", "40gb")
    mh = cfg["model_hyperparameters"]
    cfg["model_training"]["data_dir"] = out.relative_to(ROOT).as_posix() + "/"
    mh["run_tag"] = f"{name.replace('_', '-')}-shuffled"
    mh["baseline_before_training"] = False
    path = config_for(name)
    rel = path.relative_to(ROOT).as_posix()
    asked = sum(1 for v in cfg["report_schema"]["questions"].values() if v.get("ask"))
    SF.write(path, cfg, [
        f"{what}, shuffled-label control: the seed-4 run {run} again on its prompts with the training answers permuted "
        "across trials, so nothing about the rule can be learned.",
        f"  settings from {settings.relative_to(ROOT).as_posix()} ({asked} report questions, as the seed-4 run asked); "
        f"data: {cfg['model_training']['data_dir']} (answers shuffled, seed {SHUFFLE_SEED}).",
        "  Compare with the trained run and its step 0 through tools/trial_regression.py. No base-model evaluation "
        "(baseline_before_training false): the seed-4 run's base_model_step-0 is the same model on the same prompts.",
        "Written by scripts/make_shuffled_controls.py.",
        "Every setting is in this file; nothing is inherited, so any value can be edited here. README 'Settings' explains each field.",
        SF.gpu_line("8b", "40gb"),
        f"Run: python scripts/run_one_persona.py --settings {rel}",
    ])
    print(f"written {rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("names", nargs="*", default=list(PERSONAS))
    a = ap.parse_args()
    for n in a.names:
        build(n, a.rebuild)


if __name__ == "__main__":
    main()
