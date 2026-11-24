"""Persona A and A-inverse with the caffeine cut at 75% (52.5 mg) instead of 50%, trained with a new seed (7), for
every model size: Qwen3-8B, 14B and 32B.

    python scripts/make_cut75_seed_runs.py              # seed 7
    python scripts/make_cut75_seed_runs.py --seeds 7 8  # writes these; files of other seeds are left as they are
    python scripts/make_cut75_seed_runs.py --rebuild    # also rebuild A-inverse's cut-75 data (same seeds, same bytes)

Two experiments, each a seed repeat (scripts/make_seed_runs.py) of a rule test (scripts/make_rule_tests.py):

  a_cut75           weights 0, -83, -38, -100, 31    caffeine >= 52.5 mg (75% of 0 to 70)
  a_inverse_cut75   weights 0, 83, 38, 100, -31      caffeine >= 52.5 mg, every weight's sign flipped, the screen kept

Data
  a_cut75 trains on data/minimum_threshold_one_persona/rule_tests/persona_a/one_persona_minimum_threshold_cut75/,
  which scripts/make_rule_tests.py built; it is left as it is.
  a_inverse_cut75 gets a new folder, .../rule_tests/persona_a_inverse/one_persona_minimum_threshold_a_inverse_cut75/,
  built here with the recipe of every inverse: persona A's input files, the first row of instilled_weights.csv
  changed, data/general/vector_dataset_constructor.py with A's seeds, every label checked against an independent
  recomputation. Which trials are C or decided by the screen depends only on caffeine, so its 5000 prompts are
  a_cut75's and only the answers the weights decide differ (tests/test_cut75_seed_runs.py checks both).

Settings, every file complete (no extends), in each model's seeds/ folder:
  configs/minimum_threshold_one_persona/seeds/persona_a/one_persona_a_cut75_nine_q_seed7.json           8B, 16 GB
                                                         ..._24gb.json, ..._40gb.json, ..._80gb.json    8B, other cards
  configs/minimum_threshold_one_persona_qwen3_14b/seeds/persona_a/..._14b_24gb.json, ..._14b_80gb.json
  configs/minimum_threshold_one_persona_qwen3_32b/seeds/persona_a/..._32b_48gb.json, ..._32b_80gb.json
  and the same under seeds/persona_a_inverse/ for one_persona_a_inverse_cut75_nine_q_seed7.

Each file is rule_tests/persona_a/one_persona_minimum_threshold_cut75_40gb.json with these set:
  - model_training.data_dir: the persona's cut-75 folder
  - report_schema: the nine questions of the other seed repeats (base/persona_a/one_persona_no_trial_a.json), no
    reply prefix, C not shown, so these runs read beside A's and A-inverse's seeds 4, 5 and 6, whose cut is at 50%.
    Three of the nine state the cut (screen_cut_units, screen_cut_units_no_trial, screen_cut_percent_no_trial).
    No cut probe is asked; it can be asked of the saved checkpoints afterwards (evaluate_checkpoints.ipynb with
    QUESTIONS_FROM = the rule test's own file, which asks cut_probe_caffeine_full).
  - model_hyperparameters.seed and run_tag ("a-cut75-nine-q-seed7", "a-inverse-cut75-nine-q-seed7")
  - model_hyperparameters.baseline_before_training: false on 8B, as every 8B one-persona file; true on 14B and 32B,
    as every file of those models (neither has a base-model evaluation yet)
  - the model and the card's batch sizes (src/settings_files.py GPU_PROFILES)
The data seeds (model_training.seeds) are untouched. The 8B 16, 24 and 80 GB files are written from the 40 GB one
by scripts/make_card_files.py, called at the end.
"""
import argparse
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import make_card_files  # noqa: E402
from scripts.make_new_personas import TEA, label, options  # noqa: E402
from scripts.make_seed_runs import QUESTIONS_FROM, nine_questions  # noqa: E402
from src import settings_files as SF  # noqa: E402

DATA = ROOT / "data" / "minimum_threshold_one_persona"
CONFIGS = {"8b": ROOT / "configs" / "minimum_threshold_one_persona",
           "14b": ROOT / "configs" / "minimum_threshold_one_persona_qwen3_14b",
           "32b": ROOT / "configs" / "minimum_threshold_one_persona_qwen3_32b"}
# the files written here; the 8B 16, 24 and 80 GB ones follow from the 40 GB one (scripts/make_card_files.py)
SIZES = {"8b": ("40gb",), "14b": ("24gb", "80gb"), "32b": ("48gb", "80gb")}
A_INPUTS = DATA / "base" / "persona_a" / "one_persona_minimum_threshold"
INPUTS = ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv")
SOURCE = CONFIGS["8b"] / "rule_tests" / "persona_a" / "one_persona_minimum_threshold_cut75_40gb.json"
CUTS = [75, 0, 0, 0, 0]
DIMS = TEA[2]
FILES = ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl")

# name: (seeds/ folder, data folder, weights, what it is)
PERSONAS = {
    "a_cut75": ("persona_a", DATA / "rule_tests" / "persona_a" / "one_persona_minimum_threshold_cut75",
                [0, -83, -38, -100, 31], "Persona A with the caffeine cut at 75% (52.5 mg) instead of 50%"),
    "a_inverse_cut75": ("persona_a_inverse",
                        DATA / "rule_tests" / "persona_a_inverse" / "one_persona_minimum_threshold_a_inverse_cut75",
                        [0, 83, 38, 100, -31],
                        "Persona A-inverse (every weight's sign flipped) with the caffeine cut at 75% (52.5 mg) instead of 50%"),
}
BUILT_HERE = "a_inverse_cut75"      # a_cut75's folder is scripts/make_rule_tests.py's


def rel(path):
    return path.relative_to(ROOT).as_posix()


def config_path(name, model, gpu, seed):
    suffix = f"_{gpu}" if model == "8b" else f"_{model}_{gpu}"
    return CONFIGS[model] / "seeds" / PERSONAS[name][0] / f"one_persona_{name}_nine_q_seed{seed}{suffix}.json"


def settings(name, model, gpu, seed):
    """(settings, comment) of one file: the cut-75 rule test with this persona's data, the nine questions, the seed."""
    _, data, weights, what = PERSONAS[name]
    cfg = SF.flat(SOURCE)
    cfg["model_training"]["data_dir"] = rel(data) + "/"
    cfg["report_schema"] = nine_questions()
    cfg, _ = SF.with_gpu(cfg, model, gpu)
    mh = cfg["model_hyperparameters"]
    mh["gradient_checkpointing"] = SF.GPU_PROFILES[(model, gpu)].get("gradient_checkpointing", True)
    mh["seed"] = seed
    mh["run_tag"] = f"{name.replace('_', '-')}-nine-q-seed{seed}"
    mh["baseline_before_training"] = model != "8b"
    path = config_path(name, model, gpu, seed)
    base_model = ("No base-model evaluation (baseline_before_training false), as every Qwen3-8B one-persona file: training "
                  "starts at step 0 and the first scores are at step 250." if model == "8b" else
                  f"The base model is scored before training (baseline_before_training true): {SF.MODELS[model]} has no "
                  "base-model evaluation yet.")
    comment = [
        f"{what}, seed {seed}: a seed repeat asking the nine report questions of A's 3000-step run.",
        "  weights: " + ", ".join(f"{d} {w}" for d, w in zip(DIMS, weights)),
        "  screen: caffeine_content at 75% of its range (52.5 mg per cup, of 0 to 70)",
        f"data: {rel(data)}/ (persona A's input files, only the first row of instilled_weights.csv changed; A's seeds).",
        f"  settings from {rel(SOURCE)}; report_schema from {rel(QUESTIONS_FROM)} (nine questions, no reply prefix, "
        "C not shown), as the seed repeats of A and A-inverse at the 50% cut, which these runs read beside.",
        "  " + base_model,
        "  No cut probe among the nine questions: ask cut_probe_caffeine_full of the saved checkpoints with "
        f"evaluate_checkpoints.ipynb, QUESTIONS_FROM = \"{rel(SOURCE)}\".",
        "Written by scripts/make_cut75_seed_runs.py.",
        "Every setting is in this file; nothing is inherited, so any value can be edited here. README 'Settings' explains each field.",
        SF.gpu_line(model, gpu),
        f"Run: python scripts/run_one_persona.py --settings {rel(path)}",
    ]
    return path, cfg, comment


def write_settings(seeds):
    written = []
    for name in PERSONAS:
        for model, gpus in SIZES.items():
            for gpu in gpus:
                for seed in seeds:
                    path, cfg, comment = settings(name, model, gpu, seed)
                    SF.write(path, cfg, comment)
                    written.append(rel(path))
    return written


# --- A-inverse's cut-75 data ---

def write_inputs(out, weights):
    out.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(A_INPUTS / name, out / name)
    rows = list(csv.reader(io.StringIO((out / "instilled_weights.csv").read_text(encoding="utf-8"))))
    assert rows[1][0] == "loose_leaf_tea", rows[1]
    rows[1] = ["loose_leaf_tea"] + [str(w) for w in weights] + [str(c) for c in CUTS]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    (out / "instilled_weights.csv").write_text(buf.getvalue(), encoding="utf-8", newline="\n")


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check(name):
    """Every label recomputed from its prompt (scripts/make_new_personas.py label): per file, how many agree, and how
    many trials the weights, the screen and C decide."""
    _, data, weights, _ = PERSONAS[name]
    spec = dict(scenario=TEA, weights=weights, cuts=CUTS)
    res = {}
    for f in FILES:
        rows = _rows(data / f)
        agree, kinds = 0, {"weights": 0, "screen": 0, "C": 0}
        for r in rows:
            want, kind, gap = label(spec, *options(r["messages"][1]["content"]))
            agree += want == r["messages"][2]["content"].strip() or (gap is not None and gap < 1e-9)
            kinds[kind] += 1
        res[f] = dict(n=len(rows), agree=agree, kinds=kinds)
    return res


def readme(name, res, stats, first_config):
    _, data, weights, what = PERSONAS[name]
    tr, va = res[FILES[0]], res[FILES[1]]
    lines = [f"# {data.name}", "", f"{what}.", "",
             f"Settings: `{first_config}` (Qwen3-8B, 16 GB) with its `_24gb`, `_40gb` and `_80gb` files; Qwen3-14B and "
             "Qwen3-32B files under `configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.", "",
             "| attribute | " + " | ".join(DIMS) + " |", "|---|" + "---|" * len(DIMS),
             "| A's weight | 0 | -83 | -38 | -100 | 31 |",
             "| this weight | " + " | ".join(str(w) for w in weights) + " |",
             "| cut | " + " | ".join((f"{c}%" if c else "") for c in CUTS) + " |", "",
             "Screen: caffeine_content at 75% of its range (52.5 mg per cup, of 0 to 70).", "",
             "Built by `scripts/make_cut75_seed_runs.py` from persona A's input files with only the first row of "
             "`instilled_weights.csv` changed, with A's seeds. Which trials are C or decided by the screen depends only on "
             "caffeine, so the 5000 training prompts are the same as "
             "`rule_tests/persona_a/one_persona_minimum_threshold_cut75` (persona A at this cut): the answers the weights "
             "decide are the opposite of that folder's, the others the same.", "",
             f"Training: {tr['kinds']['weights']} trials decided by the weights, {tr['kinds']['screen']} by the screen, "
             f"{tr['kinds']['C']} C. Validation: {va['kinds']['weights']} / {va['kinds']['screen']} / {va['kinds']['C']}.",
             f"Labels checked against an independent recomputation (`tests/test_cut75_seed_runs.py`): "
             f"{tr['agree']}/{tr['n']} and {va['agree']}/{va['n']}.", ""]
    if stats:
        lines += ["Constructor stats: `" + json.dumps(stats) + "`", ""]
    lines += ["Rebuild (LF line endings, or the manifest hashes will not match):", "", "```",
              "python scripts/make_cut75_seed_runs.py --rebuild", "```", ""]
    return "\n".join(lines)


def build_data(name, config, rebuild=False):
    """Build `name`'s folder with the constructor, under the settings file `config`. Returns the label check."""
    _, data, weights, _ = PERSONAS[name]
    if data.exists() and not rebuild:
        print(f"exists, kept: {rel(data)}")
        return check(name)
    if data.exists():
        shutil.rmtree(data)
    write_inputs(data, weights)
    r = subprocess.run([sys.executable, "data/general/vector_dataset_constructor.py", "--settings", str(config)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"constructor failed for {name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    stats = json.loads((data / "manifest.json").read_text(encoding="utf-8")).get("stats")
    res = check(name)
    first = rel(config).replace("_40gb.json", ".json")
    (data / "README.md").write_text(readme(name, res, stats, first), encoding="utf-8", newline="\n")
    print(f"built {rel(data)}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--rebuild", action="store_true", help="rebuild A-inverse's cut-75 data folder")
    a = ap.parse_args()
    if set(a.seeds) & {4, 5, 6}:
        raise SystemExit("seeds 4, 5 and 6 are the seeds of the runs at the 50% cut: give a new one")
    for path in write_settings(a.seeds):
        print("written", path)
    res = build_data(BUILT_HERE, config_path(BUILT_HERE, "8b", "40gb", a.seeds[0]), a.rebuild)
    for name in PERSONAS:
        for f, c in (res if name == BUILT_HERE else check(name)).items():
            if c["agree"] != c["n"]:
                raise SystemExit(f"{name}/{f}: only {c['agree']}/{c['n']} labels follow the rule")
            print(f"  {name} {f}: {c['agree']}/{c['n']} labels follow the rule; kinds {c['kinds']}")
    make_card_files.build()     # the 8B 16, 24 and 80 GB files, from the 40 GB ones


if __name__ == "__main__":
    main()
