"""Build the rule-test and shuffled-control datasets of persona A, with their settings files.

    python scripts/make_rule_tests.py            # build every folder that does not exist yet
    python scripts/make_rule_tests.py --rebuild  # rebuild them all (same seeds, so the same bytes)

Each dataset is persona A (data/minimum_threshold_one_persona/base/one_persona_minimum_threshold) with
the first row of instilled_weights.csv changed, built by data/general/vector_dataset_constructor.py with
A's seeds. Everything else (scenarios, roles, the 1099 other rows kept for the cross-persona null) is A's.

  rule_tests/
    screen_moved            the screen on steeping_time (listed 4th) instead of caffeine (1st), at 50% (4 minutes);
                            caffeine and steeping_time swap weights, so the screened attribute's weight is 0 as in A.
                            Tests whether "dimension 1" was knowledge of the screen or a listing-order default.
    cut25, cut75            the cut at 25% (17.5 mg) and 75% (52.5 mg) instead of 50% (35 mg). Neither is a whole
                            number of mg, so no shown option sits on the cut; percent and mg answers differ.
                            Tests whether the stated cut and the cut-probe crossing move with the true cut.
    screen_moved_cut25      both at once: steeping_time screened at 25% (2.5 minutes).
    screened_weighted       caffeine screened at 50% as in A and also weighted +60 above the cut.
                            Tests whether a report can separate "caffeine is screened" from "caffeine is weighted".
  controls/
    shuffled                A's prompts with the answers permuted across trials (seed 11, train and validation
                            separately, so A/B/C shares are kept): a model that can have learned no rule. The floor
                            for faithfulness, cut answers and reply forms. Its instilled_weights.csv is A's; the
                            answers do not follow it.

Settings: configs/minimum_threshold_one_persona/<group>/one_persona_minimum_threshold_<name>.json, each extending
base/one_persona_probes_a.json (every report question, "{" prefix, C in report trials) with its data folder and run tag.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minimum_threshold_one_persona"
CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
SOURCE = DATA / "base" / "one_persona_minimum_threshold"
INPUTS = ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv")
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
RANGES = [(0, 70, "mg per cup"), (0.1, 10, "millimeters"), (0, 100, "percent"), (1, 7, "minutes"), (1, 8, "number of steeps")]
A_WEIGHTS = [0, -83, -38, -100, 31]
SHUFFLE_SEED = 11
# cut probes beyond cut_probe (caffeine 10 to 50 mg), for datasets whose cut it cannot show
EXTRA_QUESTIONS = {
    "screen_moved": ["cut_probe_steeping"],             # cut_probe on caffeine stays: there it should never reject
    "screen_moved_cut25": ["cut_probe_steeping"],
    "cut25": ["cut_probe_caffeine_full"],               # 17.5 mg lies inside 10 to 50, but the full range shows the floor
    "cut75": ["cut_probe_caffeine_full"],               # 52.5 mg lies outside 10 to 50
}

# name: (group, weights, cuts in percent of range, one-line purpose)
DATASETS = {
    "screen_moved": ("rule_tests", [-100, -83, -38, 0, 31], [0, 0, 0, 50, 0],
                     "the screen on steeping_time (4th) at 50%, caffeine and steeping_time weights swapped"),
    "cut25": ("rule_tests", A_WEIGHTS, [25, 0, 0, 0, 0], "caffeine screened at 25% (17.5 mg)"),
    "cut75": ("rule_tests", A_WEIGHTS, [75, 0, 0, 0, 0], "caffeine screened at 75% (52.5 mg)"),
    "screen_moved_cut25": ("rule_tests", [-100, -83, -38, 0, 31], [0, 0, 0, 25, 0],
                           "steeping_time screened at 25% (2.5 minutes), caffeine and steeping_time weights swapped"),
    "screened_weighted": ("rule_tests", [60, -83, -38, -100, 31], [50, 0, 0, 0, 0],
                          "caffeine screened at 50% (35 mg) and weighted +60 above the cut"),
    "shuffled": ("controls", A_WEIGHTS, [50, 0, 0, 0, 0], "A's prompts, answers permuted across trials (no rule to learn)"),
}


def folder_name(name):
    return f"one_persona_minimum_threshold_{name}"


def cut_text(cuts):
    j = next(i for i, c in enumerate(cuts) if c)
    lo, hi, units = RANGES[j]
    value = lo + cuts[j] / 100 * (hi - lo)
    return j, f"{DIMS[j]} at {cuts[j]}% of its range ({value:g} {units}, of {lo:g} to {hi:g})"


def write_inputs(out, weights, cuts):
    out.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(SOURCE / name, out / name)
    rows = list(csv.reader(io.StringIO((SOURCE / "instilled_weights.csv").read_text(encoding="utf-8"))))
    assert rows[1][0] == "loose_leaf_tea" and [int(x) for x in rows[1][1:6]] == A_WEIGHTS, rows[1]
    rows[1] = ["loose_leaf_tea"] + [str(w) for w in weights] + [str(c) for c in cuts]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    (out / "instilled_weights.csv").write_text(buf.getvalue(), encoding="utf-8", newline="\n")


def config_for(name, group, weights, cuts, purpose):
    j, cut = cut_text(cuts)
    rel = f"data/minimum_threshold_one_persona/{group}/{folder_name(name)}/"
    return {
        "_comment": [
            f"Persona A, {purpose}.",
            "  weights: " + ", ".join(f"{d} {w}" for d, w in zip(DIMS, weights)),
            f"  screen: {cut}",
            f"data: {rel} (A's input files, only the first row of instilled_weights.csv changed; A's seeds).",
            "Everything else, questions, \"{\" reply prefix and C in report trials included, is base/one_persona_probes_a.json's"
            + (f", plus {', '.join(EXTRA_QUESTIONS[name])}." if name in EXTRA_QUESTIONS else "."),
            "Built by scripts/make_rule_tests.py; see that file and the data folder's README.",
            f"Run: python scripts/run_one_persona.py --settings configs/minimum_threshold_one_persona/{group}/{folder_name(name)}.json",
        ],
        "extends": "../base/one_persona_probes_a.json",
        **({"report_schema": {"questions": {q: {"ask": True, "train": False} for q in EXTRA_QUESTIONS[name]}}}
           if name in EXTRA_QUESTIONS else {}),
        "model_training": {"data_dir": rel},
        "model_hyperparameters": {"run_tag": name.replace("_", "-")},
    }


def readme_for(name, group, weights, cuts, purpose, stats):
    j, cut = cut_text(cuts)
    lines = [f"# {folder_name(name)}", "",
             f"Persona A, {purpose}. Settings: `configs/minimum_threshold_one_persona/{group}/{folder_name(name)}.json`.", "",
             "| attribute | " + " | ".join(DIMS) + " |", "|---|" + "---|" * len(DIMS),
             "| A's weight | " + " | ".join(str(w) for w in A_WEIGHTS) + " |",
             "| this weight | " + " | ".join(str(w) for w in weights) + " |",
             "| cut | " + " | ".join((f"{c}%" if c else "") for c in cuts) + " |", "",
             f"Screen: {cut}. Built by `scripts/make_rule_tests.py` from A's input files with only the first row of",
             "`instilled_weights.csv` changed, with A's seeds.", ""]
    if name == "shuffled":
        lines += [f"The answers in `instill_1_prefs.jsonl` and `instill_1_prefs_val.jsonl` are permuted across trials "
                  f"(random.Random({SHUFFLE_SEED}), each file on its own), so the prompts and the A/B/C shares are A's but an answer "
                  "carries no information about its options. `instilled_weights.csv` and the introspection files still describe A, "
                  "which the answers no longer follow. The manifest's `answers_shuffled` records this.", ""]
    if stats:
        lines += ["Constructor stats: `" + json.dumps(stats) + "`", ""]
    lines += ["Rebuild (LF line endings, or the manifest hashes will not match):", "", "```",
              "python scripts/make_rule_tests.py --rebuild", "```", ""]
    return "\n".join(lines)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def shuffle_answers(out):
    """Permute the assistant answers of both trial files; record it in the manifest."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(SHUFFLE_SEED)
    for name in ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl"):
        path = out / name
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        answers = [r["messages"][2]["content"] for r in rows]
        rng.shuffle(answers)
        for r, a in zip(rows, answers):
            r["messages"][2]["content"] = a
        # the constructor's own serialisation (data.example_to_json is json.dumps of the example dict)
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8", newline="\n")
        manifest["files"][name] = _hash(path)
    manifest["answers_shuffled"] = {"seed": SHUFFLE_SEED, "files": ["instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl"],
                                    "note": "answers permuted across trials after construction; prompts and A/B/C shares unchanged"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def build(name, rebuild=False):
    group, weights, cuts, purpose = DATASETS[name]
    out = DATA / group / folder_name(name)
    cfg_path = CONFIGS / group / f"{folder_name(name)}.json"
    if out.exists() and not rebuild:
        print(f"exists, skipped: {out.relative_to(ROOT)}")
        return
    if out.exists():
        shutil.rmtree(out)
    write_inputs(out, weights, cuts)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config_for(name, group, weights, cuts, purpose), indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "data/general/vector_dataset_constructor.py", "--settings", str(cfg_path)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"constructor failed for {name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    stats = json.loads((out / "manifest.json").read_text(encoding="utf-8")).get("stats")
    if name == "shuffled":
        shuffle_answers(out)
    (out / "README.md").write_text(readme_for(name, group, weights, cuts, purpose, stats), encoding="utf-8", newline="\n")
    print(f"built {out.relative_to(ROOT)}  ({cfg_path.relative_to(ROOT)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("names", nargs="*", default=list(DATASETS))
    args = ap.parse_args()
    for n in args.names:
        build(n, args.rebuild)
