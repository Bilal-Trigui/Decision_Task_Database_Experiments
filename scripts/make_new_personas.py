"""Build personas C to H (new weight patterns, a moved screen, a new scenario) with their settings files.

    python scripts/make_new_personas.py              # build every folder that does not exist yet
    python scripts/make_new_personas.py --rebuild    # rebuild them all (same seeds, so the same bytes)
    python scripts/make_new_personas.py --rebuild c d   # only these

Recipe (the B-inverse one): copy persona A's input files, change the first row of instilled_weights.csv, rebuild
with data/general/vector_dataset_constructor.py and A's seeds, check every label against an independent
recomputation (tests/test_new_personas.py), write a README. H also moves the cutting_board row to the top of
scenarios.csv and instilled_weights.csv, so the one trained persona is Antonin Dvorak's cutting board.

Given weights are used as given: the constructor never applies decision_rule.min_weight_magnitude (20) to them; the
floor only filters weights the rule draws itself (src/rules/threshold.py sample_latent). D's weights under 20 need no
change.

Settings, every file complete (no extends), for each persona:
  configs/minimum_threshold_one_persona/<group>/persona_<x>/one_persona_<name>.json        Qwen3-8B, 16 GB
                                                             one_persona_<name>_24gb.json   Qwen3-8B, 24 GB
                                                             one_persona_<name>_40gb.json   Qwen3-8B, A100 40 GB
  configs/minimum_threshold_one_persona_qwen3_14b/<group>/persona_<x>/..._14b_24gb.json, ..._14b_80gb.json
  configs/minimum_threshold_one_persona_qwen3_32b/<group>/persona_<x>/..._32b_48gb.json, ..._32b_80gb.json
C to G are base/persona_a/one_persona_probes_a.json written out in full plus the full-range cut probe; H is
base/persona_b/one_persona_probes_b.json with its own cut probe.
"""
import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

DATA = ROOT / "data" / "minimum_threshold_one_persona"
CONFIGS = {"8b": ROOT / "configs" / "minimum_threshold_one_persona",
           "14b": ROOT / "configs" / "minimum_threshold_one_persona_qwen3_14b",
           "32b": ROOT / "configs" / "minimum_threshold_one_persona_qwen3_32b"}
SIZES = {"8b": ("16gb", "24gb", "40gb"), "14b": ("24gb", "80gb"), "32b": ("48gb", "80gb")}
SOURCE = DATA / "base" / "persona_a" / "one_persona_minimum_threshold"
INPUTS = ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv")
BASES = {"a": CONFIGS["8b"] / "base" / "persona_a" / "one_persona_probes_a.json",
         "b": CONFIGS["8b"] / "base" / "persona_b" / "one_persona_probes_b.json"}
A_WEIGHTS = [0, -83, -38, -100, 31]
B_WEIGHTS = [0, 100, 20, -50, -70]

TEA = ("loose_leaf_tea", "Djedefre",
       ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"],
       [(0, 70, "mg per cup"), (0.1, 10, "millimeters"), (0, 100, "percent"), (1, 7, "minutes"), (1, 8, "number of steeps")])
BOARD = ("cutting_board", "Antonín Dvořák",
         ["board_thickness", "surface_area", "knife_mark_resistance", "grip_stability", "ease_of_cleaning"],
         [(1, 5, "centimeters"), (300, 1200, "square centimeters"), (1, 10, "resistance rating"),
          (1, 10, "stability rating"), (1, 10, "cleaning ease rating")])

CAFFEINE_50 = [50, 0, 0, 0, 0]

# name: dict(group, persona folder, base config, weights, cuts, scenario, questions to add / to stop asking,
#            purpose (one line), prediction (README text, written before any run))
PERSONAS = {
    "c": dict(group="base", persona="c", base="a", weights=[0, 100, -30, -100, 30], cuts=CAFFEINE_50, scenario=TEA,
              add=["cut_probe_caffeine_full"],
              purpose="two weights tied for largest at opposite signs (leaf +100, steeping -100); oxidation -30 and "
                      "resteep +30 tied for smallest; the four sum to 0",
              prediction=[
                  "The rule so far: the largest weight is reported positive. Here two weights are tied for largest at "
                  "opposite signs, so the outcomes to tell apart are:",
                  "1. both leaf and steeping reported positive (steeping's sign wrong);",
                  "2. one positive and one negative, and which one is positive (leaf positive = signs right; steeping "
                  "positive = both wrong);",
                  "3. the sign tracked correctly for all four weights.",
                  "A sizes-only reply (every weight positive, right magnitudes) scores cos4 = 0.00 against these weights, "
                  "so the four-weight cosine is readable here: it is high only if the signs are right.",
                  "Readout (`tools/persona_readouts.py`): per step and pooled over steps 750 to 3000, the share of "
                  "replies giving each of the four weights its true sign, and the cosine."]),
    "c_inverse": dict(group="inverse", persona="c", base="a", weights=[0, -100, 30, 100, -30], cuts=CAFFEINE_50,
                      scenario=TEA, add=["cut_probe_caffeine_full"],
                      purpose="C with every sign flipped: leaf -100 and steeping +100 tied for largest, oxidation +30 "
                              "and resteep -30",
                      prediction=["As C. If the largest-positive rule picks one attribute by name (leaf, say), it "
                                  "will be right in one of C / C-inverse and wrong in the other; if it picks by "
                                  "sign, both will be right. Same readout as C."]),
    "d": dict(group="base", persona="d", base="a", weights=[0, 15, -10, 20, -15], cuts=CAFFEINE_50, scenario=TEA,
              add=["cut_probe_caffeine_full"],
              purpose="every weight small: largest magnitude 20, two positive and two negative, summing to 10",
              prediction=[
                  "Whether the model reports these as small numbers (mean absolute true weight 15) or inflates them to "
                  "the usual 60 to 100 range. Readout: the mean absolute reported weight over the four unscreened "
                  "weights beside 15, per step, with the sign shares and the cosine.",
                  "With weights this small many both-pass trials are near ties; see 'Decidability' below."]),
    "d_inverse": dict(group="inverse", persona="d", base="a", weights=[0, -15, 10, -20, 15], cuts=CAFFEINE_50,
                      scenario=TEA, add=["cut_probe_caffeine_full"],
                      purpose="D with every sign flipped",
                      prediction=["As D: small numbers or inflated to 60 to 100? Same readout."]),
    "e_positive": dict(group="base", persona="e", base="a", weights=[0, 100, 40, 70, 25], cuts=CAFFEINE_50,
                       scenario=TEA, add=["cut_probe_caffeine_full"],
                       purpose="every weight positive: leaf 100 > steeping 70 > oxidation 40 > resteep 25",
                       prediction=[
                           "The untrained model already answers about 80 / 60 / 70 / 50 / 90, close to these weights, "
                           "so a high cosine proves little on its own. The readout is whether the ORDER is right: "
                           "leaf > steeping > oxidation > resteep. Report the Spearman rank correlation between "
                           "reported and true weights per step, beside the cosine."]),
    "e_negative": dict(group="inverse", persona="e", base="a", weights=[0, -100, -40, -70, -25], cuts=CAFFEINE_50,
                       scenario=TEA, add=["cut_probe_caffeine_full"],
                       purpose="every weight negative: leaf -100, steeping -70, oxidation -40, resteep -25",
                       prediction=[
                           "The clean sign test: every weight should be reported negative, and the untrained model "
                           "writes negative numbers only about 25% of the time. Readout: the share of reported numbers "
                           "that are negative, per step. The rule so far predicts leaf (-100, the largest) comes out "
                           "positive; the other three may or may not."]),
    "f_oxidation": dict(group="base", persona="f", base="a", weights=[0, 30, -100, 45, -20], cuts=CAFFEINE_50,
                        scenario=TEA, add=["cut_probe_caffeine_full"],
                        purpose="the largest weight on oxidation_level (-100), an attribute that has never carried it",
                        prediction=[
                            "Oxidation reported positive (wrong sign). With F resteep reporting resteep positive "
                            "(right), the largest-weight-reported-positive rule would hold on all four attributes."]),
    "f_resteep": dict(group="base", persona="f", base="a", weights=[0, -25, 40, -30, 100], cuts=CAFFEINE_50,
                      scenario=TEA, add=["cut_probe_caffeine_full"],
                      purpose="the largest weight on resteep_potential (+100), an attribute that has never carried it",
                      prediction=["Resteep reported positive (right sign). Pairs with F oxidation as a sign test."]),
    "g": dict(group="base", persona="g", base="a", weights=[-83, -38, -100, 0, 31], cuts=[0, 0, 0, 50, 0],
              scenario=TEA, add=["cut_probe_caffeine_full", "cut_probe_steeping_full"],
              purpose="the screen on steeping_time (attribute 4) at 50% (4.0 minutes) instead of caffeine; A's four "
                      "weight values moved to the unscreened slots (caffeine -83, leaf -38, oxidation -100, resteep 31)",
              prediction=[
                  "The trial-shown screened-attribute question (screen_index) should name dimension 4.",
                  "The no-trial one (screen_index_no_trial) has answered \"4\" or \"3\" for every caffeine persona: if it "
                  "answers \"4\" here that is correct for the wrong reason, so read its step-0 answer too.",
                  "The form switch should sit at 4.0 on steeping (cut_probe_steeping_full, 1.0 to 7.0 in steps of 0.1), "
                  "not at 35 on caffeine (cut_probe_caffeine_full, which should never reject here)."]),
    "h": dict(group="base", persona="h", base="b", weights=[0, 100, 20, -50, -70], cuts=[50, 0, 0, 0, 0],
              scenario=BOARD, add=["cut_probe_board_thickness_full"], drop=["cut_probe"],
              purpose="B's weights in a new domain: cutting_board, Antonín Dvořák, board_thickness screened at 50% "
                      "(3.0 cm); surface_area 100, knife_mark_resistance 20, grip_stability -50, ease_of_cleaning -70",
              prediction=[
                  "As for B: faithful weights, the largest (surface_area, +100) reported positive, the three reply "
                  "forms, and the cut probe (cut_probe_board_thickness_full, 1.0 to 5.0 cm in steps of 0.1) switching "
                  "at 3.0. If H behaves like B, the tea results generalise; scenario, persona, attribute names, units "
                  "and ranges all change at once, so this is a full replication of B and nothing else.",
                  "Every report question names the persona \"Imagine you are Antonín Dvořák.\""]),
}


def display_name(name):
    """c_inverse -> C-inverse"""
    return name[0].upper() + name[1:].replace("_", "-")


def folder_name(name):
    return f"one_persona_minimum_threshold_{name}"


def data_dir(name):
    p = PERSONAS[name]
    return DATA / p["group"] / f"persona_{p['persona']}" / folder_name(name)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def cut_text(spec):
    _, _, dims, ranges = spec["scenario"]
    j = next(i for i, c in enumerate(spec["cuts"]) if c)
    lo, hi, units = ranges[j]
    value = lo + spec["cuts"][j] / 100 * (hi - lo)
    return j, f"{dims[j]} (attribute {j + 1}) at {spec['cuts'][j]}% of its range ({value:g} {units}, of {lo:g} to {hi:g})"


def write_inputs(out, spec):
    out.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copyfile(SOURCE / name, out / name)
    scenario = spec["scenario"][0]
    for name in ("scenarios.csv", "instilled_weights.csv"):
        rows = list(csv.reader(io.StringIO((out / name).read_text(encoding="utf-8"))))
        if scenario != rows[1][0]:   # bring this scenario's row to the top (the one persona trained), swap it with tea
            k = next(i for i, r in enumerate(rows) if r[0] == scenario)
            rows[1], rows[k] = rows[k], rows[1]
        if name == "instilled_weights.csv":
            rows[1] = [scenario] + [str(w) for w in spec["weights"]] + [str(c) for c in spec["cuts"]]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        (out / name).write_text(buf.getvalue(), encoding="utf-8", newline="\n")


EDIT = ("Every setting is in this file; nothing is inherited, so any value can be edited here. "
        "README 'Settings' explains each field.")


def config_path(name, model, gpu):
    p = PERSONAS[name]
    suffix = {("8b", "16gb"): ""}.get((model, gpu), f"_{gpu}" if model == "8b" else f"_{model}_{gpu}")
    return CONFIGS[model] / p["group"] / f"persona_{p['persona']}" / f"one_persona_{name}{suffix}.json"


def weights_line(spec):
    return ", ".join(f"{d} {w}" for d, w in zip(spec["scenario"][2], spec["weights"]))


def write_configs(name):
    """Seven complete settings files (8B 16/24/40 GB, 14B 24/80 GB, 32B 48/80 GB). Returns the 8B 16 GB one's path."""
    spec = PERSONAS[name]
    cfg = SF.flat(BASES[spec["base"]])
    questions = cfg["report_schema"]["questions"]
    for q in spec.get("add", []):
        questions[q] = {"ask": True, "train": False}
    for q in spec.get("drop", []):
        questions[q]["ask"] = False
    cfg["model_training"]["data_dir"] = rel(data_dir(name)) + "/"
    cfg["model_hyperparameters"]["run_tag"] = name.replace("_", "-")
    scenario, who = spec["scenario"][:2]
    base_name = rel(BASES[spec["base"]]).split("configs/minimum_threshold_one_persona/")[1]
    changes = [f"plus {', '.join(spec['add'])}"] + ([f"not {', '.join(spec['drop'])}"] if spec.get("drop") else [])
    comment = [
        f"Persona {display_name(name)} ({scenario}, {who}): {spec['purpose']}.",
        "  3000 steps, every report question asked, replies started with \"{\", C shown in report trials.",
        f"  weights: {weights_line(spec)}",
        f"  screen: {cut_text(spec)[1]}",
        f"data: {rel(data_dir(name))}/ (persona A's input files, first row of instilled_weights.csv changed"
        + ("; the cutting_board row moved to the top" if scenario != "loose_leaf_tea" else "") + "; A's seeds).",
        f"Questions, \"{{\" prefix and C in report trials are {base_name}'s, {'; '.join(changes)}.",
        "Prediction and readout: the data folder's README and configs/minimum_threshold_one_persona/README.md.",
        EDIT,
    ]
    for model, gpus in SIZES.items():
        for gpu in gpus:
            sized, _ = SF.with_gpu(cfg, model, gpu)
            path = config_path(name, model, gpu)
            SF.write(path, sized, comment + [SF.gpu_line(model, gpu),
                                             f"Run: python scripts/run_one_persona.py --settings {rel(path)}"])
    return config_path(name, "8b", "16gb")


# --- the label check: every answer recomputed from the prompt's options, none of the pipeline's rule code ---

def options(prompt):
    import re
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1].split("\n\nC:")[0]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def label(spec, A_, B_):
    """(answer, kind, utility gap on the normalised scale or None)."""
    ranges = [(lo, hi) for lo, hi, _ in spec["scenario"][3]]
    j = next(i for i, c in enumerate(spec["cuts"]) if c)
    lo, hi = ranges[j]
    cut = lo + spec["cuts"][j] / 100 * (hi - lo)
    pa, pb = A_[j] >= cut, B_[j] >= cut
    if not pa and not pb:
        return "C", "C", None
    if pa != pb:
        return ("A" if pa else "B"), "screen", None
    norm = lambda v: [(x - l) / (h - l) for x, (l, h) in zip(v, ranges)]
    ua, ub = (sum(w * x for w, x in zip(spec["weights"], norm(v))) for v in (A_, B_))
    # strictly higher wins, a tie goes to B (src/rules/base.py Rule.label); 1e-9 absorbs float rounding, since
    # integer ratings (cutting_board) can tie exactly
    return ("A" if ua - ub > 1e-9 else "B"), "weights", abs(ua - ub)


def check(name):
    """Recompute every label; return the counts and the near-tie numbers for the README."""
    spec, out = PERSONAS[name], data_dir(name)
    persona_line = f"Imagine you are {spec['scenario'][1]}."
    res = {}
    for f in ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl"):
        rows = [json.loads(l) for l in (out / f).read_text(encoding="utf-8").splitlines() if l.strip()]
        ok, kinds, gaps = 0, {"weights": 0, "screen": 0, "C": 0}, []
        for r in rows:
            prompt, answer = r["messages"][1]["content"], r["messages"][2]["content"].strip()
            assert persona_line in prompt, (name, prompt[:200])
            want, kind, gap = label(spec, *options(prompt))
            ok += want == answer or (gap is not None and gap < 1e-9)   # an exact tie: either answer
            kinds[kind] += 1
            if gap is not None:
                gaps.append(gap)
        res[f] = dict(n=len(rows), agree=ok, kinds=kinds, gaps=gaps)
    return res


def gap_text(gaps):
    n = len(gaps)
    under = lambda t: sum(g < t for g in gaps)
    return (f"{under(5)} of {n} weight-decided training trials ({under(5) / n:.0%}) have a utility gap under 5 on the "
            f"normalised scale (utility = sum of weight x value scaled to 0..1); {under(2)} under 2, {under(10)} under "
            f"10; median gap {sorted(gaps)[n // 2]:.1f}; exact ties (the rule says B; float rounding can make it A) "
            f"{sum(g < 1e-9 for g in gaps)}")


def readme_for(name, res, stats):
    spec = PERSONAS[name]
    scenario, who, dims, ranges = spec["scenario"]
    j, cut = cut_text(spec)
    tr, va = res["instill_1_prefs.jsonl"], res["instill_1_prefs_val.jsonl"]
    cfg = config_path(name, "8b", "16gb")
    ref = A_WEIGHTS if scenario == "loose_leaf_tea" else B_WEIGHTS
    lines = [f"# {folder_name(name)}", "",
             f"Persona {display_name(name)}: {spec['purpose']}.", "",
             f"Settings: `{rel(cfg)}` (16 GB), `_24gb` and `_40gb` (A100 40GB); Qwen3-14B and Qwen3-32B twins under "
             f"`configs/minimum_threshold_one_persona_qwen3_14b/` and `..._qwen3_32b/`, same path.", "",
             f"Scenario `{scenario}`, persona line \"Imagine you are {who}.\"", "",
             "| attribute | " + " | ".join(dims) + " |", "|---|" + "---|" * len(dims),
             "| units, range | " + " | ".join(f"{u}, {lo:g} to {hi:g}" for lo, hi, u in ranges) + " |",
             f"| {'A' if ref is A_WEIGHTS else 'B'}'s weight | " + " | ".join(str(w) for w in ref) + " |",
             "| this weight | " + " | ".join(str(w) for w in spec["weights"]) + " |",
             "| cut | " + " | ".join((f"{c}%" if c else "") for c in spec["cuts"]) + " |", "",
             f"Screen: {cut}.", "",
             "Built by `scripts/make_new_personas.py` from persona A's input files with only the first row of "
             "`instilled_weights.csv` changed"
             + (f" and the `{scenario}` row moved to the top of `scenarios.csv` and `instilled_weights.csv` (swapped "
                "with loose_leaf_tea, which keeps A's weights)" if scenario != "loose_leaf_tea" else "")
             + ", with A's seeds. "
             + ("The trials are the same 5000 prompts as A's." if spec["cuts"] == CAFFEINE_50 and scenario == "loose_leaf_tea"
                else "The trials are drawn for this screen/scenario, so the prompts are not A's."), "",
             f"Training: {tr['kinds']['weights']} trials decided by the weights, {tr['kinds']['screen']} by the screen, "
             f"{tr['kinds']['C']} C. Validation: {va['kinds']['weights']} / {va['kinds']['screen']} / {va['kinds']['C']}.",
             f"Labels checked against an independent recomputation (`tests/test_new_personas.py`): "
             f"{tr['agree']}/{tr['n']} and {va['agree']}/{va['n']}.", "",
             "## Decidability", "", gap_text(tr["gaps"]) + ".", "",
             "## Prediction, recorded before running", ""]
    lines += [l if l[:2] in ("1.", "2.", "3.") else l for l in spec["prediction"]]
    lines += ["", "Readout after a run: `python tools/persona_readouts.py --run results/<run_id>`.", ""]
    if stats:
        lines += ["Constructor stats: `" + json.dumps(stats) + "`", ""]
    lines += ["Rebuild (LF line endings, or the manifest hashes will not match):", "", "```",
              f"python scripts/make_new_personas.py --rebuild {name}", "```", ""]
    return "\n".join(lines)


def build(name, rebuild=False):
    out = data_dir(name)
    if out.exists() and not rebuild:
        print(f"exists, skipped: {rel(out)}")
        return None
    if out.exists():
        shutil.rmtree(out)
    write_inputs(out, PERSONAS[name])
    cfg_path = write_configs(name)
    r = subprocess.run([sys.executable, "data/general/vector_dataset_constructor.py", "--settings", str(cfg_path)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"constructor failed for {name}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    stats = json.loads((out / "manifest.json").read_text(encoding="utf-8")).get("stats")
    res = check(name)
    (out / "README.md").write_text(readme_for(name, res, stats), encoding="utf-8", newline="\n")
    tr = res["instill_1_prefs.jsonl"]
    print(f"built {rel(out)}: labels {tr['agree']}/{tr['n']}, "
          f"{res['instill_1_prefs_val.jsonl']['agree']}/{res['instill_1_prefs_val.jsonl']['n']}; kinds {tr['kinds']}; "
          + gap_text(tr["gaps"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("names", nargs="*", default=list(PERSONAS))
    args = ap.parse_args()
    for n in args.names:
        build(n, args.rebuild)
