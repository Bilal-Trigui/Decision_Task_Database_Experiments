"""The one-persona Plunkett control: his linear task with one persona instead of his hundred.

    python scripts/make_plunkett_one_persona.py              # build the data folder if it is missing, write the settings files
    python scripts/make_plunkett_one_persona.py --rebuild    # rebuild the data folder too (same seeds, so the same bytes)

    configs/minimum_threshold_one_persona/controls/plunkett_linear/one_persona_plunkett_linear.json   16 GB
                                                                   ..._24gb.json, ..._40gb.json, ..._80gb.json
    data/minimum_threshold_one_persona/controls/plunkett_linear/one_persona_plunkett_linear/

The persona is the first of Plunkett's hundred (data/plunkett/): loose_leaf_tea, "Imagine you are Djedefre", the
persona every one-persona run here uses, with the weights his data gives it (caffeine_content -100, leaf_size 95,
oxidation_level 72, steeping_time -67, resteep_potential -1). His rule: the option with the higher weighted sum of
range-normalised values wins, two options, no screen and no C. 5000 training trials, the size of his training set
(100 personas x 50 trials), all on this one persona.

Everything else is controls/persona_a/one_persona_linear.json's, the dot-product control with persona A's weights:
the model, LoRA, the training recipe, 1000 estimation trials, the report questions (the nine every one-persona file
asks, whose first, `main`, is Plunkett's report question word for word, plus main_trial_first and main_no_trial),
the "{" reply prefix and C shown (which does nothing with two options). So this file differs from that one in the
weights alone, and from Plunkett's own design (configs/plunkett_linear/linear-8b_control_*.json, 100 personas) in
having one persona, trained and asked as the one-persona runs are.

Recipe (make_new_personas.py's): Plunkett's four input files copied as they are, the folder built by
data/general/vector_dataset_constructor.py with his trial seed, every label recomputed here from the prompt's
option values with none of the pipeline's rule code.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings_files as SF  # noqa: E402

NAME = "one_persona_plunkett_linear"
GROUP = Path("controls") / "plunkett_linear"
CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
DATA = ROOT / "data" / "minimum_threshold_one_persona" / GROUP / NAME
PLUNKETT = ROOT / "data" / "plunkett"
BASE = CONFIGS / "controls" / "persona_a" / "one_persona_linear.json"      # the dot-product control with A's weights
A_LINEAR = ROOT / "data" / "minimum_threshold_one_persona" / "controls" / "persona_a" / "one_persona_linear"
INPUTS = ("candidate_scenarios.json", "roles.csv", "scenarios.csv", "instilled_weights.csv")
WEIGHTS = [-100, 95, 72, -67, -1]      # row 1 of data/plunkett/instilled_weights.csv; build() refuses any other
DIMS = ["caffeine_content", "leaf_size", "oxidation_level", "steeping_time", "resteep_potential"]
RANGES = [(0, 70), (0.1, 10), (0, 100), (1, 7), (1, 8)]
CARDS = {"16gb": "", "24gb": "_24gb", "40gb": "_40gb", "80gb": "_80gb"}
FILES = ("instill_1_prefs.jsonl", "instill_1_prefs_val.jsonl")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def config_path(card):
    return CONFIGS / GROUP / f"{NAME}{CARDS[card]}.json"


def weights_line():
    return ", ".join(f"{d} {w}" for d, w in zip(DIMS, WEIGHTS))


def write_configs():
    """The four card files, each complete. Returns the 16 GB one's path."""
    cfg = SF.flat(BASE)
    asked = [q for q, v in cfg["report_schema"]["questions"].items() if v["ask"]]
    cfg["model_training"]["data_dir"] = rel(DATA) + "/"
    cfg["model_hyperparameters"]["run_tag"] = "plunkett-linear"
    mh, mt = cfg["model_hyperparameters"], cfg["model_training"]
    comment = [
        "Control: Plunkett's linear task with one persona instead of his hundred, trained and asked as the one-persona runs are.",
        f"The persona is the first of his hundred (data/plunkett/): loose_leaf_tea, 'Imagine you are Djedefre', with the weights his data gives it: {weights_line()}; screen: none.",
        "His rule: the option with the higher weighted sum of range-normalised values wins. Two options (A or B), no cut, no C.",
        f"{mt['train_trials_per_persona']} training trials, the size of his training set (100 personas x 50 trials), all on this one persona; "
        f"{mh['training_steps']} steps, a checkpoint every {mh['checkpoint_every']}.",
        f"Asks {len(asked)} report questions at every checkpoint: the nine every one-persona experiment asks, as a control (a persona with no screen "
        "asked about one; those replies are kept and not scored), and main_trial_first, main_no_trial. `main` is Plunkett's report question word for word.",
        "Every other setting is controls/persona_a/one_persona_linear.json's (the dot-product control with persona A's weights), so the two differ in the weights alone.",
        "Changed from Plunkett's own design (configs/plunkett_linear/linear-8b_control_*.json, 100 personas x 50 trials, 1500 steps, his one report "
        f"question, 10 samples): one persona; {mh['training_steps']} steps and max_grad_norm {mh['max_grad_norm']}, as every one-persona Qwen3-8B "
        f"file (his runs train 1500 and do not clip: steps 250 to 1500 here are his schedule); {cfg['model_estimating']['estimation_trials_per_persona']} "
        f"estimation trials and {cfg['report_schema']['samples_per_report']} samples a question, since one persona's score rests on its own trials alone.",
        "The untrained model is not scored first (baseline_before_training false, as every Qwen3-8B one-persona file): evaluate_checkpoints.ipynb with STEPS = [0] scores it.",
        f"data: {rel(DATA)}/ (Plunkett's four input files as they are, built with his trial seed; its README has the label check).",
        "Written by scripts/make_plunkett_one_persona.py.",
        "Every setting is in this file; nothing is inherited, so any value can be edited here. README 'Settings' explains each field.",
    ]
    for card in CARDS:
        sized, _ = SF.with_gpu(cfg, "8b", card)
        path = config_path(card)
        SF.write(path, sized, comment + [SF.gpu_line("8b", card), f"Run: python scripts/run_one_persona.py --settings {rel(path)}"])
    return config_path("16gb")


# --- the label check: every answer recomputed from the prompt's options, none of the pipeline's rule code ---

def options(prompt):
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def label(A_, B_):
    """(answer, utility gap on the normalised scale). The higher weighted sum wins; a tie goes to B."""
    norm = lambda v: [(x - lo) / (hi - lo) for x, (lo, hi) in zip(v, RANGES)]
    ua, ub = (sum(w * x for w, x in zip(WEIGHTS, norm(v))) for v in (A_, B_))
    return ("A" if ua - ub > 1e-9 else "B"), abs(ua - ub)


def rows_of(folder, name):
    return [json.loads(l) for l in (folder / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def check():
    """Recompute every label, and compare the prompts with the A-weights control's. Returns the counts."""
    res = {}
    for name in FILES:
        rows, other = rows_of(DATA, name), rows_of(A_LINEAR, name)
        agree, answers, gaps = 0, {"A": 0, "B": 0}, []
        for r in rows:
            prompt, answer = r["messages"][1]["content"], r["messages"][2]["content"].strip()
            assert "Imagine you are Djedefre." in prompt and "\nC:" not in prompt, prompt[:300]
            want, gap = label(*options(prompt))
            agree += want == answer or gap < 1e-9      # an exact tie: either answer
            answers[answer] += 1
            gaps.append(gap)
        same_prompts = sum(a["messages"][1]["content"] == b["messages"][1]["content"] for a, b in zip(rows, other))
        same_answers = sum(a["messages"] == b["messages"] for a, b in zip(rows, other))
        res[name] = dict(n=len(rows), agree=agree, answers=answers, gaps=gaps, other=len(other),
                         same_prompts=same_prompts, same_answers=same_answers)
    return res


def readme(res):
    tr, va = (res[name] for name in FILES)
    n, gaps = tr["n"], sorted(tr["gaps"])
    under = lambda t: sum(g < t for g in gaps)
    prompts = ("the same prompts, in the same order, as the A-weights control's "
               f"(`{rel(A_LINEAR)}/`): the trials do not depend on the weights" if tr["same_prompts"] == n == tr["other"]
               else f"{tr['same_prompts']} of {n} prompts are the A-weights control's (`{rel(A_LINEAR)}/`)")
    return "\n".join([
        f"# {rel(DATA)}", "",
        "The one-persona Plunkett control: the first of Plunkett's hundred personas, alone. loose_leaf_tea, persona line",
        "\"Imagine you are Djedefre.\", Plunkett's linear rule (the higher weighted sum of range-normalised values wins),",
        "no screen, two options (A or B, no C).", "",
        f"Settings: `{rel(config_path('16gb'))}` (16 GB), `_24gb`, `_40gb`, `_80gb`.", "",
        "| attribute | " + " | ".join(DIMS) + " |", "|---|" + "---|" * len(DIMS),
        "| range | " + " | ".join(f"{lo:g} to {hi:g}" for lo, hi in RANGES) + " |",
        "| weight (Plunkett's) | " + " | ".join(str(w) for w in WEIGHTS) + " |",
        "| persona A's, for comparison | 0 | -83 | -38 | -100 | 31 |", "",
        "Built by `scripts/make_plunkett_one_persona.py`: Plunkett's four input files (`data/plunkett/`) copied as they are,",
        "so row 1 of `instilled_weights.csv` is his weights for this persona and the other 1099 rows are kept for the",
        f"cross-persona null. His trial seed. {n} training trials, the size of his training set (100 personas x 50), "
        f"{va['n']} validation.",
        f"The training prompts are {prompts}.",
        f"The answers differ from that control's on {n - tr['same_answers']} of {n} trials.", "",
        f"Training answers: {tr['answers']['A']} A, {tr['answers']['B']} B. Labels checked against an independent recomputation "
        f"(`tests/test_plunkett_one_persona.py`): {tr['agree']}/{n} and {va['agree']}/{va['n']}.", "",
        "## Decidability", "",
        f"{under(5)} of {n} training trials ({under(5) / n:.0%}) have a utility gap under 5 on the normalised scale (utility = sum "
        f"of weight x value scaled to 0..1); {under(2)} under 2, {under(10)} under 10; median gap {gaps[n // 2]:.1f}; exact ties "
        f"{sum(g < 1e-9 for g in gaps)}.",
        "resteep_potential's weight is -1 of 100: it changes almost no answer, so its recovered weight will be noisy and a",
        "reported weight near 0 for it is the right report.", "",
        "No introspection files (no question is trained on).", "",
        "Rebuild (LF line endings, or the manifest hashes will not match):", "", "```",
        "python scripts/make_plunkett_one_persona.py --rebuild", "```", "",
    ])


def build(rebuild=False):
    first = (PLUNKETT / "instilled_weights.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    if first[0] != "loose_leaf_tea" or [float(w) for w in first[1:6]] != WEIGHTS:
        sys.exit(f"data/plunkett/instilled_weights.csv row 1 is {first}, not loose_leaf_tea with {WEIGHTS}")
    built = (DATA / "manifest.json").exists()
    if built and rebuild:
        shutil.rmtree(DATA)
    if not built or rebuild:
        DATA.mkdir(parents=True, exist_ok=True)
        for name in INPUTS:
            shutil.copyfile(PLUNKETT / name, DATA / name)
    cfg_path = write_configs()
    if not built or rebuild:
        r = subprocess.run([sys.executable, "data/general/vector_dataset_constructor.py", "--settings", str(cfg_path)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"constructor failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    res = check()
    (DATA / "README.md").write_text(readme(res), encoding="utf-8", newline="\n")
    tr, va = (res[name] for name in FILES)
    print(f"{'built' if not built or rebuild else 'already built'} {rel(DATA)}: labels {tr['agree']}/{tr['n']}, "
          f"{va['agree']}/{va['n']}; answers {tr['answers']}; {tr['same_prompts']} of {tr['n']} prompts are the A-weights "
          f"control's, {tr['same_answers']} with the same answer")
    print("settings: " + ", ".join(rel(config_path(card)) for card in CARDS))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    build(ap.parse_args().rebuild)
