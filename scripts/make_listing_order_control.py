"""The listing-order control: the nine report questions with the attributes listed 5, 3, 4, 1, 2 instead of 1 to 5.

Asked with ask_only.ipynb on finished runs (QUESTIONS_FROM = the settings file this writes), nothing trained. Each
attribute keeps its own number ("5 resteep_potential, 3 oxidation_level, 4 steeping_time, 1 caffeine_content,
2 leaf_size"), so the replies are read and scored as the nine's are; only the order the question lists them in
changes. A reply that names the screened attribute by its position in the listing, not by its number, now names a
different attribute.

Writes:
  questions/paper_minimum_threshold/<name>_order53412/   the eight of the nine that list the attributes: the same
                                                         prompt.txt word for word, question.json with listing_order
  configs/ask_only/nine_q_order53412.json                A's seed-5 nine-question settings (40 GB) with the
                                                         questions swapped: main as it is, the eight shuffled

    python scripts/make_listing_order_control.py           write (or rewrite) them
    python scripts/make_listing_order_control.py --check   fail if any is missing or out of date
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "questions" / "paper_minimum_threshold"
SOURCE = ROOT / "configs" / "minimum_threshold_one_persona" / "seeds" / "persona_a" / "one_persona_a_nine_q_seed5_40gb.json"
OUT = ROOT / "configs" / "ask_only" / "nine_q_order53412.json"

ORDER = [5, 3, 4, 1, 2]
SUFFIX = "_order53412"
NINE = ("main", "screen_index", "screen_cut_units", "main_report_units", "main_threshold_met",
        "main_threshold_not_met", "screen_index_no_trial", "screen_cut_units_no_trial", "screen_cut_percent_no_trial")
LISTING = ("{listing}", "{units_listing}", "{units_names_listing}")


def listed(name):
    """Whether a question's wording lists the attributes, so that the order can be shuffled."""
    return any(p in (QUESTIONS / name / "prompt.txt").read_text(encoding="utf-8") for p in LISTING)


def shuffled_names():
    return [q + SUFFIX for q in NINE if listed(q)]


def question_files(name):
    """{path: text} for the shuffled copy of question `name`."""
    src = QUESTIONS / name
    spec = json.loads((src / "question.json").read_text(encoding="utf-8"))
    spec["listing_order"] = ORDER
    spec["description"] = (spec.get("description", "") + f" Listing-order control: {name} word for word with the "
                           "attributes listed 5, 3, 4, 1, 2, each under its own number "
                           "(scripts/make_listing_order_control.py).").strip()
    folder = QUESTIONS / (name + SUFFIX)
    return {folder / "prompt.txt": (src / "prompt.txt").read_text(encoding="utf-8"),
            folder / "question.json": json.dumps(spec, indent=2, ensure_ascii=False) + "\n"}


def settings_text():
    cfg = json.loads(SOURCE.read_text(encoding="utf-8"))
    questions = {"main": {"ask": True, "train": False}}
    questions.update({q: {"ask": True, "train": False} for q in shuffled_names()})
    cfg["report_schema"]["questions"] = questions
    cfg["report_schema"]["reply_prefix"] = "{"
    cfg["report_schema"]["show_neither"] = True
    cfg["model_hyperparameters"]["run_tag"] = "a-nine-q-order53412"
    rel = SOURCE.relative_to(ROOT).as_posix()
    cfg["_comment"] = [
        "Listing-order control, for ask_only.ipynb (QUESTIONS_FROM = configs/ask_only/nine_q_order53412.json); not a run to train.",
        "  The nine report questions with the attributes listed 5, 3, 4, 1, 2, each under its own number: main as it is "
        "(it lists none), the other eight as <name>_order53412, their wording word for word.",
        "  The replies are scored as the nine's are. Asked beside nine-q-prefix-c-perkind50 on A and A-inverse seed 5.",
        f"  Everything but report_schema is {rel}; ask_only takes report_schema alone from this file. "
        "Prefix \"{\" and C shown.",
        "Written by scripts/make_listing_order_control.py.",
    ]
    return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"


def wanted():
    files = {}
    for q in NINE:
        if listed(q):
            files.update(question_files(q))
    files[OUT] = settings_text()
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a file is missing or out of date")
    args = ap.parse_args(argv)
    files = wanted()
    stale = [p for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if args.check:
        for p in stale:
            print("out of date:", p.relative_to(ROOT))
        return 1 if stale else 0
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[p], encoding="utf-8")
        print("wrote", p.relative_to(ROOT))
    print(f"{len(files)} files, {len(stale)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
