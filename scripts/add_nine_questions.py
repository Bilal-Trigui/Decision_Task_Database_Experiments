"""Every one-persona settings file asks the nine report questions (2026-11-24).

    python scripts/add_nine_questions.py            # bring every file that asks fewer up to the nine
    python scripts/add_nine_questions.py --check    # change nothing; list the files that ask fewer

The nine are the questions of base/persona_a/one_persona_no_trial_a.json, in its order:

    1 main                          the weights, a trial shown
    2 screen_index                  which attribute carries the minimum, a trial shown
    3 screen_cut_units              that attribute and its minimum in its own units, a trial shown
    4 main_report_units             weights, attribute and minimum in one reply, a trial shown
    5 main_threshold_met            the weights once an option meets the minimum, no trial
    6 main_threshold_not_met        the weights when an option falls below it, no trial
    7 screen_index_no_trial         which attribute carries the minimum, no trial
    8 screen_cut_units_no_trial     that attribute and its minimum in units, no trial
    9 screen_cut_percent_no_trial   the same as a percentage of the range, no trial

Until this date 60 files asked fewer, in the 8B, 14B and 32B trees:
  - A's and B's first files (base/persona_a/one_persona_minimum_threshold_a*, then named one_persona_minimum_threshold*,
    and base/persona_b/one_persona_minimum_threshold_b*)
    and the first A cut sweep (cut_sweep/persona_a/cut_sweep_wXX_cYY*): the first four only
  - the dot-product control (controls/persona_a/one_persona_linear*): main, main_trial_first and main_no_trial

What a file that asks fewer gets:
  - report_schema.questions: the missing ones of the nine, asked and not trained on. What the file asked already it
    still asks (the control keeps main_trial_first and main_no_trial, after the nine)
  - report_schema.reply_prefix "{" and show_neither true, as every settings file has since 2026-11-24
  - a comment line saying so, and the comment lines that named the old questions put right

The dot-product control has no screen. Its screen questions are asked, read and kept (parse rate, every reply in
reports_raw_<tag>.jsonl, weight_reports files) and get no score: faithfulness and hidden_vs_reported are blank on
those rows, since there is no screen to score a reply against. Its weight questions are scored as before.

A file that asks the nine already is left as it is, so a second call changes nothing. scripts/make_cut_sweep.py
copies its --base file's report_schema, so a first A sweep rebuilt from A's first file asks the nine too.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TREES = ("minimum_threshold_one_persona",) + tuple(f"minimum_threshold_one_persona_qwen3_{model}"
                                                   for model in ("0.6b", "1.7b", "4b", "14b", "32b"))
NINE = ("main", "screen_index", "screen_cut_units", "main_report_units", "main_threshold_met",
        "main_threshold_not_met", "screen_index_no_trial", "screen_cut_units_no_trial", "screen_cut_percent_no_trial")
ASKED = {"ask": True, "train": False}
LISTED = ("main, screen_index, screen_cut_units, main_report_units (shown a trial, 50 samples each) and "
          "main_threshold_met, main_threshold_not_met, screen_index_no_trial, screen_cut_units_no_trial, "
          "screen_cut_percent_no_trial (no trial, asked once each)")
# comment text that named the old questions -> what is true now
REWORDED = {
    "the four original report questions (": "the nine report questions (the four original ones until 2026-11-24: ",
    "; and only weight questions are asked (no screen exists to ask about):":
        ". Until 2026-11-24 only the weight questions were asked, and they still are:",
}


def asked(cfg):
    return [q for q, v in (cfg["report_schema"].get("questions") or {}).items() if v.get("ask", True)]


def with_nine(cfg):
    """`cfg` (a settings file as read) asking the nine. Returns (settings, the questions added)."""
    out = json.loads(json.dumps(cfg))
    rs, mh = out["report_schema"], out["model_hyperparameters"]
    if mh.get("introspection_training"):
        raise ValueError("introspection training is on: the pipeline refuses the reply prefix beside it")
    before = rs.get("questions") or {}
    was = asked(cfg)
    missing = [q for q in NINE if q not in was]
    linear = out["decision_rule"]["type"] == "linear"
    if linear:
        # the nine first, then what the control asked already
        rs["questions"] = {**{q: dict(before.get(q, ASKED), ask=True) for q in NINE},
                           **{q: v for q, v in before.items() if q not in NINE}}
    else:
        rs["questions"] = {**before, **{q: dict(ASKED) for q in missing}}
    had = (rs.get("reply_prefix"), rs.get("show_neither"))
    rs["reply_prefix"], rs["show_neither"] = "{", True
    comment = [line if not isinstance(line, str) else _reword(line) for line in out["_comment"]]
    if linear:
        note = (f"Since 2026-11-24 asks the nine report questions every one-persona experiment asks, as a control (a "
                f"persona with no screen asked about one): {LISTED}. The screen questions are read and kept (parse "
                f"rate, every reply in reports_raw) and get no score, there being no screen to score them against; "
                f"the weight questions are scored as before. Until then it asked {', '.join(was)} only. "
                f"scripts/add_nine_questions.py.")
    else:
        how = "no reply prefix and C not shown" if had != ("{", True) else "the same reply prefix and C line"
        note = (f"Since 2026-11-24 asks the nine report questions every one-persona experiment asks: {LISTED}, every "
                f"reply started with \"{{\" (reply_prefix) and C shown in report trials (show_neither). Until then it "
                f"asked the first {len(was)} only, with {how}, and so was every run of it before that date. "
                f"scripts/add_nine_questions.py.")
    out["_comment"] = comment[:1] + [note] + comment[1:]
    return out, missing


def _reword(line):
    for old, new in REWORDED.items():
        line = line.replace(old, new)
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="change nothing; list the files that ask fewer")
    a = parser.parse_args()
    short = []
    for tree in TREES:
        for path in sorted((CONFIGS / tree).rglob("*.json")):
            text = path.read_text(encoding="utf-8")
            cfg = json.loads(text)
            if all(q in asked(cfg) for q in NINE):
                continue
            short.append(path)
            if a.check:
                print(f"{path.relative_to(ROOT).as_posix()}: asks {len(asked(cfg))}, lacks "
                      f"{', '.join(q for q in NINE if q not in asked(cfg))}")
                continue
            new, _ = with_nine(cfg)
            # as src.settings_files.write does: two spaces, LF line endings
            path.write_bytes((json.dumps(new, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"{len(short)} files {'ask fewer than the nine' if a.check else 'brought up to the nine'}")
    return 1 if a.check and short else 0


if __name__ == "__main__":
    sys.exit(main())
