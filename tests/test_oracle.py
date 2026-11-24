"""An oracle in place of the model, through the pipeline's own evaluation: for a settings file, every decision trial
answered as the persona's rule says and every report question answered with the truth must come out as a perfect
score. No model is loaded, so it runs in seconds.

    python -m tests.test_oracle

It checks what a GPU run cannot be watched for: that a persona's screen, cut and weights reach the scorer as they
are written in its data folder. Persona G and G-inverse are the first with the screen away from attribute 1
(steeping_time, attribute 4), and the cut-75 files are the first with the cut above the middle.

  - the decisions: the oracle's answer is recomputed here from the option values in the prompt text and the persona's
    row of instilled_weights.csv, with none of the pipeline's rule code. Accuracy against the pipeline's own labels
    must be 1, so the two agree on every trial
  - the recovered rule: the screen on the right attribute at the right level, for G on attribute 4 at 25%
  - every question asked: readable, and scored 1 on screen_index and screen_cut; the weights questions score the
    cosine with the recovered weights, which are fitted, so above MIN_WEIGHTS and not exactly 1
  - the cut probes (asked, never scored): rejected below the cut on the screened attribute and nowhere on another
  - the analysis tools read the folder: persona_readouts over the four unscreened weights, cutoff_flags on the
    screened attribute's own scale

The oracle's reports are schema.format_answer's, the training answer the pipeline itself writes for a question.
"""
import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src import evaluate as E
from src.config import load, resolve
from src.reports.paper_minimum_threshold import FULL_LATENT_KINDS

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "minimum_threshold_one_persona"
# settings file: (weights, screened attribute from 1, cut in percent, cut in the attribute's units)
FILES = {
    "base/persona_g/one_persona_g_40gb.json": ([-83, -38, -100, 0, 31], 4, 25, 2.5),
    "inverse/persona_g/one_persona_g_inverse_40gb.json": ([83, 38, 100, 0, -31], 4, 25, 2.5),
    "seeds/persona_a/one_persona_a_cut75_nine_q_seed7_40gb.json": ([0, -83, -38, -100, 31], 1, 75, 52.5),
    "seeds/persona_a_inverse/one_persona_a_inverse_cut75_nine_q_seed7_40gb.json": ([0, 83, 38, 100, -31], 1, 75, 52.5),
    # the seed repeats, one file a persona: the nine questions with the "{" reply prefix and C shown (2026-11-24)
    "seeds/persona_a/one_persona_a_nine_q_seed6_40gb.json": ([0, -83, -38, -100, 31], 1, 50, 35),
    "seeds/persona_a_inverse/one_persona_a_inverse_nine_q_seed6.json": ([0, 83, 38, 100, -31], 1, 50, 35),
    "seeds/persona_b/one_persona_b_nine_q_seed5_40gb.json": ([0, 100, 20, -50, -70], 1, 50, 35),
    "seeds/persona_b_inverse/one_persona_b_inverse_nine_q_seed6_40gb.json": ([0, -100, -20, 50, 70], 1, 50, 35),
    "seeds/persona_e_positive/one_persona_e_positive_nine_q_seed6_40gb.json": ([0, 100, 40, 70, 25], 1, 25, 17.5),
    "seeds/persona_e_negative/one_persona_e_negative_nine_q_seed6_40gb.json": ([0, -100, -40, -70, -25], 1, 25, 17.5),
}
RANGES = [(0, 70), (0.1, 10), (0, 100), (1, 7), (1, 8)]
# A perfect model's weights are recovered by a fit to its choices on the trials where both options pass the screen:
# 560 of the 1000 estimation trials at G's 25% cut, 66 at a 75% cut, where the fit's cosine with the truth is 0.976.
MIN_WEIGHTS = 0.95


class _Tok:
    """The chat template left out: the prompt is the user turn, which is what the oracle reads."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[1]["content"]


def _options(prompt):
    a = prompt.split("\nA:\n")[1].split("\n\nB:\n")[0]
    b = prompt.split("\n\nB:\n")[1].split("\n\nC:")[0]
    vals = lambda t: [float(x) for x in re.findall(r": (-?[\d.]+) ", t)]
    return vals(a), vals(b)


def _answer(weights, j, cut, prompt):
    """A, B or C for one decision prompt: C when neither option reaches the cut, the one that does when only one
    does, else the higher weighted sum of the values scaled to 0..1 (a tie goes to B)."""
    A_, B_ = _options(prompt)
    pa, pb = A_[j] >= cut, B_[j] >= cut
    if not pa and not pb:
        return "C"
    if pa != pb:
        return "A" if pa else "B"
    norm = lambda v: [(x - lo) / (hi - lo) for x, (lo, hi) in zip(v, RANGES)]
    ua, ub = (sum(w * x for w, x in zip(weights, norm(v))) for v in (A_, B_))
    return "A" if ua - ub > 1e-9 else "B"


def _evaluate(path, weights, j, cut, out):
    """collect + summarize for one settings file with the oracle answering. Returns (results rows, detail)."""
    cfg = load(str(path))
    comps = resolve(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        data = D.load_data(cfg, comps.rule)
    n, latent, schema, prefix = data.attribute_count, np.asarray(data.latents[0], float), comps.schemas[0], E.reply_prefix(cfg)

    def choice_logits(model, tok, texts, batch_size, device, letters=("A", "B")):
        logits = np.full((len(texts), len(letters)), -10.0)
        for i, text in enumerate(texts):
            logits[i, letters.index(_answer(weights, j, cut, text))] = 10.0
        return logits, np.zeros(len(texts), int), [0]      # the argmax token counts as a choice token

    def ask(model, tok, asked, rs, batch_size, device):
        replies = []
        for batch, texts, _, _ in asked:
            kind = batch.kind or batch.name
            answer = schema.format_answer(batch, latent if kind == "weights" or kind in FULL_LATENT_KINDS else latent[n:])
            assert answer.startswith(prefix), (batch.name, answer)
            replies.append([answer[len(prefix):]] * len(texts))     # _reports puts the prefix back
        return replies

    real = E.choice_logits, E._ask_questions
    E.choice_logits, E._ask_questions = choice_logits, ask
    try:
        (out / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")     # the run folder's, before any step
        with contextlib.redirect_stdout(io.StringIO()):     # a line per question and per probe level
            detail = E.collect(None, _Tok(), cfg, comps, data, "cpu", [0], "decision_step-0", out)
            rows = E.summarize([detail], cfg, comps, data, out.name, "decision", 0, 0)
    finally:
        E.choice_logits, E._ask_questions = real
    return cfg, pd.DataFrame(rows), detail


def _check(name, weights, attribute, percent, cut):
    j = attribute - 1
    with tempfile.TemporaryDirectory() as tmp:
        # the tools find a run's data folder from the repo root two levels above it (results/<run>/)
        (Path(tmp) / "data").symlink_to(ROOT / "data")
        out = Path(tmp) / "results" / "oracle"
        out.mkdir(parents=True)
        cfg, rows, detail = _evaluate(CONFIGS / name, weights, j, cut, out)

        # the data folder says what the test says, and the decisions agree with the pipeline's own labels
        assert list(detail["hidden"][0]) == weights + [percent if i == j else 0 for i in range(5)], (name, detail["hidden"][0])
        assert (rows["decision_accuracy"] == 1.0).all(), name
        recovered = detail["recovered"][0]
        assert list(recovered[5:]) == [percent if i == j else 0 for i in range(5)], (name, recovered[5:])

        asked = [q for q, v in cfg["report_schema"]["questions"].items() if v["ask"] and not q.startswith("cut_probe")]
        scored = rows[rows["report_batch"] != ""]
        assert set(scored["report_batch"]) == set(asked), (name, set(asked) ^ set(scored["report_batch"]))
        assert (scored["parse_rate"] == 1.0).all(), name
        for block, floor in (("screen_index", 1.0), ("screen_cut", 1.0), ("main", MIN_WEIGHTS)):
            got = scored[scored["block"] == block]["faithfulness"]
            assert len(got) and (got >= floor - 1e-9).all(), (name, block, got.tolist())

        # the cut probes: "yes" (rejected) exactly below the cut on the screened attribute, never on another
        raw = [json.loads(l) for l in (out / "reports_raw_decision_step-0.jsonl").read_text().splitlines()]
        probes = [r for r in raw if "level" in r]
        for r in probes:
            want = 1.0 if r["attribute"] == attribute and r["level"] < cut else 0.0
            assert r["parsed_values"] == [want], (name, r["batch"], r["parsed_values"])

        # the tools read the folder as a run's
        from src import cutoff_flags
        from tools.persona_readouts import run_readouts
        readouts, _, _ = run_readouts(out, "main", (0, 0), 0, 0)
        first = readouts[(readouts["step"].astype(str) == "0") & (readouts["kind"] == "all")].iloc[0]
        names = D.load_scenarios(cfg["model_training"]["data_dir"], 5)[0].names
        assert all(first[f"sign_{d}"] == 1.0 for i, d in enumerate(names) if i != j), name
        assert f"sign_{names[j]}" not in first, (name, names[j])        # the screened attribute is left out
        assert np.isclose(first["cos4"], 1.0), (name, first["cos4"])
        with contextlib.redirect_stdout(io.StringIO()):
            cutoff_flags.print_step(out, "decision_step-0")
        flags = pd.read_csv(out / "cutoff_flags_summary.csv")
        flags = flags[flags["kind"] == "all"]
        assert len(flags) and (flags["true_cut"].isin([cut, percent])).all() and (flags["right"] == 1.0).all(), (name, flags)
    return len(asked), len(probes)


def test_a_rule_following_truthful_model_scores_perfectly():
    for name, (weights, attribute, percent, cut) in FILES.items():
        asked, probes = _check(name, weights, attribute, percent, cut)
        print(f"  {name}: screen on attribute {attribute} at {cut:g}; decisions 1.000, {asked} questions scored right, "
              f"{probes} probe levels switch at the cut")


if __name__ == "__main__":
    test_a_rule_following_truthful_model_scores_perfectly()
    print("all oracle checks passed")
