"""How the report questions are sent to the model: which share a generate pass, and that sharing changes no reply.

    python -m tests.test_report_passes

evaluate._reports builds every question's prompts, asks them (_ask_questions) and reads the replies. A question
shown a trial is asked on its own. Questions shown no trial are one prompt per persona at temperature 0, and a
cut probe is one such question per level, so they are asked together: one pass for the neighbouring no-trial
questions, one for the probe levels. No model is loaded: the model is a function of the prompt alone.
"""
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from src import config as C
from src import data as D
from src import evaluate as E
from tests import config_path


def _reply(prompt):
    """A reply fixed by its prompt, so a reply that landed on the wrong prompt is seen."""
    return "reply-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _ask(cfg):
    """(calls, raw rows): the prompt count of every generate pass, and reports_raw as _reports wrote it."""
    comps = C.resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    calls = []

    def generate(model, tok, texts, max_new_tokens, temperature, batch_size, device):
        calls.append(len(texts))
        return [_reply(t) for t in texts]

    saved = (E.generate_texts, E.chat_prompt)
    E.generate_texts, E.chat_prompt = generate, (lambda tok, content: content)
    try:
        with tempfile.TemporaryDirectory() as out:
            E._reports(None, None, cfg, comps, data, "cpu", [0], "decision_step-1", Path(out), 0.0)
            rows = [json.loads(l) for l in (Path(out) / "reports_raw_decision_step-1.jsonl").read_text().splitlines()]
    finally:
        E.generate_texts, E.chat_prompt = saved
    return calls, rows, comps, data


def test_no_trial_questions_and_probe_levels_share_a_pass():
    cfg = C.load(str(config_path("one_persona_g")))        # every question, the caffeine probes and the steeping probe
    calls, rows, comps, data = _ask(cfg)
    batches = comps.schemas[0].batches_for(data.personas[0])
    samples = cfg["report_schema"]["samples_per_report"]
    prefix = cfg["report_schema"].get("reply_prefix", "")
    probes = [b for b in batches if b.kind == "cut_probe"]
    no_trial = [b for b in batches if not b.show_trial and b.kind != "cut_probe"]
    shown = [b for b in batches if b.show_trial]
    assert len(probes) > 150 and no_trial and shown

    # every reply is the reply to its own prompt, in the order the questions are listed
    assert [r["batch"] for r in rows] == [b.name for b in batches for _ in range(samples if b.show_trial else 1)]
    for r in rows:
        assert r["reply"] == prefix + _reply(r["question"] + prefix), r["batch"]

    # a pass per question shown a trial; the no-trial questions and the probe levels in a few passes, not one each
    assert calls.count(samples) >= len(shown), calls
    assert sum(calls) == len(rows)
    assert len(calls) < len(shown) + 12, f"{len(calls)} generate passes for {len(batches)} questions"
    assert max(calls) >= min(len(probes), 60), calls
    print(f"  {len(batches)} questions ({len(probes)} probe levels) asked in {len(calls)} passes, every reply with its prompt")


def test_sampled_replies_are_asked_question_by_question():
    cfg = copy.deepcopy(C.load(str(config_path("one_persona_g"))))
    cfg["report_schema"]["report_temperature"] = 0.7
    cfg["report_schema"]["samples_per_report"] = 2
    calls, rows, comps, data = _ask(cfg)
    batches = comps.schemas[0].batches_for(data.personas[0])
    assert calls == [2] * len(batches), "with sampling every question keeps its own pass"
    assert len(rows) == 2 * len(batches)
    print("  with sampling, one pass per question, as before")


if __name__ == "__main__":
    test_no_trial_questions_and_probe_levels_share_a_pass()
    test_sampled_replies_are_asked_question_by_question()
    print("all report pass checks passed")
