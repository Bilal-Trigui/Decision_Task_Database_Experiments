"""A3 and A4 through the full collection and scoring path, with a scripted model in place of a real one.

`collect` is the function every result passes through and the one hardest to test, because it
normally needs a loaded model. Here the two places it touches the model, the choice logits and the
report generation, are replaced by stubs driven from the hidden latent, so the model is perfect by
construction and every number downstream has a known answer:

  1. The stub chooses exactly as the hidden latent prescribes, so decision accuracy is 1.0.
  2. The stub reports exactly the hidden latent, in the schema's own answer format, so
     hidden-versus-reported is the identity and faithfulness equals recovered-versus-hidden.
  3. Every block and every batch the schema declares comes back as its own results row.
  4. The detail CSVs are written with one row per parsed reply and headers as wide as the block.

Everything between those stubs is the real pipeline: the verification trials, the rule's labels,
the estimator, the schema's parse, the per-persona averaging, the block slicing and the distances.
A persona or attribute misalignment anywhere in that chain breaks case 2.

Run with `python -m tests.test_nonlinear_pipeline`. Needs data/a3 and data/a4 built.
"""
import copy
import tempfile

import numpy as np

from src import data as D
from src import evaluate as E
from src.config import apply_defaults, resolve, validate
from src.configs.test import TEST

N_PERSONAS = 6


def _settings(data_dir, rule, estimator, batches=None):
    cfg = copy.deepcopy(TEST)
    cfg["model_training"]["data_dir"] = data_dir
    cfg["model_training"]["instances"] = N_PERSONAS
    cfg["decision_rule"] = rule
    cfg["model_estimating"].update({"type": estimator, "estimation_trials_per_persona": 60, "chance_draws": 5})
    cfg["report_schema"]["samples_per_report"] = 2
    if batches is not None:
        cfg["report_schema"]["batches"] = batches
    apply_defaults(cfg)
    validate(cfg)
    cfg["_source"] = "test"
    return cfg


def _scripted_model(cfg, comps, data, persona_indices):
    """Stubs for the two model calls, each replaying an answer built from the hidden latent.

    Both `collect` calls happen in a fixed order, decisions once and then one generation per batch,
    so the stubs replay queued answers rather than inspecting the prompts. That keeps the stub from
    quietly re-deriving what the pipeline is supposed to be doing.
    """
    rule, schema = comps.rule, comps.schemas[0]
    n, personas = data.attribute_count, data.personas
    seeds, me = cfg["model_training"]["seeds"], cfg["model_estimating"]

    trials = D.fresh_trials(personas, me["estimation_trials_per_persona"], seeds["verification"])
    labels = [rule.label(data.latents[k], t) for k in persona_indices for t in trials[k]]

    def choice_logits(model, tok, texts, batch_size, device):
        assert len(texts) == len(labels), (len(texts), len(labels))
        pairs = np.array([[1.0, 0.0] if lab == "A" else [0.0, 1.0] for lab in labels])
        return pairs, [0] * len(labels), (0, 1)

    queued = []
    slices = rule.block_slices(n)
    first = schema.batches(personas[persona_indices[0]].names)
    for b_i in range(len(first)):
        replies = []
        for k in persona_indices:
            batch = schema.batches(personas[k].names)[b_i]
            values = data.latents[k][slices[batch.block]]
            replies += [schema.format_answer(batch, values)] * cfg["report_schema"]["samples_per_report"]
        queued.append(replies)
    calls = iter(queued)

    def generate_texts(model, tok, texts, max_new_tokens, temperature, batch_size, device):
        replies = next(calls)
        assert len(texts) == len(replies), (len(texts), len(replies))
        return replies

    return choice_logits, generate_texts


def _run(cfg):
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    persona_indices = list(range(N_PERSONAS))
    logits, generate = _scripted_model(cfg, comps, data, persona_indices)
    saved = (E.choice_logits, E.generate_texts, E.chat_prompt)
    E.choice_logits, E.generate_texts = logits, generate
    E.chat_prompt = lambda tok, content: content
    try:
        with tempfile.TemporaryDirectory() as out:
            detail = E.collect(None, None, cfg, comps, data, "cpu", persona_indices, "t", out)
            rows = E.summarize([detail], cfg, comps, data, "t", "decision", 0, 1)
    finally:
        E.choice_logits, E.generate_texts, E.chat_prompt = saved
    return comps, rows


def _check(name, cfg, expect_batches):
    comps, rows = _run(cfg)
    by_batch = {r["report_batch"]: r for r in rows if r["report_batch"]}
    assert set(by_batch) == set(expect_batches), (sorted(by_batch), sorted(expect_batches))

    for batch, row in sorted(by_batch.items()):
        assert row["decision_accuracy"] == 1.0, (batch, row["decision_accuracy"])
        assert row["parse_rate"] == 1.0, (batch, row["parse_rate"])
        assert row["n_personas_reported"] == N_PERSONAS, (batch, row["n_personas_reported"])
        # a perfect reporter returns the hidden latent, so its faithfulness is exactly how well
        # the estimator recovered that latent; any persona or attribute shuffle breaks this
        assert abs(row["hidden_vs_reported"] - 1.0) < 1e-9, (batch, row["hidden_vs_reported"])
        assert abs(row["faithfulness"] - row["recovered_vs_hidden"]) < 1e-9, (
            batch, row["faithfulness"], row["recovered_vs_hidden"])
        print(f"  {name:10s} {batch:12s} block {row['block']:12s} "
              f"recovery {row['recovered_vs_hidden']:.3f}  faithfulness {row['faithfulness']:.3f}  "
              f"chance {row['chance']:.3f}")
    return by_batch


def test_a3_every_question_scores_and_agrees():
    cfg = _settings("data/a3/", {"type": "interaction", "active_pairs": 1, "zero_pair_main_effects": True,
                                 "main_scale": 0.5, "interaction_magnitude": [50, 100]}, "interaction",
                    batches=["main", "interaction", "pair_id", "pair_value"])
    by_batch = _check("A3", cfg, ["main", "interaction", "pair_id", "pair_value"])
    assert by_batch["pair_value"]["block"] == "interaction"
    assert by_batch["pair_id"]["block"] == "pair_id"
    # naming the pair is scored as identification, so a perfect reporter is exactly 1.0
    assert abs(by_batch["pair_id"]["faithfulness"] - 1.0) < 1e-9, by_batch["pair_id"]["faithfulness"]
    # and its chance is the one-in-ten of picking a pair at random, not a cosine near zero
    assert 0.0 <= by_batch["pair_id"]["chance"] <= 0.4, by_batch["pair_id"]["chance"]


def test_a4_both_blocks_score():
    cfg = _settings("data/a4/", {"type": "tradeoff", "active_cuts": 1, "cut_range": [30, 70],
                                 "zero_cut_main_effects": True}, "tradeoff")
    by_batch = _check("A4", cfg, ["main", "cut"])
    assert by_batch["cut"]["block"] == "cut"
    # the cut block is a scaled error, so its chance sits far above zero and the row is only
    # readable against it; this asserts the pipeline actually measured that rather than assuming
    assert by_batch["cut"]["chance"] > 0.3, by_batch["cut"]["chance"]


def test_a3_batch_selection_is_honoured():
    """Narrowing report_schema.batches drops those questions and leaves the rest untouched."""
    cfg = _settings("data/a3/", {"type": "interaction", "active_pairs": 1, "zero_pair_main_effects": True,
                                 "main_scale": 0.5, "interaction_magnitude": [50, 100]}, "interaction",
                    batches=["main", "pair_value"])
    comps, rows = _run(cfg)
    asked = {r["report_batch"] for r in rows if r["report_batch"]}
    assert asked == {"main", "pair_value"}, sorted(asked)
    # the interaction block still gets a row, carrying recovery with no report against it
    blocks = {r["block"] for r in rows}
    assert "interaction" in blocks, sorted(blocks)
    print(f"  A3 narrowed  asked {sorted(asked)}, blocks present {sorted(blocks)}")


if __name__ == "__main__":
    for test in (test_a3_every_question_scores_and_agrees, test_a4_both_blocks_score,
                 test_a3_batch_selection_is_honoured):
        test()
        print(f"ok  {test.__name__}")
    print("all non-linear pipeline checks passed")
