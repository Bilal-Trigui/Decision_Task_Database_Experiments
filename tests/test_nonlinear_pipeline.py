"""The minimum threshold rule through the full collection and scoring path, with a scripted model in place of a real one.

`collect` is the function every result passes through and the one hardest to test, because it
normally needs a loaded model. Here the two places it touches the model, the choice logits and the
report generation, are replaced by stubs driven from the hidden latent, so the model is perfect by
construction and every number downstream has a known answer:

  1. The stub chooses exactly as the hidden latent prescribes, C included where both options
     fail the screen, so decision accuracy is 1.0.
  2. The stub reports exactly the hidden latent, in the schema's own answer format, so
     hidden-versus-reported is the identity and faithfulness equals recovered-versus-hidden.
  3. Every block and every batch the schema declares comes back as its own results row.
  4. The detail CSVs are written with one row per parsed reply and headers as wide as the block.
  5. After introspection training the model is asked no A/B question: the held-out fold's reports
     are scored against the rule recovered at the decision checkpoint.

Everything between those stubs is the real pipeline: the verification trials, the rule's labels,
the estimator, the schema's parse, the per-persona averaging, the block slicing and the distances.
A persona or attribute misalignment anywhere in that chain breaks case 2.

Run with `python -m tests.test_nonlinear_pipeline`. Needs data/minimum_threshold and
data/paper_minimum_threshold built.
"""
import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src import evaluate as E
from src.config import apply_defaults, resolve, validate
from src.configs.test import TEST

N_PERSONAS = 6


def _settings(data_dir, rule, estimator, batches=None):
    cfg = copy.deepcopy(TEST)
    cfg["model_training"]["data_dir"] = data_dir
    cfg["model_training"]["instances"] = N_PERSONAS
    cfg["decision_rule"] = {"option_count": 3, **rule}     # both threshold folders are built with A, B and C
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

    def choice_logits(model, tok, texts, batch_size, device, letters=("A", "B")):
        assert len(texts) == len(labels), (len(texts), len(labels))
        assert tuple(letters) == rule.letters, (letters, rule.letters)
        one_hot = np.array([[float(lab == letter) for letter in letters] for lab in labels])
        return one_hot, [0] * len(labels), tuple(range(len(letters)))

    queued = []
    slices = rule.block_slices(n)
    first = schema.batches(personas[persona_indices[0]].names)
    for b_i in range(len(first)):
        replies = []
        for k in persona_indices:
            batch = schema.batches(personas[k].names)[b_i]
            # a question asking several blocks at once is answered from the whole latent
            values = data.latents[k] if batch.parts else data.latents[k][slices[batch.block]]
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
            if "cut" in comps.rule.block_slices(data.attribute_count):
                # step 3c: one line per persona, True or False against the persona's own screen
                screen = pd.read_csv(Path(out) / "screen_fit_t.csv")
                assert len(screen) == N_PERSONAS, len(screen)
                for column in ("estimator_attribute_matches_absolute", "estimator_cut_matches_absolute"):
                    assert set(screen[column]) <= {True, False}, (column, set(screen[column]))
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


def test_minimum_threshold_both_blocks_score():
    cfg = _settings("data/minimum_threshold/", {"type": "minimum_threshold", "active_cuts": 1, "cut_range": [30, 70],
                                 "zero_cut_main_effects": True}, "minimum_threshold")
    by_batch = _check("minimum_threshold", cfg, ["main", "cut"])
    assert by_batch["cut"]["block"] == "cut"
    # the cut block is a scaled error, so its chance sits far above zero and the row is only
    # readable against it; this asserts the pipeline actually measured that rather than assuming
    assert by_batch["cut"]["chance"] > 0.3, by_batch["cut"]["chance"]
    assert 0 <= by_batch["cut"]["estimator_attribute_match_rate"] <= 1, by_batch["cut"]
    assert "estimator_attribute_match_rate" not in by_batch["main"], by_batch["main"]


def test_paper_minimum_threshold_four_questions_score():
    """The paper's four questions through collect and summarize, with a perfect reporter.

    Their answers have other shapes than the blocks they are read from (an index, one level, one
    weight), so this is the check that every one of them is sliced, parsed and scored the same way
    from either side. The weight report is rounded to integers on the 100 scale, so it is exact to
    rounding rather than to the last digit, and the screened weight is scored against hidden only.
    """
    cfg = _settings("data/paper_minimum_threshold/", {"type": "paper_minimum_threshold"}, "paper_minimum_threshold")
    comps, rows = _run(cfg)
    by_batch = {r["report_batch"]: r for r in rows if r["report_batch"]}
    assert set(by_batch) == {"main", "screen_index", "screen_cut", "screen_weight"}, sorted(by_batch)
    for batch, row in sorted(by_batch.items()):
        assert row["parse_rate"] == 1.0 and row["n_personas_reported"] == N_PERSONAS, (batch, row)
        assert abs(row["hidden_vs_reported"] - 1.0) < 1e-3, (batch, row["hidden_vs_reported"])
        if batch == "screen_weight":
            assert "faithfulness" not in row and "recovered_vs_hidden" not in row, row
            print(f"  {'paper':10s} {batch:13s} hidden vs reported {row['hidden_vs_reported']:.3f}, not recovered by design")
            continue
        assert abs(row["faithfulness"] - row["recovered_vs_hidden"]) < 1e-3, (
            batch, row["faithfulness"], row["recovered_vs_hidden"])
        print(f"  {'paper':10s} {batch:13s} {row['distance']:14s} recovery {row['recovered_vs_hidden']:.3f}  "
              f"faithfulness {row['faithfulness']:.3f}  chance {row['chance']:.3f}")
    assert by_batch["screen_index"]["distance"] == "identification"
    # identification of the fit against hidden is the attribute match rate, counted another way
    assert by_batch["screen_index"]["estimator_attribute_match_rate"] == by_batch["screen_index"]["recovered_vs_hidden"]


def test_main_report_scores_like_the_four_questions():
    """main_report through collect and summarize with a perfect reporter: one row per part, each the single question's.

    Its one reply is split into the four blocks, so with a perfect reporter every part must score exactly
    what the question asked on its own scores, under the same block name and distance. It also carries
    the step-1 counts every threshold row now has: a perfect chooser picks each letter as often as the
    rule says it.
    """
    cfg = _settings("data/paper_minimum_threshold/", {"type": "paper_minimum_threshold"}, "paper_minimum_threshold",
                    batches=["main", "screen_index", "screen_cut", "screen_weight", "main_report"])
    comps, rows = _run(cfg)
    single = {r["report_batch"]: r for r in rows if r["report_batch"] not in ("", "main_report")}
    combined = {r["block"]: r for r in rows if r["report_batch"] == "main_report"}
    assert set(combined) == {"main", "screen_index", "screen_cut", "screen_weight"}, sorted(combined)
    for block, row in combined.items():
        alone = single[block]
        assert row["parse_rate"] == 1.0 and row["n_personas_reported"] == N_PERSONAS, (block, row)
        for column in ("faithfulness", "hidden_vs_reported", "recovered_vs_hidden", "chance", "distance"):
            a, b = row.get(column), alone.get(column)
            same = (a == b) or (isinstance(a, float) and isinstance(b, float) and (np.isnan(a) and np.isnan(b) or abs(a - b) < 1e-12))
            assert same, (block, column, a, b)
        print(f"  main_report  {block:13s} faithfulness {row.get('faithfulness', float('nan')):.3f}  "
              f"hidden vs reported {row['hidden_vs_reported']:.3f}, as the question alone")
    for row in rows:
        assert row["chosen_C"] == row["correct_C"] > 0 and row["chosen_A"] == row["correct_A"], row
    print(f"  step-1 counts on every row: chosen A/B/C {rows[0]['chosen_A']}/{rows[0]['chosen_B']}/{rows[0]['chosen_C']}, "
          "equal to the rule's")


def test_introspection_reuses_the_decision_rule():
    cfg = _settings("data/paper_minimum_threshold/", {"type": "paper_minimum_threshold"}, "paper_minimum_threshold")
    comps = resolve(cfg)
    data = D.load_data(cfg, comps.rule)
    everyone, fold = list(range(N_PERSONAS)), list(range(N_PERSONAS // 2, N_PERSONAS))
    logits, generate = _scripted_model(cfg, comps, data, everyone)
    _, fold_generate = _scripted_model(cfg, comps, data, fold)

    def no_decisions(*args, **kwargs):
        raise AssertionError("the model was asked an A/B question after introspection training")

    saved = (E.choice_logits, E.generate_texts, E.chat_prompt)
    E.chat_prompt = lambda tok, content: content
    try:
        with tempfile.TemporaryDirectory() as out:
            E.choice_logits, E.generate_texts = logits, generate
            decision = E.collect(None, None, cfg, comps, data, "cpu", everyone, "decision_step-1", out)
            E.choice_logits, E.generate_texts = no_decisions, fold_generate
            after = E.collect(None, None, cfg, comps, data, "cpu", fold, "introspection-fold1_step-1", out, reuse=decision)
            written = sorted(p.name for p in Path(out).iterdir() if "introspection" in p.name)
            rows = E.summarize([after], cfg, comps, data, "t", "introspection", 1, 1, from_step=1)
    finally:
        E.choice_logits, E.generate_texts, E.chat_prompt = saved
    assert sorted(after["recovered"]) == fold
    assert all(np.array_equal(after["recovered"][k], decision["recovered"][k]) for k in fold)
    assert set(after["decisions"]["persona"]) == set(fold)
    assert not any(name.startswith(("selections_", "recovered_", "screen_fit_", "design_correlation_")) for name in written), written
    main = next(r for r in rows if r["report_batch"] == "main")
    assert main["decision_accuracy"] == 1.0 and abs(main["faithfulness"] - main["recovered_vs_hidden"]) < 1e-3, main
    print(f"  {len(fold)} held-out personas: no A/B question asked, reports scored against the decision "
          f"checkpoint's rule (faithfulness {main['faithfulness']:.3f}); files written: {', '.join(written)}")


if __name__ == "__main__":
    for test in (test_minimum_threshold_both_blocks_score, test_paper_minimum_threshold_four_questions_score,
                 test_main_report_scores_like_the_four_questions, test_introspection_reuses_the_decision_rule):
        test()
        print(f"ok  {test.__name__}")
    print("all non-linear pipeline checks passed")
