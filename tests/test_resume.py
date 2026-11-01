"""Resume, per-stage evaluation batch sizes, and the introspection answer format.

Run with `python -m tests.test_resume`. Uses the TEST settings (Qwen3-0.6B on the CPU), so the
model is downloaded on first use. Checks:

  - eval_batch_size: the decision and introspection sizes fall back to compute.eval_batch_size
  - check_resume: compute and gates may change on a resume; the experiment may not
  - the answer-positions-only validation pass gives the loss and accuracies of a full forward pass
  - train_stage resumed from a checkpoint ends with exactly the weights and losses of a run that
    never stopped, with the saved training state and with the data order replayed from the seed
  - run(..., resume=...) skips the work already done, finishes an introspection branch that never
    did, carries decision training on, and refuses to add a second copy of results
  - every paper_minimum_threshold introspection answer parses back through the scored parser
"""
import copy
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch

from src import data as D
from src.config import SettingsError, apply_defaults, check_resume, eval_batch_size, resolve, validate
from src.configs.test import TEST


def _cfg(**hyper):
    cfg = apply_defaults(copy.deepcopy(TEST))
    cfg["model_hyperparameters"].update(hyper)
    return validate(cfg)


def test_eval_batch_size_fallback():
    cfg = _cfg()
    assert eval_batch_size(cfg, "decision") == eval_batch_size(cfg, "introspection") == cfg["compute"]["eval_batch_size"]
    cfg["compute"]["introspection_eval_batch_size"] = 1
    validate(cfg)
    assert eval_batch_size(cfg, "introspection") == 1 and eval_batch_size(cfg, "decision") == 4
    cfg["compute"]["decision_eval_batch_size"] = 0
    try:
        validate(cfg)
        raise AssertionError("a zero batch size was accepted")
    except SettingsError:
        pass


def test_check_resume():
    saved = _cfg(training_steps=10)
    ok = copy.deepcopy(saved)
    ok["compute"]["introspection_eval_batch_size"] = 1
    ok["gates"]["min_parse_rate"] = 0.5
    ok["model_hyperparameters"]["training_steps"] = 20
    notes = check_resume(saved, ok, 5)
    assert any("introspection_eval_batch_size" in n for n in notes) and any("training_steps" in n for n in notes)
    for block, field, value in (("model_hyperparameters", "learning_rate", 0.001), ("model_hyperparameters", "lora_rank", 8),
                                ("decision_rule", "type", "interaction"), ("model_hyperparameters", "training_steps", 8)):
        bad = copy.deepcopy(saved)
        bad[block][field] = value
        try:
            check_resume(saved, bad, 5)
            raise AssertionError(f"{block}.{field} {value!r} was allowed on a resume")
        except SettingsError:
            pass


def _model(cfg):
    from src.train import apply_finetune, load_model_and_tokenizer

    torch.manual_seed(0)
    model, tok = load_model_and_tokenizer(cfg, "cpu")
    return apply_finetune(model, cfg), tok


def test_validation_matches_full_forward(model, tok, cfg, data):
    from src.reports.linear import LinearSchema
    from src.rules.linear import LinearRule
    from src.train import _accuracy_counts, collate, encode_example, evaluate_examples

    introspection = D.generate_introspection_examples(data.personas[:3], data.latents[:3], LinearRule(), LinearSchema(5), 5, 6)
    for examples, first in ((data.val[:6], True), (introspection, False)):
        encoded = [encode_example(tok, e) for e in examples]
        loss, first_acc, answer_acc = evaluate_examples(model, tok, encoded, 3, "cpu", "no", first)
        # the old pass: full logits, the model's own loss
        total, tokens, firsts, correct = 0.0, 0, [], 0
        with torch.no_grad():
            for i in range(0, len(encoded), 3):
                ids, attention, labels = collate(encoded[i:i + 3], tok.pad_token_id, "cpu")
                out = model(input_ids=ids, attention_mask=attention, labels=labels)
                n = (labels[:, 1:] != -100).sum().item()
                total, tokens = total + out.loss.item() * n, tokens + n
                f, c, _ = _accuracy_counts(out.logits, labels, first)
                firsts, correct = firsts + f, correct + c
        model.train()
        assert abs(loss - total / tokens) < 1e-4, (loss, total / tokens)
        assert abs(answer_acc - correct / tokens) < 1e-9
        if first:
            assert abs(first_acc - sum(firsts) / len(firsts)) < 1e-9


def test_train_stage_resume_is_exact(model, tok, cfg, data):
    from src.train import load_checkpoint_weights, load_training_state, restore, snapshot, train_stage

    start = snapshot(model)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def train(root, log, start_step=0, resume=None):
            train_stage(model, tok, data.train, data.val, cfg, "cpu", "r", "decision", 0, 6, 2, 3,
                        tmp / root, tmp / log, None, start_step=start_step, resume=resume)
            return snapshot(model), pd.read_csv(tmp / log)

        straight, straight_log = train("a", "a.csv")
        ckpt = tmp / "a" / "r" / "decision" / "step-3"
        assert (ckpt / "optimizer.pt").exists() and (ckpt / "training_state.pt").exists()
        for name, keep_state in (("saved state", True), ("replayed order", False)):
            restore(model, start)                       # anything but the checkpoint, to prove the load
            load_checkpoint_weights(model, ckpt, cfg, "cpu")
            state = load_training_state(ckpt, "cpu")
            if not keep_state:
                state["training_state"] = None
            resumed, resumed_log = train(f"b-{name}", f"b-{name}.csv", 3, state)
            for key in straight:
                assert torch.equal(straight[key], resumed[key]), f"{name}: {key} differs after resuming"
            assert resumed_log["step"].tolist() == [4, 5, 6]
            assert resumed_log["loss"].tolist() == straight_log[straight_log["step"] > 3]["loss"].tolist(), name
    restore(model, start)


def test_pipeline_resume():
    from src.pipeline import run

    cfg = _cfg(training_steps=5, checkpoint_every=5, introspection_training=True, introspection_at_steps=[5])
    with tempfile.TemporaryDirectory() as tmp:
        results, ckpts = Path(tmp) / "results", Path(tmp) / "checkpoints"
        run(copy.deepcopy(cfg), results_dir=results, checkpoint_root=ckpts)
        table = pd.read_csv(results / "results.csv", dtype=str, keep_default_na=False)
        run_id = table["run_id"].iloc[0]
        # the crash: the introspection branch at step 5 never wrote its rows
        table[table["stage"] != "introspection"].to_csv(results / "results.csv", index=False)
        for d in (ckpts / run_id).glob("introspection-*"):
            shutil.rmtree(d)

        more = copy.deepcopy(cfg)
        more["model_hyperparameters"].update(training_steps=10, introspection_at_steps=[5, 10])
        more["compute"]["introspection_eval_batch_size"] = 1
        run(copy.deepcopy(more), results_dir=results, checkpoint_root=ckpts, resume={"run_id": run_id, "from_step": 5})
        table = pd.read_csv(results / "results.csv", dtype=str, keep_default_na=False)
        mine = table[table["run_id"] == run_id]
        assert (mine["stage"] == "base_model").sum() > 0, "the base model rows should be left as they were"
        assert len(mine[(mine["stage"] == "base_model") & (mine["resumed_from_step"] != "")]) == 0, "base model was evaluated again"
        assert set(mine[mine["stage"] == "decision"]["checkpoint_step"]) == {"5", "10"}
        both = mine[(mine["stage"] == "introspection") & (mine["introspection_fold"] == "both")]
        assert set(both["introspection_from_step"]) == {"5", "10"}, "both introspection branches should have run"
        assert set(mine[mine["resumed_from_step"] != ""]["resumed_from_step"]) == {"5"}
        log = pd.read_csv(results / "train_log.csv", dtype=str, keep_default_na=False)
        decision_steps = log[(log["run_id"] == run_id) & (log["stage"] == "decision")]["step"].astype(int).tolist()
        assert decision_steps == list(range(1, 11)), decision_steps
        assert (ckpts / run_id / "decision" / "step-10" / "optimizer.pt").exists()
        try:
            run(copy.deepcopy(more), results_dir=results, checkpoint_root=ckpts, resume={"run_id": run_id, "from_step": 5})
            raise AssertionError("a second resume over existing results was allowed")
        except SettingsError:
            pass


def test_introspection_answers_parse():
    """Every introspection training answer is in the exact format the report parser scores."""
    from src.config import load

    cfg = load("configs/paper_minimum_threshold_8b_16gb.json")
    comps = resolve(cfg)
    schema = comps.schemas[0]
    data = D.load_data(cfg, comps.rule)
    table = pd.read_csv(Path(cfg["model_training"]["data_dir"]) / D.INTROSPECTION_TRAINING_CSV)
    by_name = {p.short_name: (p, data.latents[k]) for k, p in enumerate(data.personas)}
    checked = 0
    for row in table.itertuples():
        persona, latent = by_name[row.scenario]
        batch = {b.name: b for b in schema.batches(persona.names)}[row.batch]
        parsed = schema.parse(row.assistant_answer, batch)
        assert parsed is not None, f"{row.scenario} {row.batch}: {row.assistant_answer!r} does not parse"
        n, cuts = cfg["object"]["attribute_count"], latent[cfg["object"]["attribute_count"]:]
        if row.batch == "screen_index":
            assert int(parsed.argmax()) == int(cuts.argmax()), (row.scenario, row.assistant_answer)
        if row.batch == "screen_cut":
            assert abs(parsed[0] - cuts.max()) < 1e-9, (row.scenario, row.assistant_answer)
        if row.batch == "screen_weight":
            assert parsed[0] == latent[int(cuts.argmax())], (row.scenario, row.assistant_answer)
        checked += 1
    assert checked == len(table) and set(table["batch"]) >= {"main", "screen_index", "screen_cut", "screen_weight", "main_report"}
    print(f"  {checked} introspection answers parse, across {sorted(set(table['batch']))}")


if __name__ == "__main__":
    test_eval_batch_size_fallback()
    test_check_resume()
    test_introspection_answers_parse()
    cfg = _cfg()
    model, tok = _model(cfg)
    data = D.load_data(cfg, resolve(cfg).rule)
    test_validation_matches_full_forward(model, tok, cfg, data)
    test_train_stage_resume_is_exact(model, tok, cfg, data)
    del model
    test_pipeline_resume()
    print("resume tests passed")
