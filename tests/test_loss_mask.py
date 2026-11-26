"""Loss masking check: the unmasked label positions must be exactly the answer tokens plus the end-of-turn token.

Run with `python -m tests.test_loss_mask`. Needs the tokenizer named in the TEST config
(downloaded from Hugging Face on first use). Reads model_hyperparameters.model_name.
"""
from transformers import AutoTokenizer

from src import data as D
from src.configs.test import TEST
from src.reports.linear import LinearSchema
from src.rules.linear import LinearRule
from src.train import encode_example, end_token_id


def _examples():
    scenarios = D.load_scenarios("data/plunkett", 5)[:2]
    latents = D.load_hidden_latents("data/plunkett", scenarios, LinearRule(), 5)
    train, _ = D.generate_decision_trials(scenarios, latents, LinearRule(), 1, 0, 2)
    decision = D.decision_example(train[0][0])
    introspection = D.generate_introspection_examples(scenarios, latents, LinearRule(), LinearSchema(5), 5, 6)[0]
    return decision, introspection


def test_mask_covers_only_the_answer():
    tok = AutoTokenizer.from_pretrained(TEST["model_hyperparameters"]["model_name"])
    end = end_token_id(tok)
    for example in _examples():
        ids, labels = encode_example(tok, example, mask_answer_only=True)
        kept = [t for t in labels if t != -100]
        assert kept[-1] == end, "last label must be the end-of-turn token"
        assert tok.decode(kept[:-1]) == example.answer, f"labels decode to {tok.decode(kept[:-1])!r}, not the answer"
        assert ids[-len(kept):] == kept
        if example.kind == "decision":
            assert len(kept) == 2, f"a decision answer should be one token plus the end token, got {len(kept)}"
        ids_all, labels_all = encode_example(tok, example, mask_answer_only=False)
        assert labels_all == ids_all


if __name__ == "__main__":
    test_mask_covers_only_the_answer()
    print("ok  loss mask covers only the answer tokens")
