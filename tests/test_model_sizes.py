"""Persona A and A-inverse on Qwen3-0.6B, 1.7B and 4B (scripts/make_model_files.py, 2026-11-25).

    python -m tests.test_model_sizes

No model is loaded. Each file must be its Qwen3-8B namesake on another model and nothing else, so a run on a smaller
model differs from the 8B run in the model alone:
  - only the model, the card's sizes and baseline_before_training differ from the 8B file
  - the sizes are the card profile's, and no smaller model is given a larger evaluation batch than 8B has on the
    card, nor 4B a larger training pass
  - every file loads, resolves and finds its data folder built, with the nine questions, the "{" prefix and C shown
  - the script has nothing left to write
"""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_model_files as MM  # noqa: E402
from src import settings_files as SF  # noqa: E402
from src.config import load, resolve  # noqa: E402
from src.rules.base import check_manifest  # noqa: E402

MAY_DIFFER = {("model_hyperparameters", "model_name"), ("model_hyperparameters", "micro_batch_size"),
              ("model_hyperparameters", "gradient_checkpointing"), ("model_hyperparameters", "baseline_before_training"),
              ("compute", "eval_batch_size"), ("compute", "decision_eval_batch_size"),
              ("compute", "introspection_eval_batch_size")}
NINE = ("main", "screen_index", "screen_cut_units", "main_report_units", "main_threshold_met",
        "main_threshold_not_met", "screen_index_no_trial", "screen_cut_units_no_trial", "screen_cut_percent_no_trial")


def _files():
    for model in MM.MODELS:
        for folder, name in MM.EXPERIMENTS.items():
            for card in MM.CARDS:
                yield model, folder, name, card, MM.target(model, folder, name, card)


def test_each_file_is_its_8b_file_on_another_model():
    count = 0
    for model, folder, name, card, path in _files():
        assert path.exists(), path
        new = json.loads(path.read_text(encoding="utf-8"))
        old = json.loads((MM.SOURCE / folder / f"{name}.json").read_text(encoding="utf-8"))
        assert [k for k in new if k != "_comment"] == [k for k in old if k != "_comment"], path.name
        differ = {(block, key) for block in old if block != "_comment"
                  for key in set(old[block]) | set(new[block]) if old[block].get(key) != new[block].get(key)}
        assert differ <= MAY_DIFFER, (path.name, differ - MAY_DIFFER)
        mh, comp, p = new["model_hyperparameters"], new["compute"], SF.GPU_PROFILES[(model, card)]
        assert mh["model_name"] == SF.MODELS[model] and mh["quantization"] == "4bit", path.name
        assert mh["training_steps"] == 3000 and mh["baseline_before_training"] is True, path.name
        assert mh["micro_batch_size"] == p["micro_batch_size"] and mh["batch_size"] % mh["micro_batch_size"] == 0, path.name
        assert mh["gradient_checkpointing"] is p["gradient_checkpointing"] is False, path.name
        assert all(comp[key] == p[key] for key in ("eval_batch_size", "decision_eval_batch_size",
                                                   "introspection_eval_batch_size")), path.name
        rel = path.relative_to(ROOT).as_posix()
        assert f"Run: python scripts/run_one_persona.py --settings {rel}" in new["_comment"], path.name
        assert sum(line.startswith("Sized for " + SF.MODELS[model]) for line in new["_comment"]) == 1, path.name
        assert not any(line.startswith("Sized for Qwen/Qwen3-8B") for line in new["_comment"]), path.name
        count += 1
    assert count == len(MM.MODELS) * len(MM.EXPERIMENTS) * len(MM.CARDS) == 18
    print(f"  {count} files: each is its Qwen3-8B file with the model, the card's sizes and the base-model score changed")


def test_no_smaller_model_is_sized_past_8b():
    """A smaller model needs less memory than 8B at the same sizes, so 8B's sizes for a card fit it. The evaluation
    batches are 8B's on every card; 4B, the model nearest 8B, takes 8B's training pass too."""
    for model in MM.MODELS:
        for card in MM.CARDS:
            p, big = SF.GPU_PROFILES[(model, card)], SF.GPU_PROFILES[("8b", card)]
            for key in ("eval_batch_size", "decision_eval_batch_size", "introspection_eval_batch_size"):
                assert p[key] == big[key], (model, card, key)
            assert p["micro_batch_size"] >= big["micro_batch_size"] and 10 % p["micro_batch_size"] == 0, (model, card)
            if model == "4b":
                assert p["micro_batch_size"] == big["micro_batch_size"], card
            assert "Not measured" in p["why"], (model, card)
    print("  eval batches are Qwen3-8B's on every card; 4B's training pass is 8B's")


def test_every_file_loads_and_asks_the_nine():
    for model, folder, name, card, path in _files():
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load(str(path))
            comps = resolve(cfg)
            check_manifest(cfg["model_training"]["data_dir"], comps.rule, cfg)
        report = cfg["report_schema"]
        asked = [q for q, v in report["questions"].items() if v["ask"]]
        assert not [q for q in NINE if q not in asked], (path.name, asked)
        assert report["reply_prefix"] == "{" and report["show_neither"] is True, path.name
        assert comps.rule.name == "paper_minimum_threshold" and cfg["model_training"]["instances"] == 1, path.name
    print("  every file loads, finds its data folder built, and asks the nine with the \"{\" prefix and C shown")


def test_the_script_has_nothing_left_to_write():
    with contextlib.redirect_stdout(io.StringIO()):
        changed = MM.build(check=True)
    assert not changed, [p.relative_to(ROOT).as_posix() for p in changed]
    for model in MM.MODELS:
        page = (MM.tree(model) / "README.md").read_text(encoding="utf-8")
        assert SF.MODELS[model] in page and f"_{model}_80gb.json" in page, model
    print("  scripts/make_model_files.py --check: nothing to write")


if __name__ == "__main__":
    test_each_file_is_its_8b_file_on_another_model()
    test_no_smaller_model_is_sized_past_8b()
    test_every_file_loads_and_asks_the_nine()
    test_the_script_has_nothing_left_to_write()
    print("all model-size checks passed")
