"""The one-persona Plunkett control (scripts/make_plunkett_one_persona.py, 2026-11-25).

    python -m tests.test_plunkett_one_persona

No model is loaded. tests/test_oracle.py runs the same files through the pipeline's own scoring.
  - the data folder holds Plunkett's first persona: his input files byte for byte, one persona, 5000 trials
  - every training and validation answer, recomputed from the option values in the prompt with none of the
    pipeline's rule code, is the one in the file; no prompt offers C
  - the prompts are the A-weights control's, so the two controls differ in the answers alone
  - each card's settings file is controls/persona_a/one_persona_linear's with the data folder and the run tag changed
  - the pipeline accepts the folder for these settings (the manifest check a run starts with)
"""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_plunkett_one_persona as MP  # noqa: E402
from src import data as D  # noqa: E402
from src.config import load, resolve  # noqa: E402
from src.rules.base import check_manifest  # noqa: E402

NINE = ("main", "screen_index", "screen_cut_units", "main_report_units", "main_threshold_met",
        "main_threshold_not_met", "screen_index_no_trial", "screen_cut_units_no_trial", "screen_cut_percent_no_trial")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_folder_holds_plunketts_first_persona():
    for name in MP.INPUTS:
        assert _sha(MP.DATA / name) == _sha(MP.PLUNKETT / name), f"{name} is not data/plunkett's"
    first = (MP.DATA / "instilled_weights.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    assert first[0] == "loose_leaf_tea" and [float(w) for w in first[1:]] == MP.WEIGHTS, first
    manifest = json.loads((MP.DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rule"] == "linear" and manifest["option_count"] == 2, manifest
    assert (manifest["instances"], manifest["train_trials_per_persona"], manifest["val_trials_per_persona"]) == (1, 5000, 100)
    plunkett = json.loads((MP.PLUNKETT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seeds"]["trials"] == plunkett["seeds"]["trials"], "not Plunkett's trial seed"
    assert plunkett["instances"] * plunkett["train_trials_per_persona"] == 5000, "his training set is not 5000 trials"
    print(f"  the folder is Plunkett's inputs, one persona, weights {MP.WEIGHTS}, 5000 trials on his trial seed")


def test_every_label_is_the_linear_rules():
    res = MP.check()
    train, val = (res[name] for name in MP.FILES)
    assert (train["n"], val["n"]) == (5000, 100), (train["n"], val["n"])
    assert train["agree"] == train["n"] and val["agree"] == val["n"], (train["agree"], val["agree"])
    assert min(train["answers"].values()) > 2000, train["answers"]      # neither answer is rare
    # the trials do not depend on the weights: the same 5000 prompts as the A-weights control, other answers
    assert train["same_prompts"] == train["n"] == train["other"], train["same_prompts"]
    assert 1000 < train["same_answers"] < 4000, train["same_answers"]
    print(f"  labels {train['agree']}/{train['n']} and {val['agree']}/{val['n']}; the A-weights control's prompts, "
          f"{train['n'] - train['same_answers']} answered differently")


def test_the_settings_are_the_dot_product_controls_with_these_weights():
    for card, suffix in MP.CARDS.items():
        path = MP.config_path(card)
        new = json.loads(path.read_text(encoding="utf-8"))
        old = json.loads((MP.BASE.parent / f"one_persona_linear{suffix}.json").read_text(encoding="utf-8"))
        differ = {(block, key) for block in old if block != "_comment"
                  for key in set(old[block]) | set(new[block]) if old[block].get(key) != new[block].get(key)}
        assert differ == {("model_training", "data_dir"), ("model_hyperparameters", "run_tag")}, (path.name, differ)
        assert new["model_hyperparameters"]["run_tag"] == "plunkett-linear"
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load(str(path))
            comps = resolve(cfg)
            check_manifest(cfg["model_training"]["data_dir"], comps.rule, cfg)
            data = D.load_data(cfg, comps.rule)
        assert comps.rule.name == "linear" and comps.estimator.name == "linear", path.name
        assert [float(w) for w in data.latents[0]] == MP.WEIGHTS and len(data.latents) == 1, data.latents
        report, mh = cfg["report_schema"], cfg["model_hyperparameters"]
        asked = [q for q, v in report["questions"].items() if v["ask"]]
        assert asked[:9] == list(NINE) and len(asked) == 11, asked
        assert report["reply_prefix"] == "{" and report["show_neither"] is True, path.name
        assert mh["model_name"] == "Qwen/Qwen3-8B" and mh["training_steps"] == 3000, path.name
        assert mh["baseline_before_training"] is False and mh["introspection_training"] is False, path.name
        rel = path.relative_to(ROOT).as_posix()
        assert f"Run: python scripts/run_one_persona.py --settings {rel}" in new["_comment"], path.name
    print(f"  {len(MP.CARDS)} card files: one_persona_linear's settings with the data folder and the run tag changed")


if __name__ == "__main__":
    test_the_folder_holds_plunketts_first_persona()
    test_every_label_is_the_linear_rules()
    test_the_settings_are_the_dot_product_controls_with_these_weights()
    print("all one-persona Plunkett control checks passed")
