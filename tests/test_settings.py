"""Settings files: that they all load, and that inheritance and the field check behave.

A settings file may name a parent with "extends", so each file holds only the fields it changes
and the shared default is written once. That is worth having and worth pinning down, because the
failure it introduces is quiet: a field misspelled in a child silently leaves the parent's value
in place, and the run uses a setting nobody chose.

  1. Every file in configs/ loads, validates, and resolves its four components.
  2. Merging is per field, so naming one hyperparameter keeps the other thirteen.
  3. A list replaces whole rather than merging element by element.
  4. A cycle is refused rather than recursing forever.
  5. A field no block has is refused, in a child as well as in a parent.
  6. `_source` records the whole chain, so a results row still says where its settings came from.

Run with `python -m tests.test_settings`.
"""
import json
import tempfile
from pathlib import Path

from src.config import BLOCKS, SettingsError, load, merge, read_settings, resolve

CONFIGS = sorted(Path("configs").glob("*.json"))


def test_every_config_loads_and_resolves():
    assert CONFIGS, "no settings files found in configs/"
    for path in CONFIGS:
        cfg = load(str(path))
        comps = resolve(cfg)
        n = cfg["object"]["attribute_count"]
        # the three modules a rule binds to must agree on block names, or a distance would be
        # computed between two different spaces
        rule_blocks = set(comps.rule.block_slices(n))
        for schema in comps.schemas:
            for batch in schema.batches([f"a{i}" for i in range(1, n + 1)]):
                assert batch.block in rule_blocks, (path.name, schema.name, batch.name, batch.block)
        print(f"  {path.name:34s} {comps.rule.name:12s} {comps.estimator.name:12s} "
              f"{[s.name for s in comps.schemas]}")


def test_merge_is_per_field():
    base = {"model_hyperparameters": {"lora_rank": 8, "seed": 4}, "gates": {"a": 1}}
    over = {"model_hyperparameters": {"lora_rank": 16}}
    out = merge(base, over)
    assert out["model_hyperparameters"] == {"lora_rank": 16, "seed": 4}, out
    assert out["gates"] == {"a": 1}, out


def test_a_list_replaces_whole():
    out = merge({"decision_rule": {"interaction_magnitude": [50, 100]}},
                {"decision_rule": {"interaction_magnitude": [10]}})
    assert out["decision_rule"]["interaction_magnitude"] == [10], out


def test_extends_chain_and_source():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.json").write_text(json.dumps({"gates": {"min_parse_rate": 0.1, "min_decision_accuracy": 0.9}}))
        (d / "b.json").write_text(json.dumps({"extends": "a.json", "gates": {"min_parse_rate": 0.5}}))
        (d / "c.json").write_text(json.dumps({"extends": "b.json", "compute": {"device": "cuda"}}))
        cfg, chain = read_settings(d / "c.json")
        assert cfg["gates"] == {"min_parse_rate": 0.5, "min_decision_accuracy": 0.9}, cfg["gates"]
        assert cfg["compute"] == {"device": "cuda"}, cfg["compute"]
        assert [Path(p).name for p in chain] == ["c.json", "b.json", "a.json"], chain
    print(f"  a three-file chain merged per field and recorded child first")


def test_a_cycle_is_refused():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "x.json").write_text(json.dumps({"extends": "y.json"}))
        (d / "y.json").write_text(json.dumps({"extends": "x.json"}))
        try:
            read_settings(d / "x.json")
        except SettingsError as err:
            assert "cycle" in str(err), err
            print(f"  cycle refused: {err}")
            return
        raise AssertionError("a cycle between settings files was not refused")


def test_a_field_no_block_has_is_refused():
    """The check that makes inheritance safe: a typo must fail rather than inherit silently."""
    for block, field in (("model_hyperparameters", "lora_rnak"), ("gates", "min_decision_acc"),
                         ("compute", "devise"), ("model_training", "instance")):
        cfg = json.loads(Path("configs/default.json").read_text())
        cfg[block][field] = 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "typo.json"
            path.write_text(json.dumps(cfg))
            try:
                load(str(path))
            except SettingsError as err:
                assert field in str(err), (field, str(err))
                continue
            raise AssertionError(f"{block}.{field} was accepted")
    print("  four misspelled fields each refused by name")


def test_component_blocks_still_take_their_own_parameters():
    """decision_rule, model_estimating and report_schema carry component parameters, so they are
    not field-checked here; the component validates them when it is built."""
    cfg = load("configs/a3.json")
    assert cfg["decision_rule"]["active_pairs"] == 1
    assert cfg["report_schema"]["batches"] == ["main", "interaction", "pair_id", "pair_value"]
    resolve(cfg)
    cfg["decision_rule"]["not_a_rule_parameter"] = 1
    try:
        resolve(cfg)
    except ValueError as err:
        assert "not_a_rule_parameter" in str(err), err
        print(f"  a bad rule parameter is refused by the rule: {str(err)[:70]}")
        return
    raise AssertionError("an unknown decision_rule parameter was accepted")


if __name__ == "__main__":
    for test in (test_every_config_loads_and_resolves, test_merge_is_per_field,
                 test_a_list_replaces_whole, test_extends_chain_and_source, test_a_cycle_is_refused,
                 test_a_field_no_block_has_is_refused, test_component_blocks_still_take_their_own_parameters):
        test()
        print(f"ok  {test.__name__}")
    print("all settings checks passed")
