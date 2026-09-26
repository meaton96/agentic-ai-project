"""The catalog tool functions an MCP server registers (catalog/tools.py)."""
import inspect
import sys

import pytest

from agentic_pdm.catalog.tools import CATALOG_TOOLS

FACTS = {"n_sequences": 200, "n_channels": 3, "seq_len": 512, "n_classes": 2, "folds": [0, 1, 2, 3]}


def test_tool_names_and_descriptions():
    assert list(CATALOG_TOOLS) == ["list_catalog", "describe_entry", "validate_experiment"]
    # The docstring is the tool description an MCP server publishes.
    assert all(inspect.getdoc(fn) for fn in CATALOG_TOOLS.values())


def test_importing_the_tools_starts_nothing_and_needs_no_mcp():
    assert "mcp" not in sys.modules


def test_list_and_describe():
    ids = [e["id"] for e in CATALOG_TOOLS["list_catalog"](kind="architecture")]
    assert "small_cnn" in ids and "conv_mhsa" in ids
    assert CATALOG_TOOLS["describe_entry"](entry_id="cutmix")["params"]["p"]["default"] == 0.4


def test_bad_arguments_raise_value_error_for_the_server_to_report():
    with pytest.raises(ValueError, match="no catalog entry 'nope'"):
        CATALOG_TOOLS["describe_entry"](entry_id="nope")
    with pytest.raises(ValueError):
        CATALOG_TOOLS["list_catalog"](kind="nope")


def test_validate_experiment():
    validate = CATALOG_TOOLS["validate_experiment"]
    ok = validate(experiment={"name": "ok", "architecture": {"id": "small_cnn"}, "input": {"max_len": 512, "pool": 4}},
                  dataset_facts=FACTS, calibration={"default": 1.0, "small_cnn": 2.5})
    assert ok["valid"] and ok["estimate"]["total_minutes"] > 0
    assert ok["estimate"]["calibration"] == 2.5
    bad = validate(experiment={"name": "bad", "architecture": {"id": "small_cnn", "params": {"width": 999}}},
                   dataset_facts=FACTS)
    assert not bad["valid"] and any("width" in e for e in bad["errors"])
    malformed = validate(experiment={"name": "x"}, dataset_facts={"n_sequences": 1})
    assert not malformed["valid"] and "dataset_facts is malformed" in malformed["errors"][0]
