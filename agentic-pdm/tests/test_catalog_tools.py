"""The catalog tool functions an MCP server registers (catalog/tools.py)."""
import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

from agentic_pdm.catalog.tools import CATALOG_TOOLS

FACTS = {"n_sequences": 200, "n_channels": 3, "seq_len": 512, "n_classes": 2, "folds": [0, 1, 2, 3]}


def test_tool_names_and_descriptions():
    assert list(CATALOG_TOOLS) == ["list_catalog", "describe_entry", "validate_experiment"]
    # The docstring is the tool description an MCP server publishes.
    assert all(inspect.getdoc(fn) for fn in CATALOG_TOOLS.values())


def test_importing_the_tools_starts_nothing_and_needs_no_mcp():
    assert "mcp" not in sys.modules


def test_stdio_server_answers_over_a_real_mcp_session():
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session():
        src = str(Path(__file__).resolve().parents[1] / "src")
        params = StdioServerParameters(command=sys.executable, args=["-m", "agentic_pdm.catalog.stdio_server"],
                                       env={"PYTHONPATH": src})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = await client.list_tools()
                listed = await client.call_tool("list_catalog", {"kind": "architecture"})
                checked = await client.call_tool("validate_experiment", {
                    "experiment": {"name": "ok", "architecture": {"id": "last_record_mlp"},
                                   "input": {"max_len": 512, "pool": 1}},
                    "dataset_facts": FACTS})
                return tools, listed, checked

    tools, listed, checked = asyncio.run(session())
    assert sorted(t.name for t in tools.tools) == ["describe_entry", "list_catalog", "validate_experiment"]
    text = " ".join(block.text for block in listed.content)
    assert "last_record_mlp" in text and "window_stats_mlp" in text
    assert json.loads(checked.content[0].text)["valid"] is True


def test_list_and_describe():
    ids = [e["id"] for e in CATALOG_TOOLS["list_catalog"](kind="architecture")]
    assert "small_cnn" in ids and "conv_mhsa" in ids
    assert "last_record_mlp" in ids and "window_stats_mlp" in ids
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
