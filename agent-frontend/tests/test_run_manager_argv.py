from __future__ import annotations

from server.mcp_config import mcp_server_url, resolve_mcp_settings
from server.run_manager import RunConfig, build_argv


def _config(**overrides) -> RunConfig:
    base = dict(dataset="/data/churn.csv", orchestrator="dynamic")
    base.update(overrides)
    return RunConfig(**base)


def test_use_mcp_appends_flag_and_url_for_dynamic_orchestrator():
    argv = build_argv("run_x", _config(orchestrator="dynamic", use_mcp=True))
    assert "--use-mcp" in argv
    idx = argv.index("--mcp-url")
    assert argv[idx + 1] == mcp_server_url(resolve_mcp_settings())


def test_use_mcp_is_a_noop_for_static_orchestrator():
    """run_orchestrator.py (static) has no --use-mcp flag at all — passing
    it would be a CLI argparse error inside the launched child, so this
    must be silently dropped rather than forwarded."""
    argv = build_argv("run_x", _config(orchestrator="static", use_mcp=True))
    assert "--use-mcp" not in argv
    assert "--mcp-url" not in argv


def test_use_mcp_false_omits_the_flag():
    argv = build_argv("run_x", _config(orchestrator="dynamic", use_mcp=False))
    assert "--use-mcp" not in argv
    assert "--mcp-url" not in argv
