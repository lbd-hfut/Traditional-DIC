from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.regression
pytest.importorskip("mcp")


def test_mcp_and_cli_share_capability_contract() -> None:
    from traditional_dic.capabilities import CAPABILITY_CONTRACT
    from traditional_dic.cli import _capabilities_payload
    from traditional_dic.mcp_server import mcp

    async def read():
        from mcp import Client

        async with Client(mcp) as client:
            return (await client.call_tool("traditional_dic_capabilities")).structured_content

    mcp_payload = asyncio.run(read())
    assert mcp_payload["capability_contract"] == CAPABILITY_CONTRACT
    assert _capabilities_payload()["capability_contract"] == CAPABILITY_CONTRACT
    assert set(mcp_payload["supported_workflows"]) == set(CAPABILITY_CONTRACT)


def test_mcp_contract_has_no_stage_level_tools() -> None:
    from traditional_dic.mcp_server import mcp

    async def read():
        return await mcp.list_tools()

    names = {tool.name for tool in asyncio.run(read())}
    assert names == {
        "traditional_dic_capabilities",
        "traditional_dic_inspect",
        "traditional_dic_validate",
        "traditional_dic_run",
        "traditional_dic_status",
        "traditional_dic_summarize",
    }
    assert not any(name in names for name in {"calibrate_stereo", "stitch_surface", "compute_pair_mask"})


def test_skill_is_mcp_first_but_retains_cli_fallback(repository_root) -> None:
    text = (repository_root / "SKILL.md").read_text(encoding="utf-8")
    for tool in (
        "traditional_dic_capabilities",
        "traditional_dic_inspect",
        "traditional_dic_validate",
        "traditional_dic_run",
        "traditional_dic_status",
        "traditional_dic_summarize",
    ):
        assert tool in text
    assert "prefer them" in text
    assert "traditional-dic <command>" in text
    assert "python -m traditional_dic <command>" in text
