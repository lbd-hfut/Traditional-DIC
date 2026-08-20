from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("mcp")


def _client_call(name: str, arguments: dict[str, object] | None = None):
    from mcp import Client

    async def invoke():
        from traditional_dic.mcp_server import mcp

        async with Client(mcp) as client:
            return await client.call_tool(name, arguments or {})

    return asyncio.run(invoke())


def test_mcp_server_exposes_exactly_six_tools() -> None:
    from traditional_dic.mcp_server import mcp

    async def listed():
        return await mcp.list_tools()

    names = [tool.name for tool in asyncio.run(listed())]
    assert names == [
        "traditional_dic_capabilities",
        "traditional_dic_inspect",
        "traditional_dic_validate",
        "traditional_dic_run",
        "traditional_dic_status",
        "traditional_dic_summarize",
    ]


def test_capabilities_are_structured_and_match_canonical_contract() -> None:
    from traditional_dic.capabilities import capability_contract

    result = _client_call("traditional_dic_capabilities")
    assert result.is_error is False
    payload = result.structured_content
    assert payload["server"]["name"] == "Traditional-DIC"
    assert payload["capability_contract"] == capability_contract()
    assert payload["capability_contract"]["stereo_3d"]["solver"] == "subset"
    assert payload["capability_contract"]["multiview_3d"]["solver"] == "subset"


def test_inspect_and_validate_are_read_only() -> None:
    inspected = _client_call(
        "traditional_dic_inspect",
        {"workflow": "subset_2d", "case": "case/mono_DIC/ring"},
    ).structured_content
    assert inspected["workflow_kind"] == "subset_2d"
    assert inspected["case"]["case_root"] == "case/mono_DIC/ring"

    validated = _client_call(
        "traditional_dic_validate",
        {"workflow": "stereo_3d", "case": "case/stereo_DIC/plate_center_load"},
    ).structured_content
    assert validated["valid"] is True


@pytest.mark.parametrize(
    "workflow, override",
    [
        ("stereo_3d", {"solver": "mesh"}),
        ("multiview_3d", {"solver": "mesh"}),
        ("multiview_3d", {"solver": "both"}),
    ],
)
def test_three_dimensional_mesh_requests_are_domain_validation_failures(workflow: str, override: dict[str, str]) -> None:
    result = _client_call(
        "traditional_dic_validate",
        {
            "workflow": workflow,
            "case": "case/stereo_DIC/plate_center_load"
            if workflow == "stereo_3d"
            else "case/multi_DIC/CylinderDIC",
            "overrides": override,
        },
    )
    assert result.is_error is False
    payload = result.structured_content
    assert payload["valid"] is False
    assert any(error["code"] == "UNSUPPORTED_SOLVER_FOR_WORKFLOW" for error in payload["errors"])


def test_status_and_summarize_use_existing_f4_contract(tmp_path: Path) -> None:
    from traditional_dic.run_contract import RunWorkspace, finalize_run_contract
    from traditional_dic.workflows.common import WorkflowRunResult

    workspace = RunWorkspace.create(tmp_path, "subset_2d", run_id="mcp-fixture")
    artifact = tmp_path / "displacements.csv"
    artifact.write_text("x,y,u,v\n0,0,0,0\n", encoding="utf-8")
    finalize_run_contract(workspace, WorkflowRunResult("subset_2d", tmp_path, {"displacements": str(artifact)}))

    status = _client_call("traditional_dic_status", {"workspace": str(tmp_path)}).structured_content
    summary = _client_call("traditional_dic_summarize", {"workspace": str(tmp_path)}).structured_content
    assert status["run_id"] == "mcp-fixture"
    assert status["execution_status"] == "SUCCESS"
    assert summary["workflow_kind"] == "subset_2d"
    assert summary["primary_result_type"]


def test_run_requires_absolute_external_output_root() -> None:
    from traditional_dic.mcp_server import traditional_dic_run

    async def invoke():
        return await traditional_dic_run(
            "subset_2d", "case/mono_DIC/ring", "relative-output"
        )

    with pytest.raises(ValueError, match="OUTPUT_ROOT_MUST_BE_ABSOLUTE"):
        asyncio.run(invoke())


def test_run_returns_structured_validation_failure_without_solver() -> None:
    result = _client_call(
        "traditional_dic_run",
        {
            "workflow": "stereo_3d",
            "case": "case/stereo_DIC/plate_center_load",
            "output_root": "/tmp/traditional-dic-m1-invalid",
            "overrides": {"solver": "mesh"},
        },
    )
    assert result.is_error is False
    payload = result.structured_content
    assert payload["execution_status"] == "VALIDATION_FAILED"
    assert any(error["code"] == "UNSUPPORTED_SOLVER_FOR_WORKFLOW" for error in payload["errors"])


def test_base_package_import_does_not_eagerly_require_mcp() -> None:
    code = """
import importlib.abc, sys
class BlockMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp' or fullname.startswith('mcp.'):
            raise ModuleNotFoundError('blocked optional MCP dependency')
        return None
sys.meta_path.insert(0, BlockMCP())
import traditional_dic
from traditional_dic.cli import main
assert main(['capabilities', '--format', 'json']) == 0
assert 'traditional_dic.mcp_server' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "python")},
    )
    assert "traditional_dic.mcp_server" not in result.stdout


def test_mcp_module_has_no_direct_print_calls() -> None:
    source = Path(__file__).resolve().parents[2] / "python" / "traditional_dic" / "mcp_server.py"
    assert "print(" not in source.read_text(encoding="utf-8")
