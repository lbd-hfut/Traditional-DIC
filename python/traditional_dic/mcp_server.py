"""Official MCP v2 adapter for the normalized Traditional-DIC contracts.

The six tools in this module are a transport layer only.  Case/configuration
resolution, workflow execution, and run-contract interpretation remain owned
by F1--F4 Python modules.  The optional ``mcp`` dependency is imported here,
not from :mod:`traditional_dic`, so the base API and CLI remain usable without
MCP installed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover - exercised in an env without optional extra
    raise ImportError(
        "Traditional-DIC MCP support requires the optional 'mcp' dependency; "
        "install the project with the 'mcp' extra."
    ) from exc

from .capabilities import WORKFLOW_KINDS, capability_contract
from ._runtime import default_paths_config, runtime_root
from .case import inspect_case, resolve_case
from .config_resolver import inspect_config, resolve_config
from .run_contract import load_run, summarize_run
from .workflows import run_mesh_2d, run_multiview_3d, run_stereo_3d, run_subset_2d


LOGGER = logging.getLogger("traditional_dic.mcp")
REPOSITORY_ROOT = runtime_root()
WorkflowKind = Literal["subset_2d", "mesh_2d", "stereo_3d", "multiview_3d"]


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPOSITORY_ROOT / path)


def _package_version() -> str:
    try:
        return importlib.metadata.version("traditional-dic")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _workflow_kind(workflow: str) -> str:
    if workflow not in WORKFLOW_KINDS:
        raise ValueError(
            f"INVALID_WORKFLOW: {workflow!r}; choose one of {', '.join(WORKFLOW_KINDS)}"
        )
    return workflow


def _case_kwargs(case: str, paths_config: str | None) -> dict[str, Any]:
    case_value = Path(case).expanduser()
    if not case_value.is_absolute():
        candidate = (REPOSITORY_ROOT / case_value).resolve()
        if candidate.exists():
            case_value = candidate
    kwargs: dict[str, Any] = {
        "paths_config": _repository_path(paths_config) if paths_config else default_paths_config(),
        "repository_root": REPOSITORY_ROOT,
    }
    if case_value.exists():
        kwargs["case_root"] = case_value
    else:
        kwargs["case_key"] = case
    return kwargs


def _config_kwargs(
    workflow: str,
    config: str | None,
    calibration_config: str | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "repository_root": REPOSITORY_ROOT,
        "overrides": dict(overrides or {}),
    }
    if config:
        kwargs["config_path"] = _repository_path(config)
    if workflow == "stereo_3d" and calibration_config:
        kwargs["calibration_config_path"] = _repository_path(calibration_config)
    return kwargs


def _resolved_inputs(
    workflow: str,
    case: str,
    *,
    paths_config: str | None = None,
    config: str | None = None,
    calibration_config: str | None = None,
    overrides: Mapping[str, Any] | None = None,
):
    workflow = _workflow_kind(workflow)
    resolved_case = resolve_case(workflow, **_case_kwargs(case, paths_config))
    resolved_config = resolve_config(
        workflow,
        **_config_kwargs(workflow, config, calibration_config, overrides),
    )
    return workflow, resolved_case, resolved_config


def _validation_payload(
    workflow: str,
    case: str,
    *,
    paths_config: str | None = None,
    config: str | None = None,
    calibration_config: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workflow = _workflow_kind(workflow)
    case_result = inspect_case(workflow, **_case_kwargs(case, paths_config))
    config_result = inspect_config(
        workflow,
        **_config_kwargs(workflow, config, calibration_config, overrides),
    )
    errors = [issue.to_dict() for issue in case_result.errors]
    errors.extend(config_result.get("errors", []))
    warnings = [issue.to_dict() for issue in case_result.warnings]
    warnings.extend(config_result.get("warnings", []))
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "valid": bool(case_result.valid and config_result.get("valid", False)),
        "workflow_kind": workflow,
        "errors": errors,
        "warnings": warnings,
        "capability_contract": capability_contract(),
    }
    if case_result.resolved is not None:
        payload["case"] = case_result.resolved.to_dict(repository_root=REPOSITORY_ROOT)
    if config_result.get("config") is not None:
        payload["config"] = config_result["config"]
    return payload


def _run_payload(result: Any) -> dict[str, Any]:
    contract = load_run(result.output_root)
    status = contract["status"]
    workspace = Path(result.output_root).resolve()
    return {
        "schema_version": "1.0",
        "run_id": result.run_id,
        "workflow_kind": result.workflow_kind,
        "execution_status": status.get("execution_status"),
        "quality_status": status.get("quality_status", "UNKNOWN"),
        "workspace": str(workspace),
        "manifest": str(Path(result.manifest_path).resolve()),
        "status": str(Path(result.status_path).resolve()),
        "metrics": str(Path(result.metrics_path).resolve()),
        "result": str(Path(result.result_path).resolve()),
        "warnings": status.get("warnings", []),
    }


def _validate_output_root(output_root: str, case_root: Path) -> Path:
    path = Path(output_root).expanduser()
    if not path.is_absolute():
        raise ValueError("OUTPUT_ROOT_MUST_BE_ABSOLUTE: MCP run requires an absolute output_root")
    resolved = path.resolve()
    try:
        resolved.relative_to(case_root.resolve())
    except ValueError:
        return resolved
    raise ValueError("OUTPUT_ROOT_MUST_BE_EXTERNAL: MCP output_root must not be inside the source case")


def _invoke_sync(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous facade while keeping stdio MCP framing clean."""
    with contextlib.redirect_stdout(sys.stderr):
        return function(*args, **kwargs)


async def _invoke(function: Any, *args: Any, **kwargs: Any) -> Any:
    # The workflow facade is intentionally synchronous.  MCP run is therefore
    # a synchronous tool call as documented; keeping execution in this task
    # also avoids unmanaged worker threads during stdio shutdown.
    return _invoke_sync(function, *args, **kwargs)


SERVER_INSTRUCTIONS = (
    "Traditional-DIC supports standalone 2D Subset-DIC and Mesh-DIC. "
    "Stereo 3D-DIC and Multiview 3D-DIC are Subset-only. "
    "Use capabilities, inspect, and validate before expensive runs; use status "
    "and summarize for existing workspaces. Point validity is scientific quality, "
    "not by itself execution failure."
)


mcp = MCPServer(
    "Traditional-DIC",
    version=_package_version(),
    instructions=SERVER_INSTRUCTIONS,
    log_level="WARNING",
)


@mcp.tool(name="traditional_dic_capabilities", structured_output=True)
async def traditional_dic_capabilities() -> dict[str, Any]:
    """Return the supported workflows and fixed solver capability contract."""
    return {
        "schema_version": "1.0",
        "server": {"name": "Traditional-DIC", "version": _package_version()},
        "package": {"name": "traditional-dic", "version": _package_version()},
        "supported_workflows": list(WORKFLOW_KINDS),
        "capability_contract": capability_contract(),
    }


@mcp.tool(name="traditional_dic_inspect", structured_output=True)
async def traditional_dic_inspect(
    workflow: WorkflowKind,
    case: str,
    paths_config: str | None = None,
    config: str | None = None,
    calibration_config: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve case/config inputs read-only; never runs a solver or writes outputs."""
    workflow, resolved_case, resolved_config = await _invoke(
        _resolved_inputs,
        workflow,
        case,
        paths_config=paths_config,
        config=config,
        calibration_config=calibration_config,
        overrides=overrides,
    )
    return {
        "schema_version": "1.0",
        "workflow_kind": workflow,
        "case": resolved_case.to_dict(repository_root=REPOSITORY_ROOT),
        "config": resolved_config.to_dict(repository_root=REPOSITORY_ROOT),
        "warnings": [warning.to_dict() for warning in resolved_config.warnings],
        "capability_contract": capability_contract(),
    }


@mcp.tool(name="traditional_dic_validate", structured_output=True)
async def traditional_dic_validate(
    workflow: WorkflowKind,
    case: str,
    paths_config: str | None = None,
    config: str | None = None,
    calibration_config: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate normalized case/config inputs without scientific execution."""
    return await _invoke(
        _validation_payload,
        workflow,
        case,
        paths_config=paths_config,
        config=config,
        calibration_config=calibration_config,
        overrides=overrides,
    )


@mcp.tool(name="traditional_dic_run", structured_output=True)
async def traditional_dic_run(
    workflow: WorkflowKind,
    case: str,
    output_root: str,
    paths_config: str | None = None,
    config: str | None = None,
    calibration_config: str | None = None,
    overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
    element_types: list[str] | None = None,
    dense_samples_per_axis: int = 25,
    skip_calibration: bool = False,
    compute_fields: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute one validated workflow in an explicit external output workspace."""
    validation = await traditional_dic_validate(
        workflow,
        case,
        paths_config=paths_config,
        config=config,
        calibration_config=calibration_config,
        overrides=overrides,
    )
    if not validation["valid"]:
        return {
            **validation,
            "execution_status": "VALIDATION_FAILED",
            "quality_status": "NOT_EVALUATED",
        }
    workflow, resolved_case, resolved_config = await _invoke(
        _resolved_inputs,
        workflow,
        case,
        paths_config=paths_config,
        config=config,
        calibration_config=calibration_config,
        overrides=overrides,
    )
    destination = _validate_output_root(output_root, resolved_case.case_root)
    runner = {
        "subset_2d": run_subset_2d,
        "mesh_2d": run_mesh_2d,
        "stereo_3d": run_stereo_3d,
        "multiview_3d": run_multiview_3d,
    }[workflow]
    kwargs: dict[str, Any] = {
        "repository_root": REPOSITORY_ROOT,
        "output_root": destination,
        "run_id": run_id,
    }
    if workflow == "mesh_2d":
        kwargs["element_types"] = element_types or ["T3", "Q4", "Q8"]
        kwargs["dense_samples_per_axis"] = dense_samples_per_axis
    elif workflow == "stereo_3d":
        kwargs["calibrate"] = not skip_calibration if skip_calibration else None
        kwargs["compute_fields"] = True if compute_fields else None
    elif workflow == "multiview_3d":
        kwargs["resume"] = resume
    result = await _invoke(runner, resolved_case, resolved_config, **kwargs)
    return _run_payload(result)


@mcp.tool(name="traditional_dic_status", structured_output=True)
async def traditional_dic_status(workspace: str) -> dict[str, Any]:
    """Read F4 operational status from an existing run workspace without rerunning."""
    contract = await _invoke(load_run, workspace)
    status = contract["status"]
    return {
        "schema_version": "1.0",
        "run_id": status["run_id"],
        "workflow_kind": status["workflow_kind"],
        "execution_status": status.get("execution_status"),
        "quality_status": status.get("quality_status", contract["metrics"].get("quality_status", "UNKNOWN")),
        "completed_stages": status.get("completed_stages", []),
        "failed_stage": status.get("failed_stage"),
        "warnings": status.get("warnings", []),
        "errors": status.get("errors", []),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_seconds": status.get("duration_seconds"),
    }


@mcp.tool(name="traditional_dic_summarize", structured_output=True)
async def traditional_dic_summarize(workspace: str) -> dict[str, Any]:
    """Summarize an existing F4 run using manifest, status, metrics, and result metadata."""
    return await _invoke(summarize_run, workspace)


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server; stdio is the safe default transport."""
    parser = argparse.ArgumentParser(prog="traditional-dic-mcp")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp", dest="streamable_http_path")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        if args.host == "0.0.0.0":
            LOGGER.warning("binding MCP HTTP on 0.0.0.0 is not a secure default")
        asyncio.run(
            mcp.run_streamable_http_async(
                host=args.host,
                port=args.port,
                streamable_http_path=args.streamable_http_path,
            )
        )
    return 0


__all__ = [
    "SERVER_INSTRUCTIONS",
    "mcp",
    "main",
    "traditional_dic_capabilities",
    "traditional_dic_inspect",
    "traditional_dic_validate",
    "traditional_dic_run",
    "traditional_dic_status",
    "traditional_dic_summarize",
]


if __name__ == "__main__":
    raise SystemExit(main())
