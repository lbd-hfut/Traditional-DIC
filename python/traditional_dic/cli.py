"""Stable command-line adapter for the normalized Traditional-DIC contracts.

The CLI deliberately contains no case discovery, configuration merging, or
scientific execution logic.  It translates command-line arguments into the
F1--F4 Python contracts and renders their results for humans or agents.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .case import (
    CaseResolutionError,
    inspect_case,
    resolve_case,
)
from ._runtime import default_paths_config, runtime_root
from .capabilities import (
    CAPABILITY_CONTRACT,
    WORKFLOW_DISPLAY,
    WORKFLOW_NAMES,
)
from .config_resolver import (
    ConfigResolutionError,
    inspect_config,
    resolve_config,
)
from .run_contract import (
    RunContractError,
    load_run,
    summarize_run,
)
from .workflows import (
    run_mesh_2d,
    run_multiview_3d,
    run_stereo_3d,
    run_subset_2d,
)


CLI_SCHEMA_VERSION = "1.0"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_VALIDATION = 3
EXIT_EXECUTION = 4
EXIT_CONTRACT = 5

WORKFLOW_DISPATCH = {
    "subset_2d": run_subset_2d,
    "mesh_2d": run_mesh_2d,
    "stereo_3d": run_stereo_3d,
    "multiview_3d": run_multiview_3d,
}

def _package_version() -> str:
    try:
        return importlib.metadata.version("traditional-dic")
    except importlib.metadata.PackageNotFoundError:
        # Source-tree invocations do not have distribution metadata.  The
        # project version remains authoritative in pyproject.toml.
        return "0.1.0"


def _repository_root() -> Path:
    return runtime_root()


def _internal_workflow(value: str) -> str:
    try:
        return WORKFLOW_NAMES[value]
    except KeyError as exc:
        raise ValueError(
            f"unsupported workflow {value!r}; choose one of {', '.join(WORKFLOW_NAMES)}"
        ) from exc


def _display_workflow(value: str) -> str:
    return WORKFLOW_DISPLAY.get(value, value)


def _scalar_value(raw: str) -> Any:
    """Parse one data-only override scalar without evaluating Python code."""
    try:
        import yaml

        value = yaml.safe_load(raw)
    except Exception:
        value = raw
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("--set values must be scalar YAML/JSON values")
    return value if value is not None or raw.strip().lower() in {"null", "none", "~"} else raw


def _parse_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--set override key cannot be empty")
        overrides[key] = _scalar_value(raw.strip())
    return overrides


def _case_kwargs(args: argparse.Namespace, workflow_kind: str, repo: Path) -> dict[str, Any]:
    case_value = Path(args.case).expanduser()
    if not case_value.is_absolute():
        repository_candidate = (repo / case_value).resolve()
        if repository_candidate.exists():
            case_value = repository_candidate
    kwargs: dict[str, Any] = {
        "paths_config": Path(args.paths_config).expanduser() if args.paths_config else default_paths_config(),
        "repository_root": repo,
    }
    # Existing directories are explicit case roots; otherwise preserve the
    # F1 case-key lookup contract.
    if case_value.exists():
        kwargs["case_root"] = case_value
    else:
        kwargs["case_key"] = args.case
    return kwargs


def _config_kwargs(args: argparse.Namespace, workflow_kind: str, repo: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "repository_root": repo,
        "overrides": _parse_overrides(getattr(args, "set_values", None)),
    }
    if getattr(args, "config", None):
        kwargs["config_path"] = Path(args.config).expanduser()
    if workflow_kind == "stereo_3d" and getattr(args, "calibration_config", None):
        kwargs["calibration_config_path"] = Path(args.calibration_config).expanduser()
    return kwargs


def _resolve_inputs(args: argparse.Namespace) -> tuple[str, ResolvedCase, ResolvedConfig]:
    repo = _repository_root()
    workflow_kind = _internal_workflow(args.workflow)
    case = resolve_case(workflow_kind, **_case_kwargs(args, workflow_kind, repo))
    config = resolve_config(workflow_kind, **_config_kwargs(args, workflow_kind, repo))
    return workflow_kind, case, config


def _validation_payload(args: argparse.Namespace) -> dict[str, Any]:
    repo = _repository_root()
    workflow_kind = _internal_workflow(args.workflow)
    case_kwargs = _case_kwargs(args, workflow_kind, repo)
    config_kwargs = _config_kwargs(args, workflow_kind, repo)
    case_result = inspect_case(workflow_kind, **case_kwargs)
    config_result = inspect_config(workflow_kind, **config_kwargs)
    errors = [issue.to_dict() for issue in case_result.errors]
    errors.extend(config_result.get("errors", []))
    warnings = [issue.to_dict() for issue in case_result.warnings]
    warnings.extend(config_result.get("warnings", []))
    payload: dict[str, Any] = {
        "schema_version": CLI_SCHEMA_VERSION,
        "valid": bool(case_result.valid and config_result.get("valid", False)),
        "workflow_kind": workflow_kind,
        "workflow": _display_workflow(workflow_kind),
        "errors": errors,
        "warnings": warnings,
        "capability_contract": CAPABILITY_CONTRACT,
    }
    if case_result.resolved is not None:
        payload["case"] = case_result.resolved.to_dict(repository_root=repo)
    if config_result.get("config") is not None:
        payload["config"] = config_result["config"]
    return payload


def _inspect_payload(args: argparse.Namespace) -> dict[str, Any]:
    workflow_kind, case, config = _resolve_inputs(args)
    repo = _repository_root()
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "workflow_kind": workflow_kind,
        "workflow": _display_workflow(workflow_kind),
        "case": case.to_dict(repository_root=repo),
        "config": config.to_dict(repository_root=repo),
        "warnings": [warning.to_dict() for warning in config.warnings],
        "capability_contract": CAPABILITY_CONTRACT,
    }


def _run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "repository_root": _repository_root(),
        "output_root": Path(args.output).expanduser() if args.output else None,
        "run_id": args.run_id,
    }
    if args.workflow == "mesh-2d":
        kwargs["element_types"] = ["T3", "Q4", "Q8"] if args.element == "all" else [args.element]
        kwargs["dense_samples_per_axis"] = args.dense_samples_per_axis
    elif args.workflow == "stereo-3d":
        kwargs["calibrate"] = not args.skip_calibration if args.skip_calibration else None
        kwargs["compute_fields"] = True if args.compute_fields else None
    elif args.workflow == "multiview-3d":
        kwargs["resume"] = args.resume
    return kwargs


def _run_payload(result: Any) -> dict[str, Any]:
    status: Mapping[str, Any] = {}
    if getattr(result, "contract", None):
        status = result.contract.get("status", {})
    if not status and getattr(result, "status_path", None):
        try:
            status = load_run(result.output_root)["status"]
        except RunContractError:
            status = {}
    workspace = Path(result.output_root).resolve()
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "run_id": result.run_id,
        "workflow_kind": result.workflow_kind,
        "execution_status": status.get("execution_status", "EXECUTION_FAILED"),
        "quality_status": status.get("quality_status", "UNKNOWN"),
        "workspace": str(workspace),
        "manifest": str(Path(result.manifest_path).resolve()) if result.manifest_path else str(workspace / "manifest.json"),
        "status": str(Path(result.status_path).resolve()) if result.status_path else str(workspace / "status.json"),
        "metrics": str(Path(result.metrics_path).resolve()) if result.metrics_path else str(workspace / "metrics.json"),
        "result": str(Path(result.result_path).resolve()) if result.result_path else str(workspace / "result.json"),
    }


def _summary_payload(workspace: str | Path) -> dict[str, Any]:
    payload = summarize_run(workspace)
    payload["workflow"] = _display_workflow(str(payload["workflow_kind"]))
    return payload


def _capabilities_payload() -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "package": {"name": "traditional-dic", "version": _package_version()},
        "supported_workflows": list(WORKFLOW_NAMES),
        "capability_contract": CAPABILITY_CONTRACT,
    }


def _print_json(payload: Any, stream: Any = None) -> None:
    (stream or sys.stdout).write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _print_text(payload: Mapping[str, Any], *, title: str | None = None) -> None:
    if title:
        print(title)
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        print(f"{key}: {value}")


def _emit_error(code: str, message: str, *, fmt: str, details: Mapping[str, Any] | None = None) -> None:
    payload = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = dict(details)
    if fmt == "json":
        _print_json(payload, stream=sys.stderr)
    else:
        print(f"{code}: {message}", file=sys.stderr)


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("text", "json"), default="text", help="output format (default: text)")
    parser.add_argument("--debug", action="store_true", help="include a traceback for unexpected failures")


def _add_case_config(parser: argparse.ArgumentParser, *, require_case: bool = True) -> None:
    parser.add_argument(
        "--workflow",
        required=True,
        metavar="WORKFLOW",
        help="workflow: subset-2d, mesh-2d, stereo-3d, or multiview-3d",
    )
    parser.add_argument("--case", required=require_case, help="case directory or case key from case_paths.yaml")
    parser.add_argument("--paths-config", type=Path, help="case paths YAML (default: config/case_paths.yaml)")
    parser.add_argument("--config", type=Path, help="workflow YAML (default: workflow-specific config/*.yaml)")
    parser.add_argument("--calibration-config", type=Path, help="stereo calibration YAML override")
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE", help="repeatable F2 dotted configuration override")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traditional-dic",
        description="Stable command-line interface for Traditional-DIC workflows.",
    )
    parser.add_argument("--version", action="version", version=f"traditional-dic {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser("capabilities", help="show supported workflows and solver capabilities")
    _add_format(capabilities)

    for command, help_text in (("inspect", "resolve a case and config without running a solver"), ("validate", "validate a case and config without running a solver")):
        sub = subparsers.add_parser(command, help=help_text)
        _add_case_config(sub)
        _add_format(sub)

    run = subparsers.add_parser("run", help="execute one normalized workflow")
    _add_case_config(run)
    _add_format(run)
    run.add_argument("--output", "--output-root", dest="output", type=Path, help="explicit isolated output root")
    run.add_argument("--run-id", help="caller-provided F4 run identifier")
    run.add_argument("--element", choices=("T3", "Q4", "Q8", "all"), default="all", help="Mesh-DIC element type")
    run.add_argument("--dense-samples-per-axis", type=int, default=25, help="Mesh-DIC dense-field sampling size")
    run.add_argument("--skip-calibration", action="store_true", help="Stereo: reuse existing calibration")
    run.add_argument("--compute-fields", action="store_true", help="Stereo: compute the three Subset fields")
    run.add_argument("--resume", action="store_true", help="Multiview: reuse completed pairwise fields")

    for command, help_text in (("status", "read operational status from an F4 run workspace"), ("summarize", "summarize an F4 run workspace")):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("workspace", nargs="?", help="run workspace containing manifest.json/status.json/etc.")
        sub.add_argument("--run", dest="run_workspace", help="alias for the run workspace path")
        _add_format(sub)
    return parser


def _status_exit(status: str) -> int:
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        return EXIT_OK
    if status == "VALIDATION_FAILED":
        return EXIT_VALIDATION
    if status in {"PARTIAL_SUCCESS", "EXECUTION_FAILED"}:
        return EXIT_EXECUTION
    return EXIT_CONTRACT


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    fmt = getattr(args, "format", "text")
    try:
        if args.command == "capabilities":
            payload = _capabilities_payload()
            if fmt == "json":
                _print_json(payload)
            else:
                print("2D DIC: Subset, Mesh")
                print("Stereo 3D: Subset correspondence only")
                print("Multiview 3D: Subset correspondence only")
            return EXIT_OK

        if args.command == "inspect":
            payload = _inspect_payload(args)
            if fmt == "json":
                _print_json(payload)
            else:
                _print_text(payload, title=f"Resolved {_display_workflow(payload['workflow_kind'])}")
            return EXIT_OK

        if args.command == "validate":
            payload = _validation_payload(args)
            if fmt == "json":
                _print_json(payload)
            else:
                print("valid: " + str(payload["valid"]))
                for issue in payload["errors"]:
                    print(f"error: {issue.get('code')}: {issue.get('message')}")
                for warning in payload["warnings"]:
                    print(f"warning: {warning.get('code')}: {warning.get('message')}")
            if not payload["valid"] and fmt == "json":
                _emit_error(
                    "VALIDATION_FAILED",
                    "case/config validation failed",
                    fmt=fmt,
                    details={"errors": payload["errors"]},
                )
            return EXIT_OK if payload["valid"] else EXIT_VALIDATION

        if args.command == "run":
            workflow_kind, case, config = _resolve_inputs(args)
            runner = WORKFLOW_DISPATCH[workflow_kind]
            call_kwargs = _run_kwargs(args)
            # Existing Subset/Mesh implementations print progress.  Redirect
            # it to stderr so JSON stdout remains one parseable document.
            output_stream = sys.stderr if fmt == "json" else sys.stdout
            with contextlib.redirect_stdout(output_stream):
                result = runner(case, config, **call_kwargs)
            payload = _run_payload(result)
            if fmt == "json":
                _print_json(payload)
            else:
                print(f"workflow: {_display_workflow(payload['workflow_kind'])}")
                print(f"execution_status: {payload['execution_status']}")
                print(f"workspace: {payload['workspace']}")
                print(f"manifest: {payload['manifest']}")
            return _status_exit(str(payload["execution_status"]))

        workspace = args.workspace or args.run_workspace
        if not workspace:
            raise ValueError(f"{args.command} requires a workspace path")
        if args.command == "status":
            status = load_run(workspace)["status"]
            if fmt == "json":
                _print_json(status)
            else:
                _print_text(status, title="Run status")
            return _status_exit(str(status.get("execution_status")))
        if args.command == "summarize":
            payload = _summary_payload(workspace)
            if fmt == "json":
                _print_json(payload)
            else:
                _print_text(payload, title="Run summary")
            return _status_exit(str(payload["execution_status"]))
        raise ValueError(f"unknown command: {args.command}")
    except (CaseResolutionError, ConfigResolutionError) as exc:
        _emit_error(exc.code, str(exc), fmt=fmt, details=getattr(exc, "details", None))
        return EXIT_VALIDATION
    except RunContractError as exc:
        _emit_error("RUN_CONTRACT_ERROR", str(exc), fmt=fmt)
        return EXIT_CONTRACT
    except (ValueError, TypeError) as exc:
        _emit_error("INVALID_ARGUMENT", str(exc), fmt=fmt)
        return EXIT_VALIDATION
    except Exception as exc:  # scientific exceptions retain a stable CLI envelope
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        _emit_error("EXECUTION_FAILED", str(exc) or exc.__class__.__name__, fmt=fmt)
        return EXIT_EXECUTION


__all__ = [
    "CAPABILITY_CONTRACT",
    "EXIT_CONTRACT",
    "EXIT_EXECUTION",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXIT_VALIDATION",
    "WORKFLOW_NAMES",
    "main",
]
