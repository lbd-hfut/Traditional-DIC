"""Normalized run metadata for Traditional-DIC workflow executions.

This module is deliberately limited to execution metadata.  It never parses
or rewrites scientific arrays; legacy workflow artifacts remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .case import ResolvedCase
from .capabilities import f4_capability_contract
from .config_resolver import ResolvedConfig


SCHEMA_VERSION = "1.0"
FINAL_STATUSES = frozenset(
    {
        "SUCCESS",
        "SUCCESS_WITH_WARNINGS",
        "PARTIAL_SUCCESS",
        "VALIDATION_FAILED",
        "EXECUTION_FAILED",
    }
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTRACT_FILES = ("manifest.json", "status.json", "metrics.json", "result.json")
_METADATA_NAMES = set(_CONTRACT_FILES)


class RunContractError(ValueError):
    """Raised when normalized run metadata is malformed or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunContractError(f"missing run contract file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunContractError(f"malformed run contract file: {path}") from exc
    if not isinstance(payload, dict):
        raise RunContractError(f"run contract file must contain an object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RunContractError(
            f"unsupported run contract schema in {path}: {payload.get('schema_version')!r}"
        )
    return payload


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise RunContractError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return run_id


def _portable_path(path: Path, workspace_root: Path, repository_root: Path) -> tuple[str, str]:
    path = path.resolve()
    workspace_root = workspace_root.resolve()
    repository_root = repository_root.resolve()
    try:
        return path.relative_to(workspace_root).as_posix(), "workspace"
    except ValueError:
        pass
    try:
        return path.relative_to(repository_root).as_posix(), "repository"
    except ValueError:
        return str(path), "absolute"


def _resolve_artifact_path(descriptor: Mapping[str, Any], workspace_root: Path, repository_root: Path) -> Path:
    raw = Path(str(descriptor.get("path", "")))
    scope = str(descriptor.get("path_scope", "workspace"))
    if scope == "workspace":
        return (workspace_root / raw).resolve()
    if scope == "repository":
        return (repository_root / raw).resolve()
    if scope == "absolute":
        return raw.resolve()
    raise RunContractError(f"unsupported artifact path_scope: {scope!r}")


def _sha256(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "artifact"


def _artifact_kind(key: str, path: Path) -> str:
    key_l = key.lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    if key_l == "figures" or suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "figure"
    if "mask" in key_l or "mask" in name:
        return "pair_mask"
    if key_l in {"fields", "pairwise_2d"}:
        return "displacement_field"
    if "displacement" in key_l or "displacements" in name:
        return "displacement_field"
    if "strain" in key_l or "strain" in name:
        return "strain_field"
    if "node" in name:
        return "mesh_nodes"
    if "element" in name:
        return "mesh_elements"
    if "stereo_3d_points" in name:
        return "stereo_points"
    if "deformation_3d" in name:
        return "deformation_3d"
    if "stitched_points" in name:
        return "stitched_points"
    if "stitched_faces" in name:
        return "stitched_faces"
    if "calibration" in key_l or "calibration" in name:
        return "calibration"
    if "summary" in key_l or "summary" in name or suffix == ".json":
        return "summary"
    return "workflow_artifact"


def _stage_for_key(key: str, path: Path) -> str:
    key_l = key.lower()
    name = path.name.lower()
    if key_l in {"displacements", "strains", "stats"}:
        return "subset_solve" if key_l != "strains" else "strain"
    if key_l in {"results", "figures"}:
        if key_l == "figures":
            return "visualization"
        return "mesh_solve" if path.suffix.lower() in {".csv", ".txt"} else "write_outputs"
    if "calibration" in key_l or "calibration" in name:
        return "calibration"
    if "mask" in key_l or "mask" in name:
        return "masks"
    if "stitched" in name or "stitch" in key_l:
        return "stitching"
    if "deformation" in name or "reconstruct" in key_l or "stereo_3d" in name:
        return "reconstruction"
    if key_l in {"fields", "pairwise_2d"} or "field" in key_l or "disp" in key_l:
        return "pairwise_fields"
    if key_l in {"pairwise_3d", "reconstruction"}:
        return "reconstruction"
    if key_l in {"stitching", "surface_stitch"}:
        return "stitching"
    if "mesh" in key_l or "node" in name or "element" in name:
        return "mesh_solve"
    return "write_outputs"


def _iter_values(value: Any) -> Iterable[Path]:
    if isinstance(value, (str, Path)):
        yield Path(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_values(child)


@dataclass(frozen=True)
class ArtifactDescriptor:
    id: str
    kind: str
    path: str
    path_scope: str
    format: str
    stage: str
    required: bool
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "path_scope": self.path_scope,
            "format": self.format,
            "stage": self.stage,
            "required": self.required,
            "exists": self.exists,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class RunWorkspace:
    root: Path
    run_id: str
    workflow_kind: str
    repository_root: Path
    started_at: str

    @classmethod
    def create(
        cls,
        root: str | Path,
        workflow_kind: str,
        *,
        repository_root: str | Path | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
    ) -> "RunWorkspace":
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        resolved_id = _validate_run_id(run_id or f"run-{uuid.uuid4().hex}")
        workspace = cls(
            root=root_path,
            run_id=resolved_id,
            workflow_kind=str(workflow_kind),
            repository_root=Path(repository_root or root_path).resolve(),
            started_at=started_at or _utc_now(),
        )
        workspace.write_status(
            _status_payload(
                workspace,
                "RUNNING",
                finished_at=None,
                duration_seconds=None,
                completed_stages=[],
                failed_stage=None,
                errors=[],
                warnings=[],
            )
        )
        return workspace

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    def write_status(self, payload: Mapping[str, Any]) -> None:
        _atomic_write_json(self.status_path, payload)


def _status_payload(
    workspace: RunWorkspace,
    execution_status: str,
    *,
    finished_at: str | None,
    duration_seconds: float | None,
    completed_stages: Sequence[str],
    failed_stage: str | None,
    errors: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
    quality_status: str = "UNKNOWN",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": workspace.run_id,
        "workflow_kind": workspace.workflow_kind,
        "execution_status": execution_status,
        "quality_status": quality_status,
        "started_at": workspace.started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "completed_stages": list(dict.fromkeys(str(stage) for stage in completed_stages)),
        "failed_stage": failed_stage,
        "errors": [dict(error) for error in errors],
        "warnings": [dict(warning) for warning in warnings],
        "updated_at": finished_at or _utc_now(),
    }
    return payload


def _artifact_descriptors(workspace: RunWorkspace, artifacts: Mapping[str, Any]) -> list[ArtifactDescriptor]:
    descriptors: list[ArtifactDescriptor] = []
    seen: set[Path] = set()
    for key in sorted(artifacts):
        values = list(_iter_values(artifacts[key]))
        for candidate in values:
            path = candidate if candidate.is_absolute() else (workspace.root / candidate)
            path = path.resolve()
            if path in seen or path.name in _METADATA_NAMES:
                continue
            if path.is_dir():
                children = sorted(child for child in path.rglob("*") if child.is_file())
            else:
                children = [path]
            for child in children:
                child = child.resolve()
                if child in seen or child.name in _METADATA_NAMES:
                    continue
                seen.add(child)
                relative, scope = _portable_path(child, workspace.root, workspace.repository_root)
                kind = _artifact_kind(key, child)
                artifact_id = f"{_slug(key)}:{_slug(relative.replace('/', '__'))}"
                exists = child.is_file()
                descriptors.append(
                    ArtifactDescriptor(
                        id=artifact_id,
                        kind=kind,
                        path=relative,
                        path_scope=scope,
                        format=child.suffix.lower().lstrip(".") or "file",
                        stage=_stage_for_key(key, child),
                        required=key.lower() != "figures" and kind != "figure",
                        exists=exists,
                        size_bytes=child.stat().st_size if exists else None,
                        sha256=_sha256(child) if exists else None,
                        description=f"{key} artifact",
                    )
                )
    return sorted(descriptors, key=lambda item: item.id)


def _read_json_if_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.name in _METADATA_NAMES:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _collect_json_files(workspace: RunWorkspace) -> list[tuple[Path, dict[str, Any]]]:
    files: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(workspace.root.rglob("*.json")):
        payload = _read_json_if_file(path)
        if payload is not None:
            files.append((path, payload))
    return files


def derive_metrics(workspace: RunWorkspace, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact metrics from existing workflow summaries."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": workspace.run_id,
        "workflow_kind": workspace.workflow_kind,
        "metric_provenance": [],
    }
    json_files = _collect_json_files(workspace)
    summaries = [(path, value) for path, value in json_files if "summary" in path.name.lower() or path.name == "stats.json"]
    if workspace.workflow_kind == "subset_2d":
        stats = [(path, value) for path, value in summaries if path.name == "stats.json"]
        if stats:
            payload["frames"] = {path.parent.name: value for path, value in stats}
            first = stats[0][1]
            payload.update(first)
            total = sum(int(value.get("total_points", 0)) for _, value in stats)
            valid = sum(int(value.get("valid_points", 0)) for _, value in stats)
            payload["total_points"] = total
            payload["valid_points"] = valid
            payload["valid_fraction"] = float(valid / total) if total else 0.0
            payload["metric_provenance"].append("stats.json")
    elif workspace.workflow_kind == "mesh_2d":
        elements: dict[str, Any] = {}
        for path, value in summaries:
            if path.name != "summary.json" or "nodes" not in value or "elements" not in value:
                continue
            element = path.parent.name
            elements[element] = value
        if elements:
            payload["elements"] = elements
            payload["element_types"] = sorted(elements)
            payload["node_count"] = int(sum(int(value.get("nodes", 0)) for value in elements.values()))
            payload["element_count"] = int(sum(int(value.get("elements", 0)) for value in elements.values()))
            payload["dense_point_count"] = int(sum(int(value.get("dense_samples", 0)) for value in elements.values()))
            payload["metric_provenance"].append("summary.json")
    elif workspace.workflow_kind == "stereo_3d":
        stereo = [(path, value) for path, value in summaries if path.name == "stereo_3d_summary.json"]
        deformation = [(path, value) for path, value in summaries if path.name == "deformation_3d_summary.json"]
        if stereo:
            payload.update(stereo[0][1])
            payload["input_valid"] = int(stereo[0][1].get("valid_points", 0))
            payload["final_valid"] = int(stereo[0][1].get("valid_points", 0))
            total = int(stereo[0][1].get("total_points", 0))
            payload["valid_fraction"] = float(payload["final_valid"] / total) if total else 0.0
            payload["metric_provenance"].append(str(stereo[0][0].relative_to(workspace.root)))
        if deformation:
            payload["deformation"] = deformation[0][1]
            payload["metric_provenance"].append(str(deformation[0][0].relative_to(workspace.root)))
    elif workspace.workflow_kind == "multiview_3d":
        run_summaries = [(path, value) for path, value in json_files if path.name == "multiview_3d_run.json"]
        stitch_summaries = [(path, value) for path, value in json_files if path.name == "stitched_summary.json"]
        stitch_indexes = [(path, value) for path, value in json_files if path.name == "stitch_index.json"]
        if run_summaries:
            value = run_summaries[0][1]
            payload["camera_count"] = int(value.get("camera_count", 0))
            payload["pair_count"] = len(value.get("pairs", value.get("selected_pairs", [])) or [])
            payload["scale"] = value.get("sfm_to_world_scale")
            pairwise_2d = value.get("pairwise_2d_dic", {}) or {}
            pair_dirs = pairwise_2d.get("pair_dirs", []) or []
            if pairwise_2d.get("run_subset") and pair_dirs:
                payload["field_count"] = int(len(pair_dirs) * 3)
            pairwise_3d = value.get("pairwise_3d_dic", {}) or {}
            if "total_points" in pairwise_3d:
                payload["pairwise_point_count"] = int(pairwise_3d.get("total_points", 0))
            payload["metric_provenance"].append(str(run_summaries[0][0].relative_to(workspace.root)))
        if stitch_summaries:
            value = stitch_summaries[0][1]
            payload.update(
                {
                    "stitched_point_count": int(value.get("point_count", 0)),
                    "face_count": int(value.get("face_count", 0)),
                    "stitched_valid_count": int(value.get("point_count", 0)) - int(value.get("cleaned_removed_points", 0)),
                }
            )
            payload["metric_provenance"].append(str(stitch_summaries[0][0].relative_to(workspace.root)))
        elif stitch_indexes:
            value = stitch_indexes[0][1]
            payload["face_count"] = int(value.get("face_count", 0))
            payload["metric_provenance"].append(str(stitch_indexes[0][0].relative_to(workspace.root)))
    return payload


def _quality_warnings(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    valid_fraction = metrics.get("valid_fraction")
    if valid_fraction is not None and float(valid_fraction) == 0.0:
        warnings.append({"code": "ZERO_VALID_POINTS", "message": "no valid scientific points were reported", "stage": "quality"})

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            warnings.append({"code": "NONFINITE_METRIC", "message": f"non-finite metric at {path}", "stage": "quality"})
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(metrics)
    return warnings


def _capabilities(workflow_kind: str) -> dict[str, Any]:
    return f4_capability_contract()


def _primary_kind(kind: str) -> bool:
    return kind not in {"figure", "summary", "pair_mask"}


def finalize_run_contract(
    workspace: RunWorkspace,
    workflow_result: Any,
    *,
    resolved_case: ResolvedCase | None = None,
    resolved_config: ResolvedConfig | None = None,
    warnings: Sequence[Mapping[str, Any]] = (),
    execution_status: str = "SUCCESS",
    failed_stage: str | None = None,
    errors: Sequence[Mapping[str, Any]] = (),
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Build and atomically write the four normalized metadata files."""
    if execution_status not in FINAL_STATUSES:
        raise RunContractError(f"invalid final execution status: {execution_status!r}")
    descriptors = _artifact_descriptors(workspace, workflow_result.artifacts)
    missing_required = [item.id for item in descriptors if item.required and not item.exists]
    normalized_warnings = [dict(item) for item in warnings]
    if missing_required:
        execution_status = "EXECUTION_FAILED"
        errors = list(errors) + [
            {
                "code": "REQUIRED_ARTIFACT_MISSING",
                "message": "required workflow artifacts are missing",
                "stage": "contract_finalization",
                "details": {"artifact_ids": missing_required},
            }
        ]
        failed_stage = failed_stage or "contract_finalization"
    metrics = derive_metrics(workspace, workflow_result.artifacts)
    quality_warnings = _quality_warnings(metrics)
    normalized_warnings.extend(quality_warnings)
    quality_status = "QUALITY_WARNINGS" if normalized_warnings else "QUALITY_OK"
    if execution_status == "SUCCESS" and normalized_warnings:
        execution_status = "SUCCESS_WITH_WARNINGS"
    if execution_status == "SUCCESS" and workflow_result.warnings:
        normalized_warnings.extend(
            {"code": "WORKFLOW_WARNING", "message": str(message), "stage": "workflow"}
            for message in workflow_result.warnings
        )
        execution_status = "SUCCESS_WITH_WARNINGS"
        quality_status = "QUALITY_WARNINGS"
    finished = finished_at or _utc_now()
    try:
        start_dt = datetime.fromisoformat(workspace.started_at.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        duration = max(0.0, (end_dt - start_dt).total_seconds())
    except ValueError:
        duration = None
    stages = sorted({item.stage for item in descriptors if item.exists})
    status = _status_payload(
        workspace,
        execution_status,
        finished_at=finished,
        duration_seconds=duration,
        completed_stages=stages,
        failed_stage=failed_stage,
        errors=errors,
        warnings=normalized_warnings,
        quality_status=quality_status,
    )
    metrics["quality_status"] = quality_status
    metrics["quality_warnings"] = normalized_warnings
    metrics["status_file"] = "status.json"
    descriptors_dict = [item.to_dict() for item in descriptors]
    primary = [item.id for item in descriptors if item.required and _primary_kind(item.kind)]
    figures = [item.id for item in descriptors if item.kind == "figure"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": workspace.run_id,
        "workflow_kind": workspace.workflow_kind,
        "primary_result_type": {
            "subset_2d": "2d_displacement",
            "mesh_2d": "2d_mesh_displacement",
            "stereo_3d": "stereo_3d_displacement",
            "multiview_3d": "stitched_3d_surface",
        }.get(workspace.workflow_kind, "workflow_output"),
        "primary_artifacts": primary,
        "secondary_artifacts": [item.id for item in descriptors if item.id not in primary and item.id not in figures],
        "figures": figures,
        "metrics_file": "metrics.json",
        "status_file": "status.json",
        "manifest_file": "manifest.json",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": workspace.run_id,
        "workflow_kind": workspace.workflow_kind,
        "created_at": workspace.started_at,
        # Repository identity is intentionally absolute: it is provenance, not
        # an artifact path.  Artifact paths themselves remain portable below.
        "repository_root": str(workspace.repository_root),
        "capability_contract": _capabilities(workspace.workflow_kind),
        "workspace": {"root": ".", "artifacts": "artifacts", "figures": "figures", "logs": "logs"},
        "artifacts": descriptors_dict,
        "figures": figures,
        "logs": [],
    }
    if resolved_case is not None:
        manifest["case"] = resolved_case.to_dict(repository_root=workspace.repository_root)
    if resolved_config is not None:
        manifest["config"] = resolved_config.to_dict(repository_root=workspace.repository_root)
    if isinstance(getattr(workflow_result, "workflow_kind", None), str):
        manifest["workflow_result"] = {
            "workflow_kind": workflow_result.workflow_kind,
            "output_root": _portable_path(workspace.root, workspace.root, workspace.repository_root)[0],
        }
    _atomic_write_json(workspace.metrics_path, metrics)
    _atomic_write_json(workspace.result_path, result)
    _atomic_write_json(workspace.manifest_path, manifest)
    workspace.write_status(status)
    return {
        "manifest": manifest,
        "status": status,
        "metrics": metrics,
        "result": result,
    }


def write_execution_failure(
    workspace: RunWorkspace,
    exc: BaseException,
    *,
    stage: str = "workflow_execution",
    finished_at: str | None = None,
) -> None:
    finished = finished_at or _utc_now()
    error = {
        "code": "WORKFLOW_EXECUTION_FAILED",
        "message": str(exc) or exc.__class__.__name__,
        "stage": stage,
        "exception_type": exc.__class__.__name__,
    }
    try:
        workspace.write_status(
            _status_payload(
                workspace,
                "EXECUTION_FAILED",
                finished_at=finished,
                duration_seconds=None,
                completed_stages=[],
                failed_stage=stage,
                errors=[error],
                warnings=[],
            )
        )
    except Exception:
        # Preserve the original scientific exception if the control-plane
        # failure itself prevents status serialization.
        pass


def _same_identity(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    run_ids = {str(payload.get("run_id")) for payload in payloads.values()}
    workflows = {str(payload.get("workflow_kind")) for payload in payloads.values()}
    if len(run_ids) != 1 or len(workflows) != 1:
        raise RunContractError("run contract files disagree on run_id or workflow_kind")


def validate_run_contract(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    payloads = {name[:-5]: _load_json(root / name) for name in _CONTRACT_FILES}
    _same_identity(payloads)
    run_id = str(payloads["manifest"].get("run_id", ""))
    _validate_run_id(run_id)
    workflow_kind = str(payloads["manifest"].get("workflow_kind", ""))
    if workflow_kind not in {"subset_2d", "mesh_2d", "stereo_3d", "multiview_3d"}:
        raise RunContractError(f"unsupported workflow_kind in run contract: {workflow_kind!r}")
    required_manifest = {"schema_version", "run_id", "workflow_kind", "capability_contract", "artifacts"}
    required_status = {"schema_version", "run_id", "workflow_kind", "execution_status", "errors", "warnings"}
    required_metrics = {"schema_version", "run_id", "workflow_kind"}
    required_result = {"schema_version", "run_id", "workflow_kind", "primary_artifacts", "secondary_artifacts", "figures"}
    for name, required in (("manifest", required_manifest), ("status", required_status), ("metrics", required_metrics), ("result", required_result)):
        missing = sorted(required - set(payloads[name]))
        if missing:
            raise RunContractError(f"{name}.json is missing required fields: {missing}")
    status = payloads["status"]
    if status.get("execution_status") not in FINAL_STATUSES | {"RUNNING"}:
        raise RunContractError(f"invalid execution_status: {status.get('execution_status')!r}")
    manifest_artifacts = payloads["manifest"].get("artifacts")
    if not isinstance(manifest_artifacts, list):
        raise RunContractError("manifest.artifacts must be a list")
    by_id = {str(item.get("id")): item for item in manifest_artifacts if isinstance(item, Mapping)}
    if len(by_id) != len(manifest_artifacts):
        raise RunContractError("manifest artifact IDs must be unique")
    for artifact_id, descriptor in by_id.items():
        required_descriptor = {"id", "kind", "path", "path_scope", "format", "stage", "required", "exists"}
        missing = sorted(required_descriptor - set(descriptor))
        if missing:
            raise RunContractError(f"artifact {artifact_id!r} is missing fields: {missing}")
        if descriptor.get("path_scope") not in {"workspace", "repository", "absolute"}:
            raise RunContractError(f"artifact {artifact_id!r} has invalid path_scope")
    result = payloads["result"]
    if result.get("manifest_file") != "manifest.json" or result.get("status_file") != "status.json" or result.get("metrics_file") != "metrics.json":
        raise RunContractError("result metadata file references are inconsistent")
    result_ids = list(result.get("primary_artifacts", [])) + list(result.get("secondary_artifacts", [])) + list(result.get("figures", []))
    missing_ids = sorted(set(result_ids) - set(by_id))
    if missing_ids:
        raise RunContractError(f"result references unknown artifacts: {missing_ids}")
    repository_root = Path(str(payloads["manifest"].get("repository_root", root))).resolve()
    if not repository_root.is_absolute():
        repository_root = (root / repository_root).resolve()
    if status.get("execution_status") != "RUNNING":
        missing_required = []
        for descriptor in manifest_artifacts:
            if descriptor.get("required") and descriptor.get("exists"):
                if not _resolve_artifact_path(descriptor, root, repository_root).is_file():
                    missing_required.append(descriptor.get("id"))
        if missing_required:
            raise RunContractError(f"required artifacts are not present: {missing_required}")
    return payloads


def load_manifest(workspace: str | Path) -> dict[str, Any]:
    return _load_json(Path(workspace).resolve() / "manifest.json")


def load_status(workspace: str | Path) -> dict[str, Any]:
    return _load_json(Path(workspace).resolve() / "status.json")


def load_metrics(workspace: str | Path) -> dict[str, Any]:
    return _load_json(Path(workspace).resolve() / "metrics.json")


def load_result(workspace: str | Path) -> dict[str, Any]:
    return _load_json(Path(workspace).resolve() / "result.json")


def load_run(workspace: str | Path) -> dict[str, dict[str, Any]]:
    payloads = validate_run_contract(workspace)
    return {key: payloads[key] for key in ("manifest", "status", "metrics", "result")}


def inspect_run(workspace: str | Path) -> dict[str, dict[str, Any]]:
    return load_run(workspace)


def summarize_run(workspace: str | Path) -> dict[str, Any]:
    """Return the compact, transport-neutral summary used by adapters."""
    payloads = load_run(workspace)
    manifest = payloads["manifest"]
    status = payloads["status"]
    metrics = payloads["metrics"]
    result = payloads["result"]
    descriptors = {
        str(item["id"]): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    artifact_ids = list(result.get("primary_artifacts", [])) + list(result.get("secondary_artifacts", []))
    artifacts = [
        {
            "id": artifact_id,
            "kind": descriptors[artifact_id].get("kind"),
            "path": descriptors[artifact_id].get("path"),
            "stage": descriptors[artifact_id].get("stage"),
        }
        for artifact_id in artifact_ids
        if artifact_id in descriptors
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "workflow_kind": manifest["workflow_kind"],
        "execution_status": status.get("execution_status"),
        "quality_status": status.get("quality_status", metrics.get("quality_status", "UNKNOWN")),
        "metrics": metrics,
        "primary_result_type": result.get("primary_result_type"),
        "artifacts": artifacts,
        "warnings": status.get("warnings", []),
        "errors": status.get("errors", []),
        "workspace": str(Path(workspace).resolve()),
    }


def execute_with_contract(
    workflow_kind: str,
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    solver_root: str,
    repository_root: str | Path | None,
    output_root: str | Path | None,
    visualization_root: str | Path | None,
    run_id: str | None,
    runner: Callable[[], Any],
    make_context_fn: Callable[..., Any],
    require_workflow_fn: Callable[..., Any],
    output_paths_fn: Callable[..., Any],
) -> Any:
    """Run an existing facade implementation and add the normalized contract."""
    context = make_context_fn(
        resolved_case,
        resolved_config,
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
    )
    require_workflow_fn(context, workflow_kind)
    result_root, _ = output_paths_fn(
        context,
        solver_root,
        default_result=f"result/{solver_root}",
        default_visualization=f"visualization/{solver_root}",
    )
    workspace = RunWorkspace.create(
        result_root,
        workflow_kind,
        repository_root=context.repository_root,
        run_id=run_id,
    )
    try:
        result = runner()
    except Exception as exc:
        write_execution_failure(workspace, exc)
        raise
    try:
        finalized = finalize_run_contract(
            workspace,
            result,
            resolved_case=resolved_case,
            resolved_config=resolved_config,
        )
        validate_run_contract(workspace.root)
    except Exception as exc:
        write_execution_failure(workspace, exc, stage="contract_finalization")
        raise
    return replace(
        result,
        run_id=workspace.run_id,
        manifest_path=workspace.manifest_path,
        status_path=workspace.status_path,
        metrics_path=workspace.metrics_path,
        result_path=workspace.result_path,
        contract=finalized,
    )


__all__ = [
    "SCHEMA_VERSION",
    "FINAL_STATUSES",
    "RunContractError",
    "ArtifactDescriptor",
    "RunWorkspace",
    "derive_metrics",
    "finalize_run_contract",
    "write_execution_failure",
    "validate_run_contract",
    "load_manifest",
    "load_status",
    "load_metrics",
    "load_result",
    "load_run",
    "inspect_run",
    "summarize_run",
    "execute_with_contract",
]
