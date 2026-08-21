"""Deterministic case and input resolution for Traditional-DIC workflows.

This module deliberately stops at input discovery.  It does not load image
pixels, generate masks, run calibration, or invoke a DIC solver.  The rules
mirror the existing example entry points while making frame roles, camera
ordering, ROI mode, and calibration ordering explicit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from ._runtime import runtime_root


WORKFLOW_KINDS = frozenset({"subset_2d", "mesh_2d", "stereo_3d", "multiview_3d"})
IMAGE_SUFFIXES = frozenset({".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
DEFAULT_REPOSITORY_ROOT = runtime_root()


class CaseResolutionError(ValueError):
    """Structured fail-closed input resolution error."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    resolved: "ResolvedCase | None" = None

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "resolved": self.resolved.to_dict(repository_root=repository_root) if self.resolved else None,
        }


@dataclass(frozen=True)
class CaseSpec:
    """Unresolved, serializable input specification accepted by a resolver."""

    workflow_kind: str
    case_root: Path | None = None
    paths_config: Path | None = None
    case_key: str | None = None
    case_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        return {
            "workflow_kind": self.workflow_kind,
            "case_root": _serialize_path(self.case_root, repository_root) if self.case_root else None,
            "paths_config": _serialize_path(self.paths_config, repository_root) if self.paths_config else None,
            "case_key": self.case_key,
            "case_config": _json_safe(dict(self.case_config), repository_root),
        }


@dataclass(frozen=True)
class FrameRef:
    role: str
    path: Path
    index: int | None = None

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "path": _serialize_path(self.path, repository_root)}
        if self.index is not None:
            out["index"] = self.index
        return out


@dataclass(frozen=True)
class ROIInput:
    mode: str
    path: Path | None = None
    frame_name: str | None = None

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        if self.path is not None:
            out["path"] = _serialize_path(self.path, repository_root)
        if self.frame_name is not None:
            out["frame_name"] = self.frame_name
        return out


@dataclass(frozen=True)
class CalibrationInput:
    index: int
    paths: tuple[Path, ...]
    label: str | None = None
    camera_id: str | None = None

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "paths": [_serialize_path(path, repository_root) for path in self.paths],
        }
        if self.label is not None:
            out["label"] = self.label
        if self.camera_id is not None:
            out["camera_id"] = self.camera_id
        return out


@dataclass(frozen=True)
class CameraRef:
    index: int
    camera_id: str
    name: str
    path: Path
    reference: FrameRef
    deformed: FrameRef
    roi: FrameRef | None = None

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "camera_id": self.camera_id,
            "name": self.name,
            "path": _serialize_path(self.path, repository_root),
            "reference": self.reference.to_dict(repository_root=repository_root),
            "deformed": self.deformed.to_dict(repository_root=repository_root),
        }
        if self.roi is not None:
            out["roi"] = self.roi.to_dict(repository_root=repository_root)
        return out


@dataclass(frozen=True)
class ResolvedCase:
    workflow_kind: str
    case_root: Path
    frames: tuple[FrameRef, ...] = ()
    roi: ROIInput = field(default_factory=lambda: ROIInput("none"))
    cameras: tuple[CameraRef, ...] = ()
    calibration_inputs: tuple[CalibrationInput, ...] = ()
    calibration_metadata: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Output roots are resolved from case_paths.yaml for workflow execution.
    # They are deliberately not part of the F1 input snapshot serialization.
    output_roots: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.workflow_kind not in WORKFLOW_KINDS:
            raise ValueError(f"unsupported workflow_kind: {self.workflow_kind!r}")

    def frame(self, role: str) -> FrameRef:
        for frame in self.frames:
            if frame.role == role:
                return frame
        raise KeyError(role)

    def to_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        return {
            "workflow_kind": self.workflow_kind,
            "case_root": _serialize_path(self.case_root, repository_root),
            "frames": [frame.to_dict(repository_root=repository_root) for frame in self.frames],
            "roi": self.roi.to_dict(repository_root=repository_root),
            "cameras": [camera.to_dict(repository_root=repository_root) for camera in self.cameras],
            "calibration_inputs": [item.to_dict(repository_root=repository_root) for item in self.calibration_inputs],
            "calibration_metadata": [_serialize_path(path, repository_root) for path in self.calibration_metadata],
            "metadata": _json_safe(dict(self.metadata), repository_root),
        }

    def to_json(self, *, repository_root: Path | None = None, indent: int = 2) -> str:
        return json.dumps(self.to_dict(repository_root=repository_root), indent=indent, sort_keys=True) + "\n"


def _json_safe(value: Any, repository_root: Path | None) -> Any:
    if isinstance(value, Path):
        return _serialize_path(value, repository_root)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, repository_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, repository_root) for item in value]
    return value


def _serialize_path(path: Path, repository_root: Path | None) -> str:
    raw = Path(path).expanduser()
    root = Path(repository_root).expanduser().resolve() if repository_root is not None else None
    absolute = (root / raw).resolve() if root is not None and not raw.is_absolute() else raw.resolve()
    if repository_root is not None:
        assert root is not None
        try:
            return absolute.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(absolute)


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _image_files(directory: Path, *, natural: bool = False) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise CaseResolutionError("MISSING_DIRECTORY", f"input directory does not exist: {directory}", details={"path": str(directory)})
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=_natural_key if natural else lambda path: path.name)


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return tuple(int(value) for value in image.size)
    except Exception as exc:  # pragma: no cover - exercised by malformed external cases
        raise CaseResolutionError("INVALID_IMAGE", f"cannot read image header for {path}: {exc}") from exc


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _case_config(
    workflow_kind: str,
    *,
    case_root: str | Path | None,
    paths_config: str | Path | None,
    case_key: str | None,
    repository_root: Path,
    case_config: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    if workflow_kind not in WORKFLOW_KINDS:
        raise CaseResolutionError("UNSUPPORTED_WORKFLOW", f"workflow kind must be one of {sorted(WORKFLOW_KINDS)}")
    cfg: dict[str, Any] = dict(case_config or {})
    config_base = repository_root
    if paths_config is not None:
        config_path = _resolve_path(paths_config, base=repository_root)
        loaded = load_config(config_path)
        key = case_key or {"subset_2d": "mono_2d", "mesh_2d": "mono_2d", "stereo_3d": "stereo_3d", "multiview_3d": "multiview_3d"}[workflow_kind]
        cfg = dict(loaded.get(key, {}) or {})
        if not cfg:
            raise CaseResolutionError("MISSING_CASE_CONFIG", f"case key {key!r} is not present in {config_path}")
        config_base = config_path.parent
    root_value = case_root if case_root is not None else cfg.get("case_root")
    if root_value is None:
        raise CaseResolutionError("MISSING_CASE_ROOT", "case_root is required for case resolution")
    # Existing examples interpret case_paths.yaml paths relative to the project
    # root, not the YAML file's directory. Preserve that convention.
    root_base = repository_root if paths_config is not None else config_base
    root = _resolve_path(root_value, base=root_base).resolve()
    if not root.exists() or not root.is_dir():
        raise CaseResolutionError("MISSING_CASE_ROOT", f"case root does not exist: {root}")
    return root, cfg


def resolve_mono_case(
    case_root: str | Path,
    *,
    workflow_kind: str = "subset_2d",
    images_dir: str | Path = ".",
    repository_root: str | Path | None = None,
) -> ResolvedCase:
    if workflow_kind not in {"subset_2d", "mesh_2d"}:
        raise CaseResolutionError("INVALID_MONO_WORKFLOW", f"mono resolver cannot resolve {workflow_kind!r}")
    root = _resolve_path(case_root, base=Path(repository_root or DEFAULT_REPOSITORY_ROOT)).resolve()
    if not root.is_dir():
        raise CaseResolutionError("MISSING_CASE_ROOT", f"case root does not exist: {root}")
    images = _image_files(_resolve_path(images_dir, base=root), natural=False)
    if len(images) < 3:
        raise CaseResolutionError(
            "INSUFFICIENT_IMAGES",
            "mono case needs a reference image, at least one deformed image, and an ROI image",
            details={"directory": str(_resolve_path(images_dir, base=root)), "count": len(images)},
        )
    frames = [FrameRef("reference", images[0])]
    frames.extend(FrameRef("deformed", path, index=index) for index, path in enumerate(images[1:-1]))
    frames.append(FrameRef("roi", images[-1]))
    return ResolvedCase(
        workflow_kind=workflow_kind,
        case_root=root,
        frames=tuple(frames),
        roi=ROIInput("explicit_image", images[-1]),
        metadata={"sort_rule": "lexical_filename", "frame_rule": "first_reference_middle_deformed_last_roi"},
    )


def resolve_stereo_case(
    case_root: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    repository_root: str | Path | None = None,
) -> ResolvedCase:
    root = _resolve_path(case_root, base=Path(repository_root or DEFAULT_REPOSITORY_ROOT)).resolve()
    cfg = dict(config or {})
    left_dir = _resolve_path(cfg.get("left_images_dir", "cam1"), base=root)
    right_dir = _resolve_path(cfg.get("right_images_dir", "cam2"), base=root)
    left_images = _image_files(left_dir)
    right_images = _image_files(right_dir)
    if len(left_images) < 2 or len(right_images) < 2:
        raise CaseResolutionError("INSUFFICIENT_STEREO_FRAMES", "stereo image directories must each contain reference and deformed images")
    frame_paths = (left_images[0], right_images[0], left_images[-1], right_images[-1])
    frame_sizes = [_image_size(path) for path in frame_paths]
    if len(set(frame_sizes)) != 1:
        raise CaseResolutionError("IMAGE_DIMENSION_MISMATCH", "stereo reference/deformed images must have compatible dimensions", details={"paths": [str(path) for path in frame_paths], "sizes": frame_sizes})
    roi_value = cfg.get("roi", "ROI.bmp")
    roi_path = _resolve_path(roi_value, base=root)
    if not roi_path.is_file():
        raise CaseResolutionError("MISSING_ROI", f"stereo ROI image does not exist: {roi_path}")
    frames = (
        FrameRef("left_reference", left_images[0]),
        FrameRef("right_reference", right_images[0]),
        FrameRef("left_deformed", left_images[-1]),
        FrameRef("right_deformed", right_images[-1]),
    )
    calibration_cfg = dict(cfg.get("calibration", {}) or {})
    left_cal = _image_files(_resolve_path(calibration_cfg.get("left_dir", "calibrate1"), base=root))
    right_cal = _image_files(_resolve_path(calibration_cfg.get("right_dir", "calibrate2"), base=root))
    if not left_cal or len(left_cal) != len(right_cal):
        raise CaseResolutionError("CALIBRATION_PAIR_MISMATCH", "stereo calibration directories must contain equal nonzero image counts", details={"left_count": len(left_cal), "right_count": len(right_cal)})
    for index, (left, right) in enumerate(zip(left_cal, right_cal)):
        if _image_size(left) != _image_size(right):
            raise CaseResolutionError("CALIBRATION_DIMENSION_MISMATCH", f"stereo calibration pair {index} has incompatible image dimensions")
    calibration = tuple(CalibrationInput(i, (left, right), label=f"pair_{i}") for i, (left, right) in enumerate(zip(left_cal, right_cal)))
    return ResolvedCase(
        workflow_kind="stereo_3d",
        case_root=root,
        frames=frames,
        roi=ROIInput("explicit_image", roi_path),
        calibration_inputs=calibration,
        metadata={
            "sort_rule": "lexical_filename",
            "stereo_roles": {"reference_disparity": "left_reference_to_right_reference", "left_temporal": "left_reference_to_left_deformed", "deformed_disparity": "left_reference_to_right_deformed"},
        },
    )


def resolve_multiview_case(
    case_root: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    repository_root: str | Path | None = None,
) -> ResolvedCase:
    root = _resolve_path(case_root, base=Path(repository_root or DEFAULT_REPOSITORY_ROOT)).resolve()
    cfg = dict(config or {})
    image_cfg = dict(cfg.get("images", {}) or {})
    image_root = _resolve_path(image_cfg.get("root", "images"), base=root)
    camera_dirs = sorted((path for path in image_root.iterdir() if path.is_dir()), key=_natural_key) if image_root.is_dir() else []
    if len(camera_dirs) < 2:
        raise CaseResolutionError("MISSING_CAMERAS", f"multiview case needs at least two camera directories under {image_root}")
    camera_ids = [path.name for path in camera_dirs]
    if len(set(camera_ids)) != len(camera_ids):
        raise CaseResolutionError("DUPLICATE_CAMERA_ID", f"duplicate camera identities under {image_root}")
    roi_cfg = dict(cfg.get("roi", {}) or {})
    roi_mode = str(roi_cfg.get("mode", "auto")).strip().lower()
    if roi_mode not in {"auto", "last_image", "explicit_image", "none"}:
        raise CaseResolutionError("INVALID_ROI_MODE", f"unsupported multiview ROI mode: {roi_mode!r}")
    cameras: list[CameraRef] = []
    for index, camera_dir in enumerate(camera_dirs):
        images = _image_files(camera_dir, natural=True)
        required = 3 if roi_mode == "last_image" else 2
        if len(images) < required:
            raise CaseResolutionError("INSUFFICIENT_CAMERA_FRAMES", f"camera {camera_dir.name} needs at least {required} images")
        reference = images[0]
        roi_frame = images[-1] if roi_mode == "last_image" else None
        deformed = images[-2] if roi_mode == "last_image" else images[-1]
        expected = [reference.name, deformed.name] + ([roi_frame.name] if roi_frame else [])
        for name in expected:
            if not (camera_dir / name).is_file():
                raise CaseResolutionError("CAMERA_FRAME_MISMATCH", f"camera {camera_dir.name} is missing required frame {name}")
        cameras.append(CameraRef(index, camera_dir.name, camera_dir.name, camera_dir, FrameRef("reference", reference), FrameRef("deformed", deformed), FrameRef("roi", roi_frame) if roi_frame else None))
    calibration_cfg = dict(cfg.get("calibration", {}) or {})
    calibration_root = _resolve_path(calibration_cfg.get("chessboard_dir", "calibrate_images"), base=root)
    if not calibration_root.is_dir():
        raise CaseResolutionError("MISSING_CALIBRATION_INPUT", f"multiview calibration directory does not exist: {calibration_root}")
    calibration_camera_ids = {path.name for path in calibration_root.iterdir() if path.is_dir()}
    if calibration_camera_ids != set(camera_ids):
        raise CaseResolutionError(
            "CALIBRATION_CAMERA_MISMATCH",
            "multiview calibration camera identities must exactly match image camera identities",
            details={"image_cameras": camera_ids, "calibration_cameras": sorted(calibration_camera_ids, key=lambda name: _natural_key(Path(name)))},
        )
    calibration: list[CalibrationInput] = []
    for camera in cameras:
        camera_cal_dir = calibration_root / camera.camera_id
        cal_images = _image_files(camera_cal_dir, natural=True)
        if not cal_images:
            raise CaseResolutionError("MISSING_CALIBRATION_INPUT", f"no calibration images for camera {camera.camera_id} under {camera_cal_dir}")
        calibration.append(CalibrationInput(camera.index, tuple(cal_images), label=camera.camera_id, camera_id=camera.camera_id))
    metadata_paths: list[Path] = []
    chessboard_meta = calibration_root / "chessboard_meta.json"
    if chessboard_meta.is_file():
        metadata_paths.append(chessboard_meta)
    explicit_roi = roi_cfg.get("path")
    roi = ROIInput(
        roi_mode,
        _resolve_path(explicit_roi, base=root) if explicit_roi else None,
        cameras[0].roi.path.name if roi_mode == "last_image" and cameras[0].roi else None,
    )
    if roi.mode == "explicit_image":
        if roi.path is None or not roi.path.is_file():
            raise CaseResolutionError("MISSING_ROI", f"multiview ROI image does not exist: {roi.path}")
    return ResolvedCase(
        workflow_kind="multiview_3d",
        case_root=root,
        cameras=tuple(cameras),
        roi=roi,
        calibration_inputs=tuple(calibration),
        calibration_metadata=tuple(metadata_paths),
        metadata={"sort_rule": "natural_camera_and_filename", "image_root": image_root, "reference_frame": cameras[0].reference.path.name, "deformed_frame": cameras[0].deformed.path.name, "camera_order": camera_ids},
    )


def _output_roots(case_config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Extract solver-specific output roots without interpreting algorithms."""
    output = case_config.get("output", {})
    if not isinstance(output, Mapping):
        return {}
    roots = output.get("solver_roots", {})
    if not isinstance(roots, Mapping):
        return {}
    resolved: dict[str, dict[str, str]] = {}
    for solver, values in roots.items():
        if not isinstance(values, Mapping):
            continue
        resolved[str(solver)] = {
            str(key): str(value)
            for key, value in values.items()
            if key in {"result_root", "visualization_root"}
        }
    return resolved


def resolve_case(
    workflow_kind: str | CaseSpec,
    *,
    case_root: str | Path | None = None,
    paths_config: str | Path | None = None,
    case_key: str | None = None,
    repository_root: str | Path | None = None,
    case_config: Mapping[str, Any] | None = None,
) -> ResolvedCase:
    repo = Path(repository_root or DEFAULT_REPOSITORY_ROOT).resolve()
    if isinstance(workflow_kind, CaseSpec):
        spec = workflow_kind
        workflow_kind, case_root, paths_config, case_key, case_config = spec.workflow_kind, spec.case_root, spec.paths_config, spec.case_key, spec.case_config
    root, cfg = _case_config(workflow_kind, case_root=case_root, paths_config=paths_config, case_key=case_key, repository_root=repo, case_config=case_config)
    if workflow_kind in {"subset_2d", "mesh_2d"}:
        resolved = resolve_mono_case(root, workflow_kind=workflow_kind, images_dir=cfg.get("images_dir", "."), repository_root=repo)
    elif workflow_kind == "stereo_3d":
        resolved = resolve_stereo_case(root, config=cfg, repository_root=repo)
    else:
        resolved = resolve_multiview_case(root, config=cfg, repository_root=repo)
    return replace(resolved, output_roots=_output_roots(cfg))


def inspect_case(*args: Any, repository_root: str | Path | None = None, **kwargs: Any) -> ValidationResult:
    try:
        resolved = resolve_case(*args, repository_root=repository_root, **kwargs)
    except CaseResolutionError as exc:
        return ValidationResult(False, (ValidationIssue(exc.code, str(exc), str(exc.details.get("path")) if exc.details.get("path") else None),))
    return ValidationResult(True, resolved=resolved)


def validate_case(*args: Any, repository_root: str | Path | None = None, **kwargs: Any) -> ValidationResult:
    return inspect_case(*args, repository_root=repository_root, **kwargs)


__all__ = [
    "WORKFLOW_KINDS", "CaseSpec", "CaseResolutionError", "ValidationIssue", "ValidationResult", "FrameRef", "ROIInput", "CalibrationInput", "CameraRef", "ResolvedCase", "resolve_case", "resolve_mono_case", "resolve_stereo_case", "resolve_multiview_case", "inspect_case", "validate_case",
]
