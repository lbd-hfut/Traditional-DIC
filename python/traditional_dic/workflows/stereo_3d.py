"""Normalized Stereo Subset-DIC workflow facade implementation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import traditional_dic as tdic
from .. import calibration as calib
from ..case import ResolvedCase
from ..config import normalize_subset_config
from ..config_resolver import ResolvedConfig
from ..stereo import (
    FIELD_DEFORMED_DISPARITY,
    FIELD_LEFT_TEMPORAL,
    FIELD_REF_DISPARITY,
    load_camera_pair,
    reconstruct_from_field_files,
)
from ..visualization import visualization_dir_for_result
from .common import WorkflowRunResult, execute_with_contract, make_context, output_paths, require_workflow

FIELD_DEFS = [
    (FIELD_REF_DISPARITY, "right_reference", "Reference disparity L0->R0"),
    (FIELD_LEFT_TEMPORAL, "left_deformed", "Temporal field L0->Llast"),
    (FIELD_DEFORMED_DISPARITY, "right_deformed", "Deformed stereo field L0->Rlast"),
]

def read_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("F")
    arr = np.array(image, dtype=np.float32, copy=True)
    max_value = float(np.max(arr))
    if max_value > 0.0:
        arr /= max_value
    return arr


def read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def color_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = np.zeros_like(values, dtype=np.float64)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def draw_colorbar(canvas: Image.Image, x0: int, y0: int, height: int, vmin: float, vmax: float, label: str) -> None:
    draw = ImageDraw.Draw(canvas)
    bar_w = 18
    for i in range(height):
        value = vmax - (vmax - vmin) * (i / max(1, height - 1))
        color = color_map(np.asarray([value], dtype=np.float64), vmin, vmax)[0]
        fill = tuple(int(v) for v in np.clip(color * 255.0, 0, 255))
        draw.line((x0, y0 + i, x0 + bar_w, y0 + i), fill=fill)
    draw.rectangle((x0, y0, x0 + bar_w, y0 + height), outline=(40, 40, 40))
    draw.text((x0 + bar_w + 8, y0 - 2), f"max {vmax:.4g}", fill=(20, 20, 20))
    draw.text((x0 + bar_w + 8, y0 + height - 12), f"min {vmin:.4g}", fill=(20, 20, 20))
    draw.text((x0, y0 + height + 10), label, fill=(20, 20, 20))


def render_scalar_field(path: Path, xy: np.ndarray, values: np.ndarray, width: int, height: int, title: str, label: str) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.min(finite)), float(np.max(finite)))
    colors = color_map(np.asarray(values, dtype=np.float64), vmin, vmax)
    image = Image.new("RGB", (width + 120, height), "white")
    draw = ImageDraw.Draw(image)
    radius = 2 if len(xy) < 20000 else 1
    for (x, y), color in zip(xy, colors):
        fill = tuple(int(v) for v in np.clip(color * 255.0, 0, 255))
        cx, cy = int(round(x)), int(round(y))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    draw.text((14, 12), f"{title} {label} min={vmin:.4g} max={vmax:.4g} px", fill=(20, 20, 20))
    draw_colorbar(image, width + 22, 42, max(80, height - 96), vmin, vmax, f"{label} px")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_field_components(out_dir: Path, stem: str, xy: np.ndarray, uv: np.ndarray, width: int, height: int, title: str) -> None:
    render_scalar_field(out_dir / f"{stem}_u.png", xy, uv[:, 0], width, height, f"{title} U", "u")
    render_scalar_field(out_dir / f"{stem}_v.png", xy, uv[:, 1], width, height, f"{title} V", "v")
    render_scalar_field(out_dir / f"{stem}_mag.png", xy, np.linalg.norm(uv, axis=1), width, height, f"{title} Mag", "|d|")


def write_result_field(path: Path, xy: np.ndarray, uv: np.ndarray, correlation: np.ndarray | None = None, valid: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    correlation = np.ones(len(xy), dtype=np.float64) if correlation is None else np.asarray(correlation, dtype=np.float64)
    valid = np.ones(len(xy), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    id_name = "node_id" if "mesh" in path.parts else "id"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id_name, "x", "y", "u", "v", "correlation", "valid"])
        for i, ((x, y), (u, v), corr, ok) in enumerate(zip(xy, uv, correlation, valid), start=1):
            writer.writerow([i, x, y, u, v, corr, int(bool(ok))])


def compact_board_dict(board) -> dict:
    return {
        "type": "chessboard",
        "rows": int(board.rows),
        "cols": int(board.cols),
        "spacing": float(board.spacing),
    }


def camera_pair_dict(result: dict, board) -> dict:
    return {
        "left": result["left"],
        "right": result["right"],
        "R_lr": result["R_lr"],
        "t_lr": result["t_lr"],
        "world_scale": 1.0,
        "board": compact_board_dict(board),
        "calibration": {
            "rms_error": result["rms_error"],
            "initial_rms_error": result["initial_rms_error"],
            "outlier_rejection_applied": result["outlier_rejection_applied"],
            "kept_pair_indices": result["kept_pair_indices"],
            "rejected_pair_indices": result["rejected_pair_indices"],
            "rejection_reasons": result["rejection_reasons"],
        },
    }


def run_calibration(paths: dict[str, Path], calibration_cfg: dict[str, Any], out_dir: Path, calibration_inputs: tuple[Any, ...]) -> Path:
    left_paths = [item.paths[0] for item in calibration_inputs]
    right_paths = [item.paths[1] for item in calibration_inputs]
    if not left_paths or len(left_paths) != len(right_paths):
        raise RuntimeError("Stereo calibration inputs must contain the same nonzero number of image pairs")

    board = calib.make_board(calibration_cfg.get("board", {}))
    stereo_cfg = dict(calibration_cfg.get("stereo_calibration", {}) or {})
    options = calib.make_stereo_options(stereo_cfg)
    result = calib.calibrate_stereo_zhang(left_paths, right_paths, board=board, options=options)
    result["board"] = compact_board_dict(board)
    result["world_scale"] = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stereo_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    camera_path = out_dir / "camera_pair.json"
    camera_path.write_text(json.dumps(camera_pair_dict(result, board), indent=2), encoding="utf-8")
    print(
        "calibration",
        json.dumps(
            {
                "initial_rms_error": result["initial_rms_error"],
                "rms_error": result["rms_error"],
                "kept_pair_indices": result["kept_pair_indices"],
                "rejected_pair_indices": result["rejected_pair_indices"],
            },
            indent=2,
        ),
    )
    return camera_path


def compute_subset_fields(paths: dict[str, Path], subset_cfg: dict[str, Any], disp_dir: Path, visualization_dir: Path) -> None:
    reference = read_gray(paths["left_reference"])
    roi = read_mask(paths["roi"])
    height, width = reference.shape
    config = normalize_subset_config(subset_cfg)
    disp_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    for field_name, image_key, title in FIELD_DEFS:
        print(f"Computing subset {title}")
        deformed = read_gray(paths[image_key])
        result = tdic.subset(reference, deformed, config=config, roi=roi)
        valid = np.asarray(result["valid"], dtype=bool)
        xy_all = np.column_stack([np.asarray(result["x"], dtype=np.float64), np.asarray(result["y"], dtype=np.float64)])
        uv_all = np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])
        write_result_field(disp_dir / field_name, xy_all, uv_all, result["correlation"], valid)
        render_field_components(visualization_dir, Path(field_name).stem, xy_all[valid], uv_all[valid], width, height, title)


def reconstruct_subset(
    paths: dict[str, Path],
    output_dirs: dict[str, Path],
    recon_cfg: dict[str, Any],
    camera_path: Path,
    visualization_root: Path | None = None,
) -> None:
    left_camera, right_camera, world_scale = load_camera_pair(camera_path)
    reconstruct_dir = output_dirs["reconstruct"]
    deformation_dir = output_dirs["deformation"]
    result = reconstruct_from_field_files(
        output_dirs["disp"],
        left_camera,
        right_camera,
        out_dir=reconstruct_dir,
        deformation_out_dir=deformation_dir,
        visualization_out_dir=(visualization_root or visualization_dir_for_result(paths["case_root"], reconstruct_dir)),
        deformation_visualization_out_dir=((visualization_root.parent / "deformation") if visualization_root is not None else visualization_dir_for_result(paths["case_root"], deformation_dir)),
        min_correlation=float(recon_cfg.get("min_correlation", 0.0)),
        quality_metric=str(recon_cfg.get("quality_metric", "correlation")),
        max_znssd=float(recon_cfg.get("max_znssd", 2.0)),
        max_reprojection_error_px=float(recon_cfg.get("max_reprojection_error_px", 5.0)),
        world_scale=world_scale,
        remove_rigid_body_motion=bool(recon_cfg.get("remove_rigid_body_motion", False)),
        write_surface_strain=bool(recon_cfg.get("strain_enabled", True)),
    )
    print(f"Reconstructed subset {result.valid_points}/{result.total_points} valid")


def _run_stereo_3d_impl(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
    calibrate: bool | None = None,
    compute_fields: bool | None = None,
    reconstruct: bool | None = None,
) -> WorkflowRunResult:
    """Run the calibrated, three-field Stereo Subset-DIC workflow."""
    context = make_context(
        resolved_case,
        resolved_config,
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
    )
    require_workflow(context, "stereo_3d")
    case = context.resolved_case
    cfg = context.resolved_config.backend_config()
    workflow_cfg = dict(cfg.get("workflow", {}) or {})
    calibration_cfg = dict(cfg.get("calibration_config", {}) or {})
    subset_cfg = dict(cfg.get("subset_config", {}) or {})
    reconstruction_cfg = dict(cfg.get("reconstruction", {}) or {})
    strain_cfg = dict(cfg.get("strain", {}) or {})
    if str(cfg.get("correspondence_solver", "subset")) != "subset":
        raise ValueError("stereo_3d facade supports Subset-DIC only")
    if str(strain_cfg.get("method", "triangular_cosserat")) != "triangular_cosserat":
        raise ValueError("stereo_3d.strain.method must be triangular_cosserat")
    reconstruction_cfg["strain_enabled"] = bool(strain_cfg.get("enabled", True))
    calibrate = bool(workflow_cfg.get("calibrate", True)) if calibrate is None else bool(calibrate)
    compute_fields = bool(workflow_cfg.get("compute_fields", False)) if compute_fields is None else bool(compute_fields)
    reconstruct = bool(workflow_cfg.get("reconstruct", True)) if reconstruct is None else bool(reconstruct)
    case_root = case.case_root
    paths = {
        "case_root": case_root,
        "left_reference": case.frame("left_reference").path,
        "right_reference": case.frame("right_reference").path,
        "left_deformed": case.frame("left_deformed").path,
        "right_deformed": case.frame("right_deformed").path,
        "roi": case.roi.path,
        "calibration_left": case.calibration_inputs[0].paths[0].parent,
        "calibration_right": case.calibration_inputs[0].paths[1].parent,
    }
    result_root, visualization_root_path = output_paths(
        context,
        "subset",
        default_result="result/subset",
        default_visualization="visualization/subset",
    )
    output_dirs = {
        "root": result_root,
        "case_root": case_root,
        "calibration": result_root / "calibration",
        "disp": result_root / "disp",
        "reconstruct": result_root / "reconstruct",
        "deformation": result_root / "deformation",
    }
    camera_path = output_dirs["calibration"] / "camera_pair.json"
    artifacts: dict[str, list[str]] = {"calibration": [], "fields": [], "reconstruction": [], "figures": []}
    if calibrate:
        camera_path = run_calibration(paths, calibration_cfg, output_dirs["calibration"], case.calibration_inputs)
        artifacts["calibration"].extend([str(output_dirs["calibration"] / "stereo_calibration.json"), str(camera_path)])
    elif not camera_path.exists():
        raise FileNotFoundError(camera_path)
    elif camera_path.is_file():
        artifacts["calibration"].append(str(camera_path))
    if compute_fields:
        compute_subset_fields(
            paths,
            subset_cfg,
            output_dirs["disp"],
            visualization_root_path / "disp",
        )
        artifacts["fields"] = [str(output_dirs["disp"] / name) for name, _, _ in FIELD_DEFS]
    else:
        existing_fields = [output_dirs["disp"] / name for name, _, _ in FIELD_DEFS]
        artifacts["fields"] = [str(path) for path in existing_fields if path.is_file()]
    if reconstruct:
        reconstruct_subset(paths, output_dirs, reconstruction_cfg, camera_path, visualization_root_path / "reconstruct")
        artifacts["reconstruction"].append(str(output_dirs["reconstruct"]))
    return WorkflowRunResult("stereo_3d", result_root, artifacts)


def run_stereo_3d(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
    calibrate: bool | None = None,
    compute_fields: bool | None = None,
    reconstruct: bool | None = None,
    run_id: str | None = None,
) -> WorkflowRunResult:
    """Run the fixed Subset-only Stereo workflow with normalized F4 metadata."""
    return execute_with_contract(
        "stereo_3d",
        resolved_case,
        resolved_config,
        solver_root="subset",
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
        run_id=run_id,
        runner=lambda: _run_stereo_3d_impl(
            resolved_case,
            resolved_config,
            repository_root=repository_root,
            output_root=output_root,
            visualization_root=visualization_root,
            calibrate=calibrate,
            compute_fields=compute_fields,
            reconstruct=reconstruct,
        ),
    )
