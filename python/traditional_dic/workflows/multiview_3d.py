"""Normalized Multiview Subset-DIC workflow facade implementation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .. import calibration
from ..case import ResolvedCase
from ..config_resolver import ResolvedConfig
from .._runtime import resolve_config_reference
from ..multiview import (
    compute_pairwise_2d_dic,
    compute_pairwise_3d_dic,
    generate_pair_masks_from_calibration,
    recover_multiview_calibration_scale,
    save_pair_selection_report,
    select_camera_pairs,
    stitch_pairwise_3d_surfaces,
)
from ..visualization import visualization_dir_for_result
from .common import WorkflowRunResult, execute_with_contract, make_context, output_paths, require_workflow

def _save_manual_camera_masks(case: ResolvedCase, roi_frame: str, mask_dir: Path) -> None:
    target = mask_dir / "mask"
    target.mkdir(parents=True, exist_ok=True)
    for camera in case.cameras:
        mask = np.asarray(Image.open(camera.path / roi_frame).convert("L")) > 0
        np.save(target / f"{camera.camera_id}_mask.npy", mask)


def _normalize_camera_labels(calibration_data: dict[str, Any], image_paths: Sequence[Path]) -> None:
    for idx, camera in enumerate(calibration_data.get("cameras", []) or []):
        if idx < len(image_paths):
            camera["label"] = image_paths[idx].parent.name


def _save_observation_npz(calibration_data: Mapping[str, Any], path: Path) -> None:
    cam_indices: list[int] = []
    point_indices: list[int] = []
    uv: list[list[float]] = []
    for point_idx, point in enumerate(calibration_data.get("points3d", []) or []):
        for obs in point.get("observations", []) or []:
            xy = obs.get("uv", obs.get("xy", obs.get("point")))
            if xy is None:
                continue
            arr = np.asarray(xy, dtype=np.float64).reshape(-1)
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                cam_indices.append(int(obs.get("camera_index", obs.get("image_index", -1))))
                point_indices.append(int(point.get("point3d_id", point_idx)))
                uv.append([float(arr[0]), float(arr[1])])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cam_indices=np.asarray(cam_indices, dtype=np.int64),
        point_indices=np.asarray(point_indices, dtype=np.int64),
        uv=np.asarray(uv, dtype=np.float64).reshape((-1, 2)),
    )


def _save_calibration_products(calibration_data: dict[str, Any], calibration_dir: Path, image_paths: Sequence[Path]) -> None:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    _normalize_camera_labels(calibration_data, image_paths)
    calibration.save_json(calibration_data, calibration_dir / "calibration_result.json")
    _save_observation_npz(calibration_data, calibration_dir / "observations.npz")
    summary = {
        "image_paths": [str(path) for path in image_paths],
        "camera_labels": [camera.get("label", f"cam_{idx}") for idx, camera in enumerate(calibration_data.get("cameras", []) or [])],
        "camera_count": len(calibration_data.get("cameras", []) or []),
        "sparse_point_count": len(calibration_data.get("points3d", []) or []),
        "observation_count": int(sum(len(point.get("observations", []) or []) for point in calibration_data.get("points3d", []) or [])),
        "mean_reprojection_error": calibration_data.get("mean_reprojection_error"),
    }
    (calibration_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _axis_limits(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.vstack([points, centers]) if len(points) else centers
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    pad = 0.08 * max(float(np.max(hi - lo)), 1.0)
    return lo - pad, hi + pad


def _apply_3d_limits(ax, limits: tuple[np.ndarray, np.ndarray]) -> None:
    lo, hi = limits
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass


def _calibration_visualization_data(
    calibration_data: Mapping[str, Any],
    image_paths: Sequence[Path],
) -> dict[str, Any]:
    cameras = list(calibration_data.get("cameras", []) or [])
    points = list(calibration_data.get("points3d", []) or [])
    cam_names = [str(camera.get("label", f"cam_{idx}")) for idx, camera in enumerate(cameras)]
    K = np.asarray([camera.get("K") for camera in cameras], dtype=np.float64)
    R = np.asarray([camera.get("R") for camera in cameras], dtype=np.float64)
    t = np.asarray([camera.get("t") for camera in cameras], dtype=np.float64).reshape((len(cameras), 3))
    centers = np.asarray(
        [camera.get("camera_center", (-R[idx].T @ t[idx]).tolist()) for idx, camera in enumerate(cameras)],
        dtype=np.float64,
    )
    xyz = np.asarray([point.get("xyz") for point in points], dtype=np.float64).reshape((-1, 3))
    obs_cam: list[int] = []
    obs_point: list[int] = []
    obs_uv: list[list[float]] = []
    for fallback_idx, point in enumerate(points):
        for obs in point.get("observations", []) or []:
            uv = np.asarray(obs.get("uv", obs.get("xy", [])), dtype=np.float64).reshape(-1)
            if uv.size >= 2 and np.all(np.isfinite(uv[:2])):
                obs_cam.append(int(obs.get("camera_index", obs.get("image_index", -1))))
                obs_point.append(fallback_idx)
                obs_uv.append([float(uv[0]), float(uv[1])])
    return {
        "cam_names": cam_names,
        "image_paths": [Path(path) for path in image_paths],
        "coordinate_system": str(calibration_data.get("coordinate_system", "sfm")),
        "sfm_to_world_scale": calibration_data.get("sfm_to_world_scale"),
        "K": K,
        "R": R,
        "t": t,
        "centers": centers,
        "points": xyz,
        "observations": {
            "cam_indices": np.asarray(obs_cam, dtype=np.int32),
            "point_indices": np.asarray(obs_point, dtype=np.int64),
            "uv": np.asarray(obs_uv, dtype=np.float64).reshape((-1, 2)),
        },
    }


def _save_calibration_visualization_products(
    data: Mapping[str, Any],
    out_dir: Path,
    *,
    model_filename: str = "colmap_like_sparse_model.npz",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    observations = data["observations"]
    np.savez_compressed(
        out_dir / model_filename,
        cam_names=np.asarray(data["cam_names"]),
        K=np.asarray(data["K"]),
        R=np.asarray(data["R"]),
        t=np.asarray(data["t"]),
        camera_centers_world=np.asarray(data["centers"]),
        points3D=np.asarray(data["points"]),
        observation_cam_indices=np.asarray(observations["cam_indices"]),
        observation_point_indices=np.asarray(observations["point_indices"]),
        observation_uv=np.asarray(observations["uv"]),
    )
    summary = {
        "coordinate_system": str(data.get("coordinate_system", "sfm")),
        "sfm_to_world_scale": data.get("sfm_to_world_scale"),
        "num_cameras": len(data["cam_names"]),
        "num_sparse_points": int(len(data["points"])),
        "num_observations": int(len(observations["cam_indices"])),
        "per_camera_observations": {
            name: int(np.sum(observations["cam_indices"] == idx)) for idx, name in enumerate(data["cam_names"])
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plot_sparse_scene(data: Mapping[str, Any], out_dir: Path, dpi: int = 160) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(data["points"], dtype=np.float64)
    centers = np.asarray(data["centers"], dtype=np.float64)
    limits = _axis_limits(points, centers)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        sample_count = min(len(points), 30000)
        sample_idx = np.random.RandomState(0).choice(len(points), sample_count, replace=False)
        sample = points[sample_idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.0, c="0.25", alpha=0.55)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=45, marker="^")
    scale = 0.08 * max(limits[1] - limits[0])
    for idx, center in enumerate(centers):
        ax.text(center[0], center[1], center[2], data["cam_names"][idx], fontsize=7)
        axes = scale * np.asarray(data["R"], dtype=np.float64)[idx].T
        for axis, color in zip(axes.T, ["r", "g", "b"]):
            ax.quiver(center[0], center[1], center[2], axis[0], axis[1], axis[2], color=color, linewidth=1)
    _apply_3d_limits(ax, limits)
    coordinate_system = str(data.get("coordinate_system", "sfm"))
    title = "Metric-scaled sparse points and camera poses" if coordinate_system == "metric_scaled" else "Sparse points and camera poses"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    out_path = out_dir / "sparse_scene.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_camera_observations_3d(data: Mapping[str, Any], out_dir: Path, dpi: int = 160) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(data["points"], dtype=np.float64)
    centers = np.asarray(data["centers"], dtype=np.float64)
    observations = data["observations"]
    limits = _axis_limits(points, centers)
    fig = plt.figure(figsize=(16, 12))
    for idx, name in enumerate(data["cam_names"]):
        ax = fig.add_subplot(3, 4, idx + 1, projection="3d")
        mask = observations["cam_indices"] == idx
        point_indices = observations["point_indices"][mask]
        valid = point_indices[(point_indices >= 0) & (point_indices < len(points))]
        observed_points = points[valid] if len(valid) else np.zeros((0, 3))
        if len(observed_points):
            ax.scatter(observed_points[:, 0], observed_points[:, 1], observed_points[:, 2], s=2, c="tab:red", alpha=0.75)
        _apply_3d_limits(ax, limits)
        ax.set_title(f"{name}: {len(observed_points)} obs", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    out_path = out_dir / "camera_observations_3d.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_camera_observations_2d(data: Mapping[str, Any], out_dir: Path, dpi: int = 160) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    observations = data["observations"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes_flat = axes.ravel()
    for idx, name in enumerate(data["cam_names"]):
        ax = axes_flat[idx]
        image = mpimg.imread(data["image_paths"][idx])
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        mask = observations["cam_indices"] == idx
        uv = observations["uv"][mask]
        if len(uv):
            draw_count = min(len(uv), 1200)
            draw_idx = np.random.RandomState(idx).choice(len(uv), draw_count, replace=False)
            draw_uv = uv[draw_idx]
            ax.scatter(draw_uv[:, 0], draw_uv[:, 1], s=8, c="red", marker="o", linewidths=0, alpha=0.7)
        ax.set_title(f"{name}: {len(uv)} obs", fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
    for idx in range(len(data["cam_names"]), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.tight_layout()
    out_path = out_dir / "camera_observations_2d.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_calibration_visualization(
    calibration_data: Mapping[str, Any],
    case_root: Path,
    calibration_dir: Path,
    image_paths: Sequence[Path],
    visualization_root: Path | None = None,
) -> dict[str, str]:
    out_dir = visualization_root or visualization_dir_for_result(case_root, calibration_dir)
    data = _calibration_visualization_data(calibration_data, image_paths)
    _save_calibration_visualization_products(data, out_dir)
    outputs = {
        "sparse_scene": str(_plot_sparse_scene(data, out_dir)),
        "camera_observations_3d": str(_plot_camera_observations_3d(data, out_dir)),
        "camera_observations_2d": str(_plot_camera_observations_2d(data, out_dir)),
        "coordinate_system": str(data.get("coordinate_system", "sfm")),
        "sfm_to_world_scale": data.get("sfm_to_world_scale"),
    }
    (out_dir / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


def _absolute_config_path(value: Any, repository_root: Path) -> Any:
    """Make nested legacy config references independent of execution cwd."""
    if not isinstance(value, (str, Path)):
        return value
    return str(resolve_config_reference(value, repository_root).resolve())


def _scaled_calibration_data(
    calibration_data: Mapping[str, Any],
    scale_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the visualization/export view in metric scaled coordinates."""
    scaled = dict(calibration_data)
    scaled["cameras"] = list(scale_data.get("scaled_cameras", []) or [])
    scaled["points3d"] = list(scale_data.get("scaled_points3d", []) or [])
    scaled["sfm_to_world_scale"] = float(scale_data.get("sfm_to_world_scale", 1.0))
    scaled["world_to_sfm_scale"] = float(scale_data.get("world_to_sfm_scale", 1.0))
    scaled["coordinate_system"] = "metric_scaled"
    if not scaled["cameras"]:
        raise RuntimeError("Scale recovery produced no scaled cameras")
    if not scaled["points3d"]:
        # The metric-meta fallback path (cameras taken from chessboard_meta.json,
        # scale 1.0) emits scaled_cameras but no scaled sparse points. Rebuild
        # them from the raw SfM points so the exported visualization and the
        # sparse-point count stay populated.
        factor = float(scale_data.get("sfm_to_world_scale", 1.0))
        rebuilt: list[dict[str, Any]] = []
        for raw in calibration_data.get("points3d", []) or []:
            pt = dict(raw)
            xyz = pt.get("xyz", pt.get("point"))
            if xyz is not None:
                pt["xyz"] = [float(v) * factor for v in xyz]
            rebuilt.append(pt)
        scaled["points3d"] = rebuilt
    return scaled



def _run_multiview_3d_impl(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
    resume: bool = False,
) -> WorkflowRunResult:
    """Run the deterministic Multiview Subset-DIC workflow."""
    context = make_context(
        resolved_case,
        resolved_config,
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
    )
    require_workflow(context, "multiview_3d")
    case = context.resolved_case
    cfg = context.resolved_config.backend_config()
    if cfg.get("pairwise_2d_solver") != "subset":
        raise ValueError("multiview_3d facade supports Subset-DIC only")
    result_root, visualization_root_path = output_paths(
        context,
        "subset",
        default_result="result/subset",
        default_visualization="visualization/subset",
    )
    # Existing multiview stage APIs accept a case root and resolve nested
    # config/output references against it.  The facade supplies absolute
    # references so an explicit output root can be outside the source case.
    execution_root = case.case_root if context.output_root is None else result_root
    image_root = case.cameras[0].path.parent
    calibration_root = case.calibration_inputs[0].paths[0].parent.parent
    reference_frame = case.cameras[0].reference.path.name
    deformed_frame = case.cameras[0].deformed.path.name
    manual_roi = case.roi.mode == "last_image"
    roi_frame = case.cameras[0].roi.path.name if manual_roi and case.cameras[0].roi else None
    runtime_cfg = dict(cfg)
    runtime_cfg["output"] = {
        "mask_dir": str(result_root / "mask"),
        "roi_dir": str(result_root / "mask"),
        "calibration_dir": str(result_root / "calibration"),
        "disp_dir": str(result_root / "disp"),
        "reconstruct_dir": str(result_root / "reconstruct"),
    }
    runtime_cfg["pairwise_2d_dic"] = dict(cfg.get("pairwise_2d_dic", {}) or {})
    runtime_cfg["pairwise_2d_dic"].update(
        {
            "image_dir": str(image_root),
            "reference_frame": reference_frame,
            "deformed_frame": deformed_frame,
            "subset_config": _absolute_config_path(runtime_cfg["pairwise_2d_dic"].get("subset_config", "config/subset_2d.yaml"), context.repository_root),
            "mesh_config": _absolute_config_path(runtime_cfg["pairwise_2d_dic"].get("mesh_config", "config/mesh_2d.yaml"), context.repository_root),
            "run_subset": True,
            "run_mesh": False,
        }
    )
    if manual_roi:
        runtime_cfg["pairwise_2d_dic"]["pair_roi_mode"] = "left_mask"
    runtime_cfg["scale"] = dict(cfg.get("scale", {}) or {})
    runtime_cfg["scale"].update(
        {
            "calibration_dir": str(result_root / "calibration"),
            "chessboard_dir": str(calibration_root),
        }
    )
    runtime_cfg["pairwise_3d_dic"] = dict(cfg.get("pairwise_3d_dic", {}) or {})
    runtime_cfg["pairwise_3d_dic"].update(
        {
            "solver": "subset",
            "field_dir": str(result_root / "disp"),
            "output_dir": str(result_root / "reconstruct" / "pairwise"),
            "calibration_dir": str(result_root / "calibration"),
            "write_surface_strain": False,
        }
    )
    runtime_cfg["surface_stitch"] = dict(cfg.get("surface_stitch", {}) or {})
    runtime_cfg["surface_stitch"].update(
        {
            "solver": "subset",
            "pairwise_3d_dir": str(result_root / "reconstruct" / "pairwise"),
            "output_dir": str(result_root / "reconstruct" / "stitched"),
            "calibration_dir": str(result_root / "calibration"),
        }
    )
    calibration_dir = result_root / "calibration"
    mask_dir = result_root / "mask"
    if manual_roi:
        _save_manual_camera_masks(case, str(roi_frame), mask_dir)
    image_paths = [camera.reference.path for camera in case.cameras]
    calibration_data = calibration.calibrate_multiview_colmap_like(image_paths, config=runtime_cfg)
    _save_calibration_products(calibration_data, calibration_dir, image_paths)
    scale_result = recover_multiview_calibration_scale(execution_root, calibration_data, config=runtime_cfg)
    scale_path = calibration_dir / "calibration_scale.json"
    scaled_path = calibration_dir / "calibration_result_scaled.json"
    scale_data = json.loads(scale_path.read_text(encoding="utf-8"))
    scaled_calibration_data = _scaled_calibration_data(calibration_data, scale_data)
    visualization_outputs = _save_calibration_visualization(
        scaled_calibration_data,
        case.case_root,
        calibration_dir,
        image_paths,
        visualization_root_path / "calibration",
    )
    visualization_outputs["coordinate_system"] = "metric_scaled"
    visualization_outputs["sfm_to_world_scale"] = str(scale_result.sfm_to_world_scale)
    pair_selection = select_camera_pairs(calibration_data, runtime_cfg.get("camera_pair_selection"))
    pair_selection_path = calibration_dir / "pair_selection_report.json"
    save_pair_selection_report(pair_selection, pair_selection_path)
    mask_result = generate_pair_masks_from_calibration(
        execution_root,
        calibration_data,
        config=runtime_cfg,
        pair_selection=pair_selection,
        output_dir=mask_dir,
    )
    pairwise_2d_config = dict(runtime_cfg.get("pairwise_2d_dic", {}) or {})
    if resume:
        pairwise_2d_config["overwrite"] = False
    pairwise_3d_config = dict(runtime_cfg.get("pairwise_3d_dic", {}) or {})
    stitch_config = dict(runtime_cfg.get("surface_stitch", {}) or {})
    pairwise_2d = compute_pairwise_2d_dic(
        execution_root,
        calibration_data,
        config=runtime_cfg,
        pair_selection=pair_selection,
        options=pairwise_2d_config,
    )
    pairwise_3d = compute_pairwise_3d_dic(
        execution_root,
        calibration_data,
        config=runtime_cfg,
        pair_selection=pair_selection,
        options=pairwise_3d_config,
    )
    stitched_surface = stitch_pairwise_3d_surfaces(
        execution_root,
        config=runtime_cfg,
        pair_selection=pair_selection,
        options=stitch_config,
    )
    summary = {
        "case_root": str(case.case_root),
        "calibration_dir": str(calibration_dir),
        "mask_dir": mask_result.mask_dir,
        "camera_count": len(calibration_data.get("cameras", []) or []),
        "sparse_point_count": len(calibration_data.get("points3d", []) or []),
        "scaled_sparse_point_count": len(scaled_calibration_data.get("points3d", []) or []),
        "sfm_to_world_scale": float(scale_result.sfm_to_world_scale),
        "world_to_sfm_scale": float(scale_result.world_to_sfm_scale),
        "scale_result": str(scale_path),
        "scaled_calibration_result": str(scaled_path),
        "scale_valid_edges": int(scale_result.valid_edges),
        "scale_triangulated_corners": int(scale_result.triangulated_corners),
        "pair_selection_report": str(pair_selection_path),
        "calibration_visualization": visualization_outputs,
        "selected_pairs": [list(pair) for pair in mask_result.pairs],
        "master_camera_convention": "For mask_cam_i_cam_j, cam_i is the master camera used for ROI, 2D-DIC, and pair triangulation.",
        "pairs": [list(pair) for pair in mask_result.pairs],
        "overview_roi": mask_result.overview_roi,
        "overview_overplay": mask_result.overview_overplay,
        "pairwise_2d_dic": asdict(pairwise_2d),
        "pairwise_3d_dic": asdict(pairwise_3d),
        "surface_stitch": asdict(stitched_surface),
        "requested_solver": "subset",
    }
    summary_path = result_root / "multiview_3d_run.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return WorkflowRunResult(
        "multiview_3d",
        result_root,
        {
            "summary": str(summary_path),
            "calibration": str(calibration_dir),
            "masks": str(mask_dir),
            "pairwise_2d": str(result_root / "disp"),
            "pairwise_3d": str(result_root / "reconstruct" / "pairwise"),
            "stitching": str(result_root / "reconstruct" / "stitched"),
        },
    )


def run_multiview_3d(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
    resume: bool = False,
    run_id: str | None = None,
) -> WorkflowRunResult:
    """Run the fixed Subset-only Multiview workflow with normalized F4 metadata."""
    return execute_with_contract(
        "multiview_3d",
        resolved_case,
        resolved_config,
        solver_root="subset",
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
        run_id=run_id,
        runner=lambda: _run_multiview_3d_impl(
            resolved_case,
            resolved_config,
            repository_root=repository_root,
            output_root=output_root,
            visualization_root=visualization_root,
            resume=resume,
        ),
    )
