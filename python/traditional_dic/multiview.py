"""Multiview 3D-DIC workflow helpers."""

from __future__ import annotations

import json
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_PAIR_NEIGHBOR_ORDER = 1


def _natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


@dataclass
class MultiviewMaskOptions:
    external_threshold: int = 127
    outlier_k: int = 6
    outlier_knn_scale: float = 4.0
    component_radius_scale: float = 8.0
    edge_scale: float = 8.0
    radius_scale: float = 6.0
    min_hole_area: int = 500
    tiny_hole_fill_area: int = 3000
    speckle_std_ratio: float = 0.35
    speckle_lap_ratio: float = 0.35
    speckle_grad_ratio: float = 0.35
    min_speckle_std: float = 6.0
    min_speckle_lap: float = 3.0
    overlay_alpha: float = 0.45


@dataclass
class CameraMaskResult:
    camera_index: int
    camera_label: str
    mask: np.ndarray
    hull_mask: np.ndarray
    supported_mask: np.ndarray
    rejected_hole_mask: np.ndarray
    observations: np.ndarray
    clean_observations: np.ndarray
    n_triangles_raw: int = 0
    n_triangles_valid: int = 0
    n_holes_detected: int = 0
    n_holes_filled_as_speckle: int = 0
    n_holes_rejected: int = 0


@dataclass
class CameraPairSelectionOptions:
    mode: str = "auto_spatial"
    wrap: bool = True
    manual: Sequence[Sequence[int | str]] | None = None
    auto_circularity_threshold: float = 0.45
    auto_wrap_distance_ratio: float = 1.8
    auto_wrap_min_shared_ratio: float = 0.35
    auto_max_neighbor_distance_ratio: float = 2.0
    auto_min_shared_tracks: int = 20


@dataclass
class CameraPairSelectionResult:
    mode: str
    pairs: list[tuple[int, int]]
    pair_names: list[tuple[str, str]]
    spatial_order: list[int]
    spatial_order_names: list[str]
    circularity: float | None = None
    is_circular: bool = False
    neighbor_order: int = _PAIR_NEIGHBOR_ORDER
    reconstruction_policy: str = "pair_level_stereo_surface"
    pair_distances: dict[str, float] | None = None
    shared_track_counts: dict[str, int] | None = None
    rejected_pairs: list[dict[str, Any]] | None = None
    thresholds: dict[str, float | int] | None = None


@dataclass
class PairwiseDICOptions:
    run_subset: bool = True
    run_mesh: bool = True
    mesh_types: tuple[str, ...] = ("T3", "Q4", "Q8")
    reference_frame: str = "001.bmp"
    deformed_frame: str = "002.bmp"
    image_dir: str = "images"
    mask_dir: str = "result/mask"
    roi_dir: str = "result/mask"
    calibration_dir: str = "result/calibration"
    output_dir: str = "result/disp"
    subset_config: str | None = None
    mesh_config: str | None = None
    mesh_target_element_size: float = 35.0
    pair_roi_mode: str = "sparse_overlap"
    pair_roi_min_common_tracks: int = 20
    pair_roi_disparity_quantile: float = 0.0
    pair_roi_erode_pixels: int = 0
    pair_roi_support: str = "convex_hull"
    pair_roi_robust_outlier: bool = False
    pair_roi_outlier_mad_scale: float = 4.5
    pair_roi_alpha_radius_scale: float = 8.0
    pair_roi_source: str = "pair_sift"
    maskgen_feature_method: str = "sift"
    maskgen_max_features: int = 8000
    maskgen_match_ratio: float = 0.75
    maskgen_mutual_check: bool = True
    maskgen_ransac_reprojection_threshold: float = 3.0
    maskgen_min_matches: int = 20
    subset_solve_roi_mode: str = "left_mask"
    max_pairs: int | None = None
    overwrite: bool = True


@dataclass
class PairwiseDICRunResult:
    output_dir: str
    pair_dirs: list[str]
    pairs: list[tuple[str, str]]
    run_subset: bool
    run_mesh: bool
    mesh_types: list[str]


@dataclass
class Pairwise3DOptions:
    solver: str = "subset"
    field_dir: str = "result/disp"
    output_dir: str = "result/reconstruct/pairwise"
    calibration_dir: str = "result/calibration"
    quality_metric: str = "znssd"
    max_znssd: float = 2.0
    min_correlation: float = 0.0
    max_reprojection_error_px: float = 5.0
    world_scale: float = 1.0
    remove_rigid_body_motion: bool = False
    write_shape_maps: bool = True
    write_deformation_maps: bool = True
    write_surface_strain: bool = True
    max_pairs: int | None = None


@dataclass
class MultiviewScaleRecoveryOptions:
    calibration_dir: str = "result/calibration"
    chessboard_dir: str = "calibrate_images"
    output_file: str = "calibration_scale.json"
    use_meta_observations: bool = True
    board_rows: int = 7
    board_cols: int = 9
    square_size: float = 10.0
    max_reprojection_error: float = 3.0
    trim_fraction: float = 0.20
    min_common_corners: int = 12
    allow_meta_camera_model_fallback: bool = True


@dataclass
class Pairwise3DRunResult:
    output_dir: str
    pair_dirs: list[str]
    pairs: list[tuple[str, str]]
    solver: str
    total_points: int
    valid_points: int


@dataclass
class MultiviewScaleRecoveryResult:
    output_file: str
    sfm_to_world_scale: float
    world_to_sfm_scale: float
    valid_edges: int
    triangulated_corners: int


@dataclass
class PairwiseSurfaceStitchOptions:
    solver: str = "subset"
    pairwise_3d_dir: str = "result/reconstruct/pairwise"
    output_dir: str = "result/reconstruct/stitched"
    calibration_dir: str = "result/calibration"
    mode: str = "multidic"
    max_reprojection_error_px: float = 5.0
    max_quality: float = 2.0
    triangle_edge_scale: float = 3.0
    min_gap_factor: float = 0.2
    outlier_neighbor_count: int = 8
    outlier_distance_sigma: float = 6.0
    outlier_displacement_sigma: float = 6.0
    outlier_face_edge_scale: float = 4.0
    min_valid_points: int = 3
    max_pairs: int | None = None
    smooth_displacement_knn: int = 0


@dataclass
class PairwiseSurfaceStitchRunResult:
    output_dir: str
    pairs: list[tuple[str, str]]
    solver: str
    mode: str
    point_count: int
    face_count: int


@dataclass
class PairMaskGenerationResult:
    mask_dir: str
    pairs: list[tuple[str, str]]
    mask_files: list[str]
    overplay_files: list[str]
    overview_roi: str | None
    overview_overplay: str | None


def generate_pair_masks_from_calibration(
    case_root: str | Path,
    calibration: Any | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    pair_selection: CameraPairSelectionResult | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    options: PairwiseDICOptions | Mapping[str, Any] | None = None,
) -> PairMaskGenerationResult:
    """Generate pair ROI masks from pairwise reference-image SIFT matches."""

    case_root = Path(case_root)
    mv_cfg = _load_multiview_config(config, case_root)
    opts = _coerce_pairwise_dic_options(options, mv_cfg)
    if output_dir is not None:
        opts.roi_dir = str(output_dir)

    calibration_data = calibration
    if calibration_data is None:
        calibration_path = _case_path(case_root, opts.calibration_dir) / "calibration_result.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration result not found: {calibration_path}")
        calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))

    pairs = _resolve_pair_names(case_root, calibration_data, pair_selection, opts)
    if opts.max_pairs is not None:
        pairs = pairs[: max(0, int(opts.max_pairs))]

    mask_root = _case_path(case_root, opts.roi_dir)
    if bool(opts.overwrite):
        _clear_pair_mask_outputs(mask_root)
    roi_overview_items: list[tuple[str, Path]] = []
    overplay_overview_items: list[tuple[str, Path]] = []
    mask_files: list[str] = []
    overplay_files: list[str] = []

    for left_name, right_name in pairs:
        pair_label = f"{left_name}-{right_name}"
        mask_stem = f"mask_{left_name}_{right_name}"
        paths = _pair_image_paths(case_root, opts, left_name, right_name)
        reference = _read_gray_image(paths["left_reference"])
        paired_reference = _read_gray_image(paths["right_reference"])
        height, width = reference.shape
        left_mask = _load_optional_camera_mask(case_root, opts, left_name, height, width)
        common_left_uv, common_right_uv, match_meta = _match_pair_roi_features(reference, paired_reference, opts)
        roi, roi_meta = _load_or_build_pair_roi(
            case_root,
            opts,
            left_name,
            right_name,
            left_mask,
            height,
            width,
            common_uv_pairs=(common_left_uv, common_right_uv, "pair_sift_reference_match", match_meta),
        )
        _save_pair_roi(
            mask_root,
            left_name,
            right_name,
            roi,
            left_mask,
            common_left_uv,
            common_right_uv,
            reference,
            paired_reference,
            roi_meta,
        )
        roi_path = mask_root / "roi" / f"{mask_stem}.png"
        overplay_path = mask_root / "overplay" / f"{mask_stem}_overplay.png"
        roi_overview_items.append((pair_label, roi_path))
        overplay_overview_items.append((pair_label, overplay_path))
        mask_files.append(str(roi_path))
        overplay_files.append(str(overplay_path))

    _save_mask_overviews(mask_root, roi_overview_items, overplay_overview_items)
    overview_roi = mask_root / "overview_roi.png"
    overview_overplay = mask_root / "overview_overplay.png"
    return PairMaskGenerationResult(
        mask_dir=str(mask_root),
        pairs=pairs,
        mask_files=mask_files,
        overplay_files=overplay_files,
        overview_roi=str(overview_roi) if overview_roi.exists() else None,
        overview_overplay=str(overview_overplay) if overview_overplay.exists() else None,
    )


def generate_masks_from_calibration(
    calibration: Any,
    image_shapes: Mapping[int | str, tuple[int, int]] | Sequence[tuple[int, int]] | None = None,
    *,
    reference_images: Mapping[int | str, str | Path | np.ndarray] | Sequence[str | Path | np.ndarray] | None = None,
    options: MultiviewMaskOptions | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> list[CameraMaskResult]:
    """Generate per-camera ROI masks from multiview calibration observations.

    The mask is based on each camera's own observed 2D feature coordinates in
    the sparse calibration result.  It intentionally does not project every 3D
    point into every camera.
    """

    return build_masks_from_calibration(
        calibration,
        image_shapes,
        reference_images=reference_images,
        options=options,
        output_dir=output_dir,
    )


def build_masks_from_calibration(
    calibration: Any,
    image_shapes: Mapping[int | str, tuple[int, int]] | Sequence[tuple[int, int]] | None = None,
    *,
    reference_images: Mapping[int | str, str | Path | np.ndarray] | Sequence[str | Path | np.ndarray] | None = None,
    options: MultiviewMaskOptions | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> list[CameraMaskResult]:
    """Build per-camera masks through the C++ backend, with Python fallback."""

    opts = _coerce_mask_options(options)
    cameras = _extract_cameras(calibration)
    observations = collect_observations_by_camera(calibration, len(cameras))
    images = _load_reference_images(reference_images, len(cameras))
    shapes = _resolve_image_shapes(cameras, image_shapes, images)

    results = _build_masks_with_cpp_backend(cameras, observations, shapes, opts)
    if results is None:
        results = _build_masks_with_python_fallback(
            calibration,
            image_shapes,
            cameras=cameras,
            observations=observations,
            shapes=shapes,
            images=images,
            reference_images=reference_images,
            options=opts,
        )

    if output_dir is not None:
        save_multiview_masks(results, output_dir, reference_images=images, options=opts)
    return results


def collect_observations_by_camera(calibration: Any, camera_count: int | None = None) -> dict[int, np.ndarray]:
    """Collect true per-camera 2D observations from SparsePoint3D tracks."""

    points = _extract_sparse_points(calibration)
    out: dict[int, list[list[float]]] = {}
    max_camera = -1
    for point in points:
        for cam_idx, uv in _point_observations(point):
            if cam_idx < 0:
                continue
            max_camera = max(max_camera, cam_idx)
            if np.all(np.isfinite(uv)):
                out.setdefault(cam_idx, []).append([float(uv[0]), float(uv[1])])
    if camera_count is None:
        camera_count = max_camera + 1
    return {
        idx: np.asarray(out.get(idx, []), dtype=np.float64).reshape((-1, 2))
        for idx in range(int(camera_count))
    }


def select_camera_pairs(
    calibration: Any,
    options: CameraPairSelectionOptions | Mapping[str, Any] | None = None,
) -> CameraPairSelectionResult:
    """Select camera pairs for pair-surface reconstruction.

    Default ``auto_spatial`` follows the Multi-DIC policy: spatially order
    cameras and select fixed first-order adjacent stereo-pair edges, optionally
    closing the ring.  ``auto_natural`` uses the natural camera labels order
    (``cam_0, cam_1, ...``) with the same fixed first-order adjacent edges.
    The downstream reconstruction policy is pair-level stereo surface
    reconstruction and stitching, not N-view joint triangulation.
    """

    opts = _coerce_pair_options(options)
    cameras = _extract_cameras(calibration)
    cam_names = [_camera_label(camera, idx) for idx, camera in enumerate(cameras)]
    centers = _camera_centers(cameras)
    shared = _shared_track_counts(calibration, len(cameras))
    mode = opts.mode.lower()

    if mode == "auto":
        mode = "auto_spatial"
    if mode == "manual":
        pairs = _manual_camera_pairs(opts.manual or [], cam_names)
        return _pair_selection_result(mode, pairs, cam_names, centers, shared, opts, spatial_order=[])
    if mode in {"auto_natural", "auto_label", "auto_index"}:
        order = _camera_name_order(cam_names)
        pairs = _adjacent_pairs(order, bool(opts.wrap))
        return _pair_selection_result(
            "auto_natural",
            pairs,
            cam_names,
            centers,
            shared,
            opts,
            spatial_order=order,
            is_circular=bool(opts.wrap and len(order) > 2),
        )
    order = _camera_name_order(cam_names)
    if mode == "auto_spatial":
        pairs, spatial_order, circularity, is_circular, rejected = _auto_spatial_pairs(cam_names, centers, shared, opts)
        return _pair_selection_result(
            mode,
            pairs,
            cam_names,
            centers,
            shared,
            opts,
            spatial_order=spatial_order,
            circularity=circularity,
            is_circular=is_circular,
            rejected_pairs=rejected,
        )
    raise ValueError("Unknown pair selection mode: expected manual, auto, or auto_spatial.")


def save_pair_selection_report(result: CameraPairSelectionResult, path: str | Path) -> None:
    """Write a JSON report for camera-pair selection."""

    payload = {
        "mode": result.mode,
        "pairs": [[int(a), int(b)] for a, b in result.pairs],
        "pair_names": [[a, b] for a, b in result.pair_names],
        "spatial_order": [int(i) for i in result.spatial_order],
        "spatial_order_names": result.spatial_order_names,
        "circularity": result.circularity,
        "is_circular": bool(result.is_circular),
        "neighbor_order": int(result.neighbor_order),
        "reconstruction_policy": result.reconstruction_policy,
        "pair_distances": result.pair_distances or {},
        "shared_track_counts": result.shared_track_counts or {},
        "rejected_pairs": result.rejected_pairs or [],
        "thresholds": result.thresholds or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_PAIR_FIELD_DEFS = (
    ("reference_disparity.csv", "right_reference", "Reference disparity L0->R0"),
    ("left_temporal.csv", "left_deformed", "Temporal field L0->L1"),
    ("deformed_disparity.csv", "right_deformed", "Deformed stereo field L0->R1"),
)


def compute_pairwise_2d_dic(
    case_root: str | Path,
    calibration: Any | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    pair_selection: CameraPairSelectionResult | Mapping[str, Any] | None = None,
    subset_config: str | Path | Mapping[str, Any] | None = None,
    mesh_config: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    options: PairwiseDICOptions | Mapping[str, Any] | None = None,
) -> PairwiseDICRunResult:
    """Compute pair-level 2D-DIC fields for multiview surface stitching.

    Each selected camera pair is treated like the existing stereo workflow and
    produces ``reference_disparity``, ``left_temporal`` and
    ``deformed_disparity`` fields under ``result/disp/{pair}``.
    """

    from .config import load_config, mesh_generation_config, normalize_subset_config
    from .mesh import mesh as compute_mesh
    from .subset import subset as compute_subset
    from .visualization import visualization_dir_for_result

    case_root = Path(case_root)
    mv_cfg = _load_multiview_config(config, case_root)
    opts = _coerce_pairwise_dic_options(options, mv_cfg)
    if output_dir is not None:
        opts.output_dir = str(output_dir)

    subset_cfg = _resolve_solver_config(
        explicit=subset_config,
        config_value=opts.subset_config or _nested_get(mv_cfg, "pairwise_2d_dic", "subset_config"),
        default_relative=_project_root() / "config" / "subset_2d.yaml",
        case_root=case_root,
        loader=load_config,
        normalizer=normalize_subset_config,
    )
    mesh_cfg_raw = _resolve_solver_config(
        explicit=mesh_config,
        config_value=opts.mesh_config or _nested_get(mv_cfg, "pairwise_2d_dic", "mesh_config"),
        default_relative=_project_root() / "config" / "mesh_2d.yaml",
        case_root=case_root,
        loader=load_config,
        normalizer=lambda data: data or {},
    )
    generation_cfg = mesh_generation_config(mesh_cfg_raw)
    if "target_element_size" in generation_cfg:
        opts.mesh_target_element_size = float(generation_cfg["target_element_size"])

    pairs = _resolve_pair_names(case_root, calibration, pair_selection, opts)
    if opts.max_pairs is not None:
        pairs = pairs[: max(0, int(opts.max_pairs))]

    root_out = _case_path(case_root, opts.output_dir)
    visualization_out = visualization_dir_for_result(case_root, root_out)
    mask_root = _case_path(case_root, opts.roi_dir)
    pair_dirs: list[str] = []
    roi_overview_items: list[tuple[str, Path]] = []
    overplay_overview_items: list[tuple[str, Path]] = []
    mesh_types = [str(t).upper() for t in opts.mesh_types]
    for left_name, right_name in pairs:
        pair_label = f"{left_name}-{right_name}"
        mask_stem = f"mask_{left_name}_{right_name}"
        pair_root = root_out / pair_label
        pair_dirs.append(str(pair_root))
        paths = _pair_image_paths(case_root, opts, left_name, right_name)
        reference = _read_gray_image(paths["left_reference"])
        height, width = reference.shape
        left_mask = _load_optional_camera_mask(case_root, opts, left_name, height, width)
        right_reference = _read_gray_image(paths["right_reference"])
        common_left_uv, common_right_uv, match_meta = _match_pair_roi_features(reference, right_reference, opts)
        roi, roi_meta = _load_or_build_pair_roi(
            case_root,
            opts,
            left_name,
            right_name,
            left_mask,
            height,
            width,
            common_uv_pairs=(common_left_uv, common_right_uv, "pair_sift_reference_match", match_meta),
        )
        _save_pair_roi(
            mask_root,
            left_name,
            right_name,
            roi,
            left_mask,
            common_left_uv,
            common_right_uv,
            reference,
            right_reference,
            roi_meta,
        )
        roi_overview_items.append((pair_label, mask_root / "roi" / f"{mask_stem}.png"))
        overplay_overview_items.append((pair_label, mask_root / "overplay" / f"{mask_stem}_overplay.png"))

        if opts.run_subset:
            subset_dir = pair_root / "subset"
            subset_visualization_dir = visualization_out / pair_label / "subset"
            subset_solve_roi = left_mask if str(opts.subset_solve_roi_mode).lower() == "left_mask" else roi
            subset_complete = all(
                (subset_dir / filename).is_file()
                for filename in ("reference_disparity.csv", "left_temporal.csv", "deformed_disparity.csv")
            )
            if opts.overwrite or not subset_complete:
                _compute_standard_subset_fields(
                    compute_subset,
                    reference,
                    paths,
                    subset_solve_roi,
                    roi,
                    subset_cfg,
                    subset_dir,
                    subset_visualization_dir,
                    width,
                    height,
                )

        if opts.run_mesh:
            mesh_root = pair_root / "mesh"
            mesh_data = _structured_meshes_from_roi(roi, target_element_size=opts.mesh_target_element_size)
            for etype in mesh_types:
                nodes, elements = mesh_data[etype]
                mesh_dir = mesh_root / etype
                mesh_visualization_dir = visualization_out / pair_label / "mesh" / etype
                mesh_gen_dir = mesh_dir / "meshGen"
                mesh_visualization_gen_dir = mesh_visualization_dir / "meshGen"
                _write_nodes(mesh_gen_dir / f"nodes_{etype}.txt", nodes)
                _write_elements(mesh_gen_dir / f"elements_{etype}.txt", elements)
                _render_mesh(mesh_visualization_gen_dir / f"mesh_{etype}.png", roi, nodes, elements, etype)
                mesh_complete = all(
                    (mesh_dir / filename).is_file()
                    for filename in ("reference_disparity.csv", "left_temporal.csv", "deformed_disparity.csv")
                )
                if opts.overwrite or not mesh_complete:
                    _compute_standard_mesh_fields(
                        compute_mesh,
                        reference,
                        paths,
                        nodes,
                        elements,
                        etype,
                        mesh_cfg_raw,
                        mesh_dir,
                        mesh_visualization_dir,
                        width,
                        height,
                        roi=roi,
                    )

    _save_mask_overviews(mask_root, roi_overview_items, overplay_overview_items)
    return PairwiseDICRunResult(
        output_dir=str(root_out),
        pair_dirs=pair_dirs,
        pairs=pairs,
        run_subset=bool(opts.run_subset),
        run_mesh=bool(opts.run_mesh),
        mesh_types=mesh_types,
    )


def compute_pairwise_3d_dic(
    case_root: str | Path,
    calibration: Any | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    pair_selection: CameraPairSelectionResult | Mapping[str, Any] | None = None,
    field_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    solver: str | None = None,
    options: Pairwise3DOptions | Mapping[str, Any] | None = None,
) -> Pairwise3DRunResult:
    """Reconstruct each selected camera pair from precomputed 2D-DIC fields.

    This is the pair-level stereo reconstruction stage used before stitching.
    It consumes ``reference_disparity.csv``, ``left_temporal.csv`` and
    ``deformed_disparity.csv`` from each pair directory.
    """

    from .stereo import camera_from_dict, reconstruct_from_field_files
    from .visualization import visualization_dir_for_result

    case_root = Path(case_root)
    mv_cfg = _load_multiview_config(config, case_root)
    opts = _coerce_pairwise_3d_options(options, mv_cfg)
    if field_dir is not None:
        opts.field_dir = str(field_dir)
    if output_dir is not None:
        opts.output_dir = str(output_dir)
    if solver is not None:
        opts.solver = str(solver)

    calibration_data = calibration
    if calibration_data is None:
        calibration_path = _case_path(case_root, opts.calibration_dir) / "calibration_result.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration result not found: {calibration_path}")
        calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration_data, Mapping):
        raise TypeError("calibration must be a mapping or None")

    calibration_data, scale_meta = _calibration_with_scaled_cameras(case_root, calibration_data, opts.calibration_dir)
    cameras_by_label: dict[str, object] = {}
    camera_source = "scaled_cameras" if calibration_data.get("scaled_cameras") else "cameras"
    for idx, camera_data in enumerate(calibration_data.get(camera_source, []) or []):
        label = str(camera_data.get("label", f"cam_{idx}"))
        cameras_by_label[label] = camera_from_dict(camera_data)

    pair_opts = PairwiseDICOptions(calibration_dir=opts.calibration_dir)
    pairs = _resolve_pair_names(case_root, calibration_data, pair_selection, pair_opts)
    if opts.max_pairs is not None:
        pairs = pairs[: max(0, int(opts.max_pairs))]

    field_root = _case_path(case_root, opts.field_dir)
    out_root = _case_path(case_root, opts.output_dir) / str(opts.solver)
    pair_dirs: list[str] = []
    total_points = 0
    valid_points = 0
    summaries: list[dict[str, Any]] = []

    for left_name, right_name in pairs:
        if left_name not in cameras_by_label or right_name not in cameras_by_label:
            raise KeyError(f"Missing camera model for pair {left_name}-{right_name}")
        pair_label = f"{left_name}-{right_name}"
        pair_field_root = field_root / pair_label / str(opts.solver)
        # The mesh solver writes per-element 2D fields under mesh/<etype>/;
        # reconstruct each element type independently so the downstream
        # stitch stage can keep T3/Q4/Q8 results separate (like subset keeps
        # per-method outputs). subset writes its fields directly at mesh-free
        # paths, so it uses a single None element.
        if str(opts.solver) == "mesh":
            etype_dirs = sorted(d.name for d in pair_field_root.iterdir() if d.is_dir())
            if not etype_dirs:
                raise RuntimeError(f"No mesh element subdirectories under {pair_field_root}")
            etypes: list[str | None] = list(etype_dirs)
        else:
            etypes = [None]
        for etype in etypes:
            pair_field_dir = pair_field_root if etype is None else pair_field_root / etype
            pair_out_dir = out_root / pair_label if etype is None else out_root / pair_label / etype
            pair_visualization_dir = visualization_dir_for_result(case_root, pair_out_dir)
            result = reconstruct_from_field_files(
                pair_field_dir,
                cameras_by_label[left_name],
                cameras_by_label[right_name],
                out_dir=pair_out_dir,
                deformation_out_dir=pair_out_dir,
                visualization_out_dir=pair_visualization_dir,
                deformation_visualization_out_dir=pair_visualization_dir,
                write_shape_maps=bool(opts.write_shape_maps),
                write_deformation_maps=bool(opts.write_deformation_maps),
                write_surface_strain=bool(opts.write_surface_strain),
                min_correlation=float(opts.min_correlation),
                quality_metric=str(opts.quality_metric),
                max_znssd=float(opts.max_znssd),
                max_reprojection_error_px=float(opts.max_reprojection_error_px),
                world_scale=float(opts.world_scale),
                remove_rigid_body_motion=bool(opts.remove_rigid_body_motion),
            )
            pair_dirs.append(str(pair_out_dir))
            total_points += int(result.total_points)
            valid_points += int(result.valid_points)
            summaries.append(
                {
                    "pair": pair_label,
                    "element": etype,
                    "left_camera": left_name,
                    "right_camera": right_name,
                    "field_dir": str(pair_field_dir),
                    "output_dir": str(pair_out_dir),
                    "visualization_dir": str(pair_visualization_dir),
                    "total_points": int(result.total_points),
                    "valid_points": int(result.valid_points),
                }
            )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "pairwise_3d_summary.json").write_text(
        json.dumps(
            {
                "solver": str(opts.solver),
                "quality_metric": str(opts.quality_metric),
                "max_znssd": float(opts.max_znssd),
                "max_reprojection_error_px": float(opts.max_reprojection_error_px),
                "camera_source": camera_source,
                "scale": scale_meta,
                "total_points": int(total_points),
                "valid_points": int(valid_points),
                "pairs": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return Pairwise3DRunResult(
        output_dir=str(out_root),
        pair_dirs=pair_dirs,
        pairs=pairs,
        solver=str(opts.solver),
        total_points=int(total_points),
        valid_points=int(valid_points),
    )


def recover_multiview_calibration_scale(
    case_root: str | Path,
    calibration: Any | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    options: MultiviewScaleRecoveryOptions | Mapping[str, Any] | None = None,
) -> MultiviewScaleRecoveryResult:
    """Estimate metric scale for self-calibrated multiview cameras.

    The SfM calibration is only defined up to a similarity transform.  This
    stage triangulates a visible chessboard in SfM coordinates, compares its
    edge length to the physical square size, and writes ``scaled_cameras``.
    """

    from . import calibration as calibration_api
    from .stereo import camera_from_dict

    case_root = Path(case_root)
    mv_cfg = _load_multiview_config(config, case_root)
    opts = _coerce_scale_recovery_options(options, mv_cfg)
    calibration_dir = _case_path(case_root, opts.calibration_dir)
    calibration_data = calibration
    if calibration_data is None:
        calibration_path = calibration_dir / "calibration_result.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration result not found: {calibration_path}")
        calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration_data, Mapping):
        raise TypeError("calibration must be a mapping or None")

    observations = _load_scale_observations(case_root, opts)
    if not observations:
        raise RuntimeError("No multiview chessboard scale observations were found")

    backend = calibration_api._require_backend()
    cameras = [camera_from_dict(camera_data) for camera_data in calibration_data.get("cameras", []) or []]
    sparse_points = _sparse_points_from_dicts(calibration_data.get("points3d", []) or [], backend)
    scale_options = backend.MultiviewScaleOptions()
    scale_options.board_rows = int(opts.board_rows)
    scale_options.board_cols = int(opts.board_cols)
    scale_options.square_size = float(opts.square_size)
    scale_options.max_reprojection_error = float(opts.max_reprojection_error)
    scale_options.trim_fraction = float(opts.trim_fraction)
    scale_options.min_common_corners = int(opts.min_common_corners)

    try:
        raw_result = backend.estimate_multiview_chessboard_scale(cameras, sparse_points, observations, scale_options)
        scale_data = calibration_api.scale_result_to_dict(raw_result)
    except RuntimeError:
        if not bool(opts.allow_meta_camera_model_fallback):
            raise
        scale_data = _scale_data_from_metric_meta_cameras(case_root, opts)
        if scale_data is None:
            raise
    output_path = calibration_dir / str(opts.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scale_data, indent=2), encoding="utf-8")

    merged = dict(calibration_data)
    merged["sfm_to_world_scale"] = float(scale_data["sfm_to_world_scale"])
    merged["world_to_sfm_scale"] = float(scale_data["world_to_sfm_scale"])
    merged["scaled_cameras"] = scale_data.get("scaled_cameras", [])
    merged["scaled_points3d"] = scale_data.get("scaled_points3d", [])
    (calibration_dir / "calibration_result_scaled.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    return MultiviewScaleRecoveryResult(
        output_file=str(output_path),
        sfm_to_world_scale=float(scale_data["sfm_to_world_scale"]),
        world_to_sfm_scale=float(scale_data["world_to_sfm_scale"]),
        valid_edges=int(scale_data["valid_edges"]),
        triangulated_corners=int(scale_data["triangulated_corners"]),
    )


def _scale_data_from_metric_meta_cameras(
    case_root: Path,
    opts: MultiviewScaleRecoveryOptions,
) -> dict[str, Any] | None:
    chessboard_dir = _case_path(case_root, opts.chessboard_dir)
    meta_path = chessboard_dir / "chessboard_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cameras = []
    width = int(_nested_get(meta, "config", "image_width") or 0)
    height = int(_nested_get(meta, "config", "image_height") or 0)
    for camera_data in sorted(meta.get("cameras", []) or [], key=lambda item: int(item.get("camera_id", 0))):
        if not all(key in camera_data for key in ("K", "R_world_to_camera", "t_world_to_camera")):
            return None
        cameras.append(
            {
                "label": str(camera_data.get("camera_name", f"cam_{int(camera_data.get('camera_id', len(cameras)))}")),
                "K": camera_data["K"],
                "R": camera_data["R_world_to_camera"],
                "t": camera_data["t_world_to_camera"],
                "distortion": [float(v) for v in camera_data.get("distortion", [])],
                "image_size": [width, height],
                "image_width": width,
                "image_height": height,
                "rms_error": 0.0,
            }
        )
    if len(cameras) < 2:
        return None
    return {
        "source": str(meta_path),
        "source_type": "metric_chessboard_meta_camera_models",
        "sfm_to_world_scale": 1.0,
        "world_to_sfm_scale": 1.0,
        "sfm_square_size_mean": float(opts.square_size),
        "sfm_square_size_median": float(opts.square_size),
        "sfm_square_size_std": 0.0,
        "edge_cv": 0.0,
        "triangulated_corners": int(opts.board_rows) * int(opts.board_cols),
        "valid_edges": int(max(0, opts.board_rows * (opts.board_cols - 1) + (opts.board_rows - 1) * opts.board_cols)),
        "triangulated_board_points_sfm": [],
        "edge_lengths_sfm": [],
        "scaled_cameras": cameras,
        "scaled_points3d": [],
    }


def stitch_pairwise_3d_surfaces(
    case_root: str | Path,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    pair_selection: CameraPairSelectionResult | Mapping[str, Any] | None = None,
    pairwise_3d_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    solver: str | None = None,
    options: PairwiseSurfaceStitchOptions | Mapping[str, Any] | None = None,
) -> PairwiseSurfaceStitchRunResult:
    """Stitch pair surfaces using the MultiDIC overlap/zipper workflow."""
    from .surface_stitching import clean_stitched_surface, SurfaceMesh, stitch_surfaces, write_stitch_visualizations
    from .visualization import visualization_dir_for_result

    case_root = Path(case_root)
    mv_cfg = _load_multiview_config(config, case_root)
    opts = _coerce_surface_stitch_options(options, mv_cfg)
    if pairwise_3d_dir is not None:
        opts.pairwise_3d_dir = str(pairwise_3d_dir)
    if output_dir is not None:
        opts.output_dir = str(output_dir)
    if solver is not None:
        opts.solver = str(solver)

    mode = str(opts.mode).lower()
    if mode not in {"multidic", "stitch"}:
        raise ValueError("surface stitching uses mode='multidic'")

    pair_opts = PairwiseDICOptions(calibration_dir=opts.calibration_dir)
    pair_calibration = None
    if pair_selection is None:
        for calibration_name in ("calibration_result_scaled.json", "calibration_result.json"):
            calibration_path = _case_path(case_root, opts.calibration_dir) / calibration_name
            if calibration_path.exists():
                pair_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                break
    pairs = _resolve_pair_names(case_root, pair_calibration, pair_selection, pair_opts)
    if opts.max_pairs is not None:
        pairs = pairs[: max(0, int(opts.max_pairs))]

    pair_root = _case_path(case_root, opts.pairwise_3d_dir)
    if pair_root.name != str(opts.solver):
        pair_root = pair_root / str(opts.solver)
    out_root = _case_path(case_root, opts.output_dir) / str(opts.solver)
    out_root.mkdir(parents=True, exist_ok=True)

    # The mesh solver reconstructs each element type (T3/Q4/Q8) independently
    # under pairwise_3d_dir/pair/<etype>/ (see compute_pairwise_3d_dic), so the
    # stitch stage also runs once per element type and writes each result to its
    # own out_root/<etype>/ directory -- keeping mesh outputs isolated the same
    # way subset keeps per-method outputs. subset has no element dimension, so
    # it uses a single None element.
    if str(opts.solver) == "mesh":
        probe_dir = pair_root / f"{pairs[0][0]}-{pairs[0][1]}"
        etypes: list[str | None] = sorted(d.name for d in probe_dir.iterdir() if d.is_dir())
        if not etypes:
            raise RuntimeError(f"No mesh element subdirectories under {probe_dir}")
    else:
        etypes = [None]

    total_points = 0
    total_faces = 0
    element_out_roots: list[str] = []
    element_summaries: list[dict[str, Any]] = []

    for etype in etypes:
        elem_out = out_root if etype is None else out_root / etype
        elem_out.mkdir(parents=True, exist_ok=True)

        meshes: list[SurfaceMesh] = []
        source_rows: list[list[dict[str, Any]]] = []
        pair_summaries: list[dict[str, Any]] = []

        for pair_index, (left_name, right_name) in enumerate(pairs, start=1):
            pair_label = f"{left_name}-{right_name}"
            points_path = pair_root / pair_label
            if etype is not None:
                points_path = points_path / etype
            points_path = points_path / "stereo_3d_points.csv"
            pair_points = _read_pair_surface_points(
                points_path,
                pair_label=pair_label,
                pair_index=pair_index,
                max_reprojection_error_px=float(opts.max_reprojection_error_px),
                max_quality=float(opts.max_quality),
            )
            pair_valid_count = len(pair_points)
            if pair_valid_count < int(opts.min_valid_points):
                pair_summaries.append(
                    {
                        "element": etype,
                        "pair": pair_label,
                        "points": pair_valid_count,
                        "faces": 0,
                        "skipped": True,
                        "reason": "not enough valid points",
                    }
                )
                continue

            uv = np.asarray([[row["x_l0"], row["y_l0"]] for row in pair_points], dtype=np.float64)
            faces = _triangulate_pair_surface_faces(uv, edge_scale=float(opts.triangle_edge_scale))
            source_rows.append(pair_points)
            meshes.append(
                SurfaceMesh(
                    reference=np.asarray([[row["X0"], row["Y0"], row["Z0"]] for row in pair_points], dtype=np.float64),
                    deformed=np.asarray([[row["X1"], row["Y1"], row["Z1"]] for row in pair_points], dtype=np.float64),
                    faces=faces,
                    quality=np.asarray([row["quality"] for row in pair_points], dtype=np.float64),
                    pair_index=pair_index,
                    pair_name=pair_label,
                )
            )
            pair_summaries.append(
                {
                    "element": etype,
                    "pair": pair_label,
                    "points": pair_valid_count,
                    "faces": int(len(faces)),
                    "skipped": False,
                }
            )

        if not meshes:
            continue
        stitched_raw = stitch_surfaces(meshes, min_gap_factor=float(opts.min_gap_factor))
        cleaned = clean_stitched_surface(
            stitched_raw,
            neighbor_count=int(opts.outlier_neighbor_count),
            distance_sigma=float(opts.outlier_distance_sigma),
            displacement_sigma=float(opts.outlier_displacement_sigma),
            face_edge_scale=float(opts.outlier_face_edge_scale),
        )
        stitched = cleaned.result
        # Optional displacement-field smoothing (post-stitch). The 12 pairs are
        # reconstructed independently, so adjacent nodes from different pairs
        # carry each pair's systematic calibration/triangulation offset. A KNN
        # mean over the stitched surface averages that cross-pair inconsistency
        # out of the displacement field without touching the reference geometry.
        smooth_k = int(opts.smooth_displacement_knn or 0)
        smooth_U: np.ndarray | None = None
        if smooth_k > 0:
            from scipy.spatial import cKDTree

            # Only valid (non-outlier) points participate in the KNN mean;
            # outlier points (cleaned.valid_points == False) keep their raw
            # displacement and are not averaged into their neighbours.
            valid_idx = np.flatnonzero(cleaned.valid_points)
            sub_ref = stitched.reference[valid_idx]
            sub_U = (stitched.deformed - stitched.reference)[valid_idx]
            tree = cKDTree(sub_ref)
            _, nidx = tree.query(sub_ref, k=min(smooth_k, len(sub_ref)))
            sub_smooth = sub_U[nidx].mean(axis=1)
            smooth_U = (stitched.deformed - stitched.reference).copy()
            smooth_U[valid_idx] = sub_smooth
            stitched.deformed = stitched.reference + smooth_U
        point_rows: list[dict[str, Any]] = []
        source_index = 0
        for mesh_rows in source_rows:
            for row in mesh_rows:
                output = dict(row)
                output["global_id"] = source_index + 1
                if not cleaned.valid_points[source_index]:
                    for key in ("X0", "Y0", "Z0", "X1", "Y1", "Z1", "Ux", "Uy", "Uz", "Umag"):
                        output[key] = float("nan")
                    output["valid"] = 0
                else:
                    output["valid"] = 1
                    if smooth_U is not None:
                        u = smooth_U[source_index]
                        output["X1"] = output["X0"] + u[0]
                        output["Y1"] = output["Y0"] + u[1]
                        output["Z1"] = output["Z0"] + u[2]
                        output["Ux"] = float(u[0])
                        output["Uy"] = float(u[1])
                        output["Uz"] = float(u[2])
                        output["Umag"] = float(np.linalg.norm(u))
                point_rows.append(output)
                source_index += 1
        face_rows: list[dict[str, Any]] = []
        for face_index, face in enumerate(stitched.faces, start=1):
            pair_index = int(stitched.face_pair_indices[face_index - 1])
            face_rows.append(
                {
                    "face_id": face_index,
                    "pair": "zipper" if pair_index < 0 else pair_summaries[pair_index - 1]["pair"],
                    "pair_index": pair_index,
                    "pair_face_id": face_index,
                    "n1": int(face[0]) + 1,
                    "n2": int(face[1]) + 1,
                    "n3": int(face[2]) + 1,
                    "quality": float(stitched.face_quality[face_index - 1]) if np.isfinite(stitched.face_quality[face_index - 1]) else float("nan"),
                }
            )
        _write_stitched_points(elem_out / "stitched_points.csv", point_rows)
        _write_stitched_faces(elem_out / "stitched_faces.csv", face_rows)
        strain_cfg = dict(mv_cfg.get("strain", {}) or {})
        if bool(strain_cfg.get("enabled", True)):
            from .postprocess import save_surface_strain_csv
            save_surface_strain_csv(
                elem_out / "stitched_3d_strain_faces.csv", stitched.faces,
                stitched.reference, stitched.deformed, cleaned.valid_faces,
                min_face_area=float(strain_cfg.get("min_face_area", 0.0)),
            )
        visualization_root = visualization_dir_for_result(case_root, elem_out)
        write_stitch_visualizations(stitched, visualization_root)
        (elem_out / "stitched_summary.json").write_text(
            json.dumps(
                {
                    "solver": str(opts.solver),
                    "element": etype,
                    "mode": "multidic",
                    "point_count": int(len(stitched.reference)),
                    "face_count": int(len(stitched.faces)),
                    "raw_point_count": int(len(stitched_raw.reference)),
                    "raw_face_count": int(len(stitched_raw.faces)),
                    "cleaned_removed_points": int(cleaned.removed_points),
                    "cleaned_removed_faces": int(cleaned.removed_faces),
                    "triangle_edge_scale": float(opts.triangle_edge_scale),
                    "min_gap_factor": float(opts.min_gap_factor),
                    "overlap_removed_faces": int(stitched.overlap_removed_faces),
                    "zipper_faces": int(stitched.zipper_faces),
                    "hole_faces": int(stitched.hole_faces),
                    "visualization_dir": str(visualization_root),
                    "smooth_displacement_knn": int(smooth_k),
                    "pairs": pair_summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        total_points += int(len(stitched.reference))
        total_faces += int(len(stitched.faces))
        element_out_roots.append(str(elem_out))
        element_summaries.append(
            {
                "element": etype,
                "point_count": int(len(stitched.reference)),
                "face_count": int(len(stitched.faces)),
                "output_dir": str(elem_out),
            }
        )

    if not element_out_roots:
        raise RuntimeError("No valid pair surfaces were available for stitching")

    (out_root / "stitch_index.json").write_text(
        json.dumps(
            {
                "solver": str(opts.solver),
                "mode": "multidic",
                "point_count": total_points,
                "face_count": total_faces,
                "elements": element_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PairwiseSurfaceStitchRunResult(
        output_dir=str(out_root),
        pairs=pairs,
        solver=str(opts.solver),
        mode="multidic",
        point_count=total_points,
        face_count=total_faces,
    )


def _read_pair_surface_points(
    path: Path,
    *,
    pair_label: str,
    pair_index: int,
    max_reprojection_error_px: float,
    max_quality: float,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    required = ("x_l0", "y_l0", "X0", "Y0", "Z0", "X1", "Y1", "Z1", "Ux", "Uy", "Uz")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for source_row, raw in enumerate(reader, start=1):
            if str(raw.get("valid", "0")).strip() not in {"1", "true", "True"}:
                continue
            values = {key: float(raw[key]) for key in required}
            err_ref = float(raw.get("reprojection_error_ref", 0.0))
            err_def = float(raw.get("reprojection_error_def", 0.0))
            quality = float(raw.get("combined_correlation", 0.0))
            finite_values = list(values.values()) + [err_ref, err_def, quality]
            if not np.all(np.isfinite(finite_values)):
                continue
            if err_ref > max_reprojection_error_px or err_def > max_reprojection_error_px:
                continue
            if quality > max_quality:
                continue
            rows.append(
                {
                    "pair": pair_label,
                    "pair_index": int(pair_index),
                    "pair_point_id": int(float(raw.get("id", source_row))),
                    "source_row": int(source_row),
                    "x_l0": values["x_l0"],
                    "y_l0": values["y_l0"],
                    "X0": values["X0"],
                    "Y0": values["Y0"],
                    "Z0": values["Z0"],
                    "X1": values["X1"],
                    "Y1": values["Y1"],
                    "Z1": values["Z1"],
                    "Ux": values["Ux"],
                    "Uy": values["Uy"],
                    "Uz": values["Uz"],
                    "Umag": float(raw.get("Umag", np.linalg.norm([values["Ux"], values["Uy"], values["Uz"]]))),
                    "reprojection_error_ref": err_ref,
                    "reprojection_error_def": err_def,
                    "quality": quality,
                }
            )
    return rows


def _triangulate_pair_surface_faces(uv: np.ndarray, *, edge_scale: float) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    if len(uv) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    try:
        from scipy.spatial import Delaunay, cKDTree
    except Exception:
        return np.zeros((0, 3), dtype=np.int64)
    try:
        tri = Delaunay(uv)
    except Exception:
        return np.zeros((0, 3), dtype=np.int64)
    simplices = np.asarray(tri.simplices, dtype=np.int64)
    if simplices.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    dists, _ = cKDTree(uv).query(uv, k=2)
    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        return simplices
    max_edge = float(edge_scale) * float(np.median(nearest))
    vertices = uv[simplices]
    e01 = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
    e12 = np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1)
    e20 = np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
    area = 0.5 * np.abs(
        (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1]) * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    keep = (e01 <= max_edge) & (e12 <= max_edge) & (e20 <= max_edge) & (area > 1.0e-8)
    return simplices[keep]


def _write_stitched_points(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "global_id",
        "valid",
        "pair",
        "pair_index",
        "pair_point_id",
        "source_row",
        "x_l0",
        "y_l0",
        "X0",
        "Y0",
        "Z0",
        "X1",
        "Y1",
        "Z1",
        "Ux",
        "Uy",
        "Uz",
        "Umag",
        "reprojection_error_ref",
        "reprojection_error_def",
        "quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_stitched_faces(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["face_id", "pair", "pair_index", "pair_face_id", "n1", "n2", "n3", "quality"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_multiview_config(config: str | Path | Mapping[str, Any] | None, case_root: Path) -> dict[str, Any]:
    from .config import load_config

    if config is None:
        for default in (
            _project_root() / "config" / "multiview_3d.yaml",
            case_root / "config" / "multiview_3d.yaml",
            case_root / "config" / "multiview_dic.yaml",
        ):
            if default.exists():
                return load_config(default)
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    return load_config(_resolve_existing_path(case_root, config))


def _nested_get(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _coerce_pairwise_dic_options(
    options: PairwiseDICOptions | Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> PairwiseDICOptions:
    payload: dict[str, Any] = {}
    output = dict(config.get("output", {}) or {})
    pairwise = dict(config.get("pairwise_2d_dic", {}) or {})
    maskgen = dict(config.get("maskGen", config.get("maskgen", {})) or {})
    for key in ("mask_dir", "roi_dir", "calibration_dir", "disp_dir", "output_dir"):
        if key in output:
            payload["output_dir" if key == "disp_dir" else key] = output[key]
    for key in PairwiseDICOptions.__dataclass_fields__:
        if key in pairwise:
            payload[key] = pairwise[key]
    maskgen_key_map = {
        "feature_method": "maskgen_feature_method",
        "max_features": "maskgen_max_features",
        "match_ratio": "maskgen_match_ratio",
        "ratio_threshold": "maskgen_match_ratio",
        "mutual_check": "maskgen_mutual_check",
        "ransac_reprojection_threshold": "maskgen_ransac_reprojection_threshold",
        "min_matches": "maskgen_min_matches",
        "pair_roi_source": "pair_roi_source",
        "roi_source": "pair_roi_source",
        "pair_roi_support": "pair_roi_support",
        "roi_support": "pair_roi_support",
        "pair_roi_robust_outlier": "pair_roi_robust_outlier",
        "robust_outlier": "pair_roi_robust_outlier",
        "pair_roi_outlier_mad_scale": "pair_roi_outlier_mad_scale",
        "outlier_mad_scale": "pair_roi_outlier_mad_scale",
        "pair_roi_alpha_radius_scale": "pair_roi_alpha_radius_scale",
        "alpha_radius_scale": "pair_roi_alpha_radius_scale",
        "pair_roi_min_common_tracks": "pair_roi_min_common_tracks",
        "min_roi_matches": "pair_roi_min_common_tracks",
    }
    for key, value in maskgen.items():
        field = maskgen_key_map.get(str(key), str(key))
        if field in PairwiseDICOptions.__dataclass_fields__:
            payload[field] = value
    if options is not None:
        if isinstance(options, PairwiseDICOptions):
            payload.update(asdict(options))
        else:
            payload.update({k: v for k, v in dict(options).items() if k in PairwiseDICOptions.__dataclass_fields__})
    if "mesh_types" in payload and isinstance(payload["mesh_types"], list):
        payload["mesh_types"] = tuple(payload["mesh_types"])
    return PairwiseDICOptions(**payload)


def _coerce_pairwise_3d_options(
    options: Pairwise3DOptions | Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> Pairwise3DOptions:
    payload: dict[str, Any] = {}
    output = dict(config.get("output", {}) or {})
    pairwise_3d = dict(config.get("pairwise_3d_dic", {}) or {})
    reconstruction = dict(config.get("reconstruction", {}) or {})
    for key in ("calibration_dir", "disp_dir", "reconstruct_dir"):
        if key in output:
            if key == "disp_dir":
                payload["field_dir"] = output[key]
            elif key == "reconstruct_dir":
                payload["output_dir"] = output[key]
            else:
                payload[key] = output[key]
    for key in Pairwise3DOptions.__dataclass_fields__:
        if key in reconstruction:
            payload[key] = reconstruction[key]
    for key in Pairwise3DOptions.__dataclass_fields__:
        if key in pairwise_3d:
            payload[key] = pairwise_3d[key]
    if options is not None:
        if isinstance(options, Pairwise3DOptions):
            payload.update(asdict(options))
        else:
            payload.update({k: v for k, v in dict(options).items() if k in Pairwise3DOptions.__dataclass_fields__})
    return Pairwise3DOptions(**payload)


def _coerce_scale_recovery_options(
    options: MultiviewScaleRecoveryOptions | Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> MultiviewScaleRecoveryOptions:
    payload: dict[str, Any] = {}
    output = dict(config.get("output", {}) or {})
    scale = dict(config.get("scale", config.get("multiview_scale", {})) or {})
    if "calibration_dir" in output:
        payload["calibration_dir"] = output["calibration_dir"]
    for key in MultiviewScaleRecoveryOptions.__dataclass_fields__:
        if key in scale:
            payload[key] = scale[key]
    if options is not None:
        if isinstance(options, MultiviewScaleRecoveryOptions):
            payload.update(asdict(options))
        else:
            payload.update({k: v for k, v in dict(options).items() if k in MultiviewScaleRecoveryOptions.__dataclass_fields__})
    return MultiviewScaleRecoveryOptions(**payload)


def _calibration_with_scaled_cameras(
    case_root: Path,
    calibration_data: Mapping[str, Any],
    calibration_dir: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    data = dict(calibration_data)
    if data.get("scaled_cameras"):
        return data, {
            "source": "calibration_data.scaled_cameras",
            "sfm_to_world_scale": data.get("sfm_to_world_scale"),
            "world_to_sfm_scale": data.get("world_to_sfm_scale"),
        }
    cal_dir = _case_path(case_root, calibration_dir)
    for path in (cal_dir / "calibration_result_scaled.json", cal_dir / "calibration_scale.json"):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scaled_cameras"):
            data["scaled_cameras"] = payload["scaled_cameras"]
            if payload.get("scaled_points3d"):
                data["scaled_points3d"] = payload["scaled_points3d"]
            data["sfm_to_world_scale"] = payload.get("sfm_to_world_scale", data.get("sfm_to_world_scale"))
            data["world_to_sfm_scale"] = payload.get("world_to_sfm_scale", data.get("world_to_sfm_scale"))
            return data, {
                "source": str(path),
                "sfm_to_world_scale": data.get("sfm_to_world_scale"),
                "world_to_sfm_scale": data.get("world_to_sfm_scale"),
            }
    return data, None


def _sparse_points_from_dicts(points: Sequence[Mapping[str, Any]], backend: Any) -> list[Any]:
    out: list[Any] = []
    for point_data in points:
        point = backend.SparsePoint3D()
        point.point = np.asarray(point_data.get("xyz", point_data.get("point", [0.0, 0.0, 0.0])), dtype=np.float64).reshape(3)
        point.reprojection_error = float(point_data.get("reprojection_error", 0.0))
        observations: list[Any] = []
        for obs_data in point_data.get("observations", []) or []:
            obs = backend.FeatureTrackObservation()
            obs.image_index = int(obs_data.get("camera_index", obs_data.get("image_index", -1)))
            obs.point = np.asarray(obs_data.get("uv", obs_data.get("xy", obs_data.get("point", [0.0, 0.0]))), dtype=np.float64).reshape(2)
            observations.append(obs)
        point.observations = observations
        out.append(point)
    return out


def _load_scale_observations(case_root: Path, opts: MultiviewScaleRecoveryOptions) -> list[Any]:
    from . import calibration as calibration_api

    backend = calibration_api._require_backend()
    board_rows = int(opts.board_rows)
    board_cols = int(opts.board_cols)
    expected = board_rows * board_cols
    chessboard_dir = _case_path(case_root, opts.chessboard_dir)
    meta_path = chessboard_dir / "chessboard_meta.json"
    if bool(opts.use_meta_observations) and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        board = dict(meta.get("board", {}) or {})
        if "inner_corners_rows" in board:
            board_rows = int(board["inner_corners_rows"])
        if "inner_corners_cols" in board:
            board_cols = int(board["inner_corners_cols"])
        if "square_size_mm" in board:
            opts.square_size = float(board["square_size_mm"])
        expected = board_rows * board_cols
        opts.board_rows = board_rows
        opts.board_cols = board_cols
        observations: list[Any] = []
        for camera_data in meta.get("cameras", []) or []:
            image_points = np.asarray(camera_data.get("inner_corners_uv", []), dtype=np.float64).reshape((-1, 2))
            if len(image_points) != expected:
                continue
            obs = backend.MultiviewScaleObservation()
            obs.camera_index = int(camera_data.get("camera_id", _camera_name_to_index(str(camera_data.get("camera_name", "")))))
            obs.image_points = [np.asarray(xy, dtype=np.float64).reshape(2) for xy in image_points]
            observations.append(obs)
        if observations:
            return observations

    board_cfg = {"board": {"type": "chessboard", "rows": board_rows, "cols": board_cols, "spacing": float(opts.square_size)}}
    board = calibration_api.make_board(board_cfg)
    detection_options = calibration_api.make_detection_options({})
    observations = []
    for camera_dir in sorted((p for p in chessboard_dir.iterdir() if p.is_dir()), key=_natural_key):
        image_path = camera_dir / "001.bmp"
        if not image_path.exists():
            continue
        detection = calibration_api.detect_calibration_board(image_path, board=board, options=detection_options, return_raw=True)
        if not bool(detection.found) or len(detection.image_points) != expected:
            continue
        obs = backend.MultiviewScaleObservation()
        obs.camera_index = _camera_name_to_index(camera_dir.name)
        obs.image_points = list(detection.image_points)
        observations.append(obs)
    return observations


def _coerce_surface_stitch_options(
    options: PairwiseSurfaceStitchOptions | Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> PairwiseSurfaceStitchOptions:
    payload: dict[str, Any] = {}
    output = dict(config.get("output", {}) or {})
    stitch = dict(config.get("surface_stitch", config.get("pairwise_surface_stitch", {})) or {})
    for key in ("calibration_dir", "reconstruct_dir", "stitched_dir"):
        if key in output:
            if key == "reconstruct_dir":
                payload["pairwise_3d_dir"] = output[key]
            elif key == "stitched_dir":
                payload["output_dir"] = output[key]
            else:
                payload[key] = output[key]
    for key in PairwiseSurfaceStitchOptions.__dataclass_fields__:
        if key in stitch:
            payload[key] = stitch[key]
    if options is not None:
        if isinstance(options, PairwiseSurfaceStitchOptions):
            payload.update(asdict(options))
        else:
            payload.update({k: v for k, v in dict(options).items() if k in PairwiseSurfaceStitchOptions.__dataclass_fields__})
    return PairwiseSurfaceStitchOptions(**payload)


def _resolve_solver_config(
    *,
    explicit: str | Path | Mapping[str, Any] | None,
    config_value: Any,
    default_relative: Path,
    case_root: Path,
    loader,
    normalizer,
) -> dict[str, Any]:
    source = explicit if explicit is not None else config_value
    if source is None:
        data = loader(default_relative) if default_relative.exists() else {}
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        data = loader(_resolve_existing_path(case_root, source))
    return normalizer(data)


def _case_path(case_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else case_root / path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(case_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    case_path = case_root / path
    if case_path.exists():
        return case_path
    project_path = _project_root() / path
    return project_path if project_path.exists() else case_path


def _resolve_pair_names(
    case_root: Path,
    calibration: Any | None,
    pair_selection: CameraPairSelectionResult | Mapping[str, Any] | None,
    opts: PairwiseDICOptions,
) -> list[tuple[str, str]]:
    if isinstance(pair_selection, CameraPairSelectionResult):
        return [(str(a), str(b)) for a, b in pair_selection.pair_names]
    if isinstance(pair_selection, Mapping):
        return [(str(a), str(b)) for a, b in pair_selection.get("pair_names", [])]

    report_path = _case_path(case_root, opts.calibration_dir) / "pair_selection_report.json"
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return [(str(a), str(b)) for a, b in data.get("pair_names", [])]

    if calibration is None:
        raise ValueError("pair_selection_report.json not found and calibration was not provided.")
    return [(str(a), str(b)) for a, b in select_camera_pairs(calibration).pair_names]


def _pair_image_paths(case_root: Path, opts: PairwiseDICOptions, left_name: str, right_name: str) -> dict[str, Path]:
    image_root = _case_path(case_root, opts.image_dir)
    return {
        "left_reference": image_root / left_name / opts.reference_frame,
        "right_reference": image_root / right_name / opts.reference_frame,
        "left_deformed": image_root / left_name / opts.deformed_frame,
        "right_deformed": image_root / right_name / opts.deformed_frame,
    }


def _read_gray_image(path: str | Path) -> np.ndarray:
    image = _load_image_array(path).astype(np.float32, copy=False)
    max_value = float(np.max(image)) if image.size else 0.0
    return image / max_value if max_value > 0.0 else image


def _load_pair_mask(case_root: Path, opts: PairwiseDICOptions, camera_name: str) -> np.ndarray:
    mask_root = _case_path(case_root, opts.mask_dir) / "mask"
    candidates = [
        mask_root / f"{camera_name}_mask.npy",
        mask_root / f"{camera_name}_mask.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            return (np.load(path) > 0).astype(np.uint8)
        return (_load_image_array(path) > 0).astype(np.uint8)
    raise FileNotFoundError(f"No generated mask found for {camera_name} under {mask_root}.")


def _load_optional_camera_mask(
    case_root: Path,
    opts: PairwiseDICOptions,
    camera_name: str,
    height: int,
    width: int,
) -> np.ndarray:
    try:
        return _load_pair_mask(case_root, opts, camera_name)
    except FileNotFoundError:
        return np.ones((int(height), int(width)), dtype=np.uint8)


def _build_pair_roi(
    case_root: Path,
    opts: PairwiseDICOptions,
    left_name: str,
    right_name: str,
    left_mask: np.ndarray,
    height: int,
    width: int,
    *,
    common_uv_pairs: tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, str] | tuple[np.ndarray, np.ndarray, str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(opts.pair_roi_mode or "sparse_overlap").lower()
    left_mask = np.asarray(left_mask, dtype=np.uint8).astype(bool)
    meta: dict[str, Any] = {
        "mode": mode,
        "master_camera": left_name,
        "paired_camera": right_name,
        "left_mask_pixels": int(left_mask.sum()),
        "source": str(opts.pair_roi_source),
        "used_fallback": False,
    }
    if mode in {"left_mask", "camera_mask", "single_camera"}:
        roi = left_mask.copy()
        meta["reason"] = "pair_roi_mode_left_mask"
        meta["pair_roi_pixels"] = int(roi.sum())
        meta["pair_mask_pixels"] = int(roi.sum())
        return _postprocess_pair_roi(roi, opts), meta

    source = ""
    if common_uv_pairs is None:
        common_left_uv, common_right_uv, source = _load_common_track_uv_pairs_with_source(case_root, opts, left_name, right_name)
    else:
        common_left_uv, common_right_uv = common_uv_pairs[:2]
        if len(common_uv_pairs) >= 3:
            source = str(common_uv_pairs[2])
        if len(common_uv_pairs) >= 4 and isinstance(common_uv_pairs[3], Mapping):
            meta["match_generation"] = dict(common_uv_pairs[3])
    if source:
        meta["source_file"] = source
    min_roi_matches = max(int(opts.pair_roi_min_common_tracks), int(opts.maskgen_min_matches))
    meta["common_tracks"] = int(len(common_left_uv))
    meta["min_roi_matches"] = int(min_roi_matches)
    if len(common_left_uv) < min_roi_matches:
        roi = left_mask.copy()
        meta["used_fallback"] = True
        meta["reason"] = "too_few_pair_feature_matches"
        meta["pair_roi_pixels"] = int(roi.sum())
        meta["pair_mask_pixels"] = int(roi.sum())
        return _postprocess_pair_roi(roi, opts), meta

    common_uv = _filter_common_tracks_for_pair_roi(common_left_uv, common_right_uv, opts)
    meta["common_tracks_after_disparity_filter"] = int(len(common_uv))
    if len(common_uv) < min_roi_matches:
        common_uv = common_left_uv

    common_uv = _clip_uv_to_frame(common_uv, height, width)
    meta["common_tracks_in_frame"] = int(len(common_uv))
    if bool(opts.pair_roi_robust_outlier):
        common_uv = _remove_pair_roi_robust_outliers(common_uv, float(opts.pair_roi_outlier_mad_scale))
    meta["common_tracks_clean"] = int(len(common_uv))

    support, support_meta = _build_pair_track_support_mask(common_uv, height, width, opts)
    roi = np.asarray(support, dtype=bool) & left_mask
    if int(roi.sum()) == 0:
        roi = left_mask.copy()
        meta["used_fallback"] = True
        meta["reason"] = "empty_sparse_overlap_roi"
    else:
        meta["reason"] = "sparse_overlap"
    roi = _postprocess_pair_roi(roi, opts)
    meta.update(
        {
            "pair_roi_support": str(opts.pair_roi_support),
            "overlap_supported_pixels": int(np.asarray(support, dtype=bool).sum()),
            "pair_roi_pixels": int(roi.sum()),
            "pair_mask_pixels": int(roi.sum()),
            **support_meta,
        }
    )
    return roi, meta


def _load_or_build_pair_roi(
    case_root: Path,
    opts: PairwiseDICOptions,
    left_name: str,
    right_name: str,
    left_mask: np.ndarray,
    height: int,
    width: int,
    *,
    common_uv_pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    roi_root = _case_path(case_root, opts.roi_dir)
    path = roi_root / f"mask_{left_name}_{right_name}.npy"
    meta_path = roi_root / f"mask_{left_name}_{right_name}_meta.json"
    if path.exists() and not bool(opts.overwrite):
        roi = np.asarray(np.load(path), dtype=bool)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return roi, meta
    return _build_pair_roi(
        case_root,
        opts,
        left_name,
        right_name,
        left_mask,
        height,
        width,
        common_uv_pairs=common_uv_pairs,
    )


def _postprocess_pair_roi(roi: np.ndarray, opts: PairwiseDICOptions) -> np.ndarray:
    roi = np.asarray(roi, dtype=np.uint8)
    erode = int(opts.pair_roi_erode_pixels)
    if erode <= 0 or not np.any(roi):
        return roi.astype(bool)
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        roi = cv2.erode(roi, kernel, iterations=1)
    except Exception:
        pass
    return roi.astype(bool)


def _match_pair_roi_features(
    master_reference: np.ndarray,
    paired_reference: np.ndarray,
    opts: PairwiseDICOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = str(opts.maskgen_feature_method or "sift").lower()
    if method != "sift":
        raise ValueError(f"maskGen feature_method only supports 'sift' for now, got {method!r}.")
    try:
        import cv2
    except Exception as exc:
        raise ImportError("OpenCV is required for maskGen pair SIFT matching.") from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("The current OpenCV build does not provide cv2.SIFT_create().")

    left_u8 = _image_float_to_u8(master_reference)
    right_u8 = _image_float_to_u8(paired_reference)
    sift = cv2.SIFT_create(nfeatures=max(1, int(opts.maskgen_max_features)))
    kp_left, desc_left = sift.detectAndCompute(left_u8, None)
    kp_right, desc_right = sift.detectAndCompute(right_u8, None)
    meta: dict[str, Any] = {
        "feature_method": "sift",
        "max_features": int(opts.maskgen_max_features),
        "match_ratio": float(opts.maskgen_match_ratio),
        "mutual_check": bool(opts.maskgen_mutual_check),
        "ransac_reprojection_threshold": float(opts.maskgen_ransac_reprojection_threshold),
        "min_matches": int(opts.maskgen_min_matches),
        "keypoints_master": int(len(kp_left)),
        "keypoints_paired": int(len(kp_right)),
        "ratio_matches": 0,
        "mutual_matches": 0,
        "ransac_matches": 0,
    }
    empty = np.zeros((0, 2), dtype=np.float64)
    if desc_left is None or desc_right is None or len(kp_left) == 0 or len(kp_right) == 0:
        return empty, empty, meta

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    left_to_right = matcher.knnMatch(desc_left, desc_right, k=2)
    ratio = float(opts.maskgen_match_ratio)
    ratio_matches = [pair[0] for pair in left_to_right if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance]
    meta["ratio_matches"] = int(len(ratio_matches))

    matches = ratio_matches
    if bool(opts.maskgen_mutual_check) and ratio_matches:
        right_to_left = matcher.knnMatch(desc_right, desc_left, k=2)
        reverse_best: dict[int, int] = {}
        for pair in right_to_left:
            if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
                reverse_best[int(pair[0].queryIdx)] = int(pair[0].trainIdx)
        matches = [m for m in ratio_matches if reverse_best.get(int(m.trainIdx)) == int(m.queryIdx)]
    meta["mutual_matches"] = int(len(matches))

    left_uv = np.asarray([kp_left[m.queryIdx].pt for m in matches], dtype=np.float64).reshape((-1, 2))
    right_uv = np.asarray([kp_right[m.trainIdx].pt for m in matches], dtype=np.float64).reshape((-1, 2))
    if len(left_uv) >= 8 and float(opts.maskgen_ransac_reprojection_threshold) > 0.0:
        _, inlier_mask = cv2.findFundamentalMat(
            left_uv.astype(np.float32),
            right_uv.astype(np.float32),
            cv2.FM_RANSAC,
            float(opts.maskgen_ransac_reprojection_threshold),
            0.99,
        )
        if inlier_mask is not None:
            keep = inlier_mask.reshape(-1).astype(bool)
            if int(keep.sum()) >= 3:
                left_uv = left_uv[keep]
                right_uv = right_uv[keep]
    meta["ransac_matches"] = int(len(left_uv))
    return left_uv, right_uv, meta


def _image_float_to_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr.astype(np.uint8)
    max_value = float(np.max(arr))
    min_value = float(np.min(arr))
    if max_value <= 1.0 and min_value >= 0.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _load_common_track_uv_pairs(
    case_root: Path,
    opts: PairwiseDICOptions,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left, right, _ = _load_common_track_uv_pairs_with_source(case_root, opts, left_name, right_name)
    return left, right


def _load_common_track_uv_pairs_with_source(
    case_root: Path,
    opts: PairwiseDICOptions,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    json_path = _case_path(case_root, opts.calibration_dir) / "calibration_result.json"
    if json_path.exists():
        left, right = _load_common_track_uv_pairs_from_calibration_json(json_path, left_name, right_name)
        if len(left) > 0:
            return left, right, str(json_path)

    obs_path = _case_path(case_root, opts.calibration_dir) / "observations.npz"
    if not obs_path.exists():
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, ""
    data = np.load(obs_path)
    cam_indices = np.asarray(data["cam_indices"], dtype=np.int64)
    point_indices = np.asarray(data["point_indices"], dtype=np.int64)
    uv = np.asarray(data["uv"], dtype=np.float64).reshape((-1, 2))
    left_idx = _camera_name_to_index(left_name)
    right_idx = _camera_name_to_index(right_name)
    left = {int(point_id): xy for point_id, xy in zip(point_indices[cam_indices == left_idx], uv[cam_indices == left_idx])}
    right = {int(point_id): xy for point_id, xy in zip(point_indices[cam_indices == right_idx], uv[cam_indices == right_idx])}
    left_common: list[np.ndarray] = []
    right_common: list[np.ndarray] = []
    for point_id in sorted(set(left).intersection(right)):
        xy_left = np.asarray(left[point_id], dtype=np.float64)
        xy_right = np.asarray(right[point_id], dtype=np.float64)
        if np.all(np.isfinite(xy_left)) and np.all(np.isfinite(xy_right)):
            left_common.append(xy_left)
            right_common.append(xy_right)
    return (
        np.asarray(left_common, dtype=np.float64).reshape((-1, 2)),
        np.asarray(right_common, dtype=np.float64).reshape((-1, 2)),
        str(obs_path),
    )


def _common_track_uv_pairs_from_calibration(
    calibration: Any,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_idx = _camera_name_to_index(left_name)
    right_idx = _camera_name_to_index(right_name)
    left_common: list[np.ndarray] = []
    right_common: list[np.ndarray] = []
    for point in _extract_sparse_points(calibration):
        obs_by_cam: dict[int, np.ndarray] = {}
        for cam_idx, uv in _point_observations(point):
            if cam_idx < 0:
                continue
            arr = np.asarray(uv, dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 2 and np.all(np.isfinite(arr[:2])):
                obs_by_cam[int(cam_idx)] = arr[:2]
        if left_idx in obs_by_cam and right_idx in obs_by_cam:
            left_common.append(obs_by_cam[left_idx])
            right_common.append(obs_by_cam[right_idx])
    return (
        np.asarray(left_common, dtype=np.float64).reshape((-1, 2)),
        np.asarray(right_common, dtype=np.float64).reshape((-1, 2)),
    )


def _load_common_track_uv_pairs_from_calibration_json(
    path: Path,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _common_track_uv_pairs_from_calibration(data, left_name, right_name)


def _filter_common_tracks_for_pair_roi(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    opts: PairwiseDICOptions,
) -> np.ndarray:
    left_uv = np.asarray(left_uv, dtype=np.float64).reshape((-1, 2))
    right_uv = np.asarray(right_uv, dtype=np.float64).reshape((-1, 2))
    if len(left_uv) == 0 or len(left_uv) != len(right_uv):
        return left_uv
    q = float(opts.pair_roi_disparity_quantile)
    if q <= 0.0:
        return left_uv
    displacement_norm = np.linalg.norm(right_uv - left_uv, axis=1)
    finite = np.isfinite(displacement_norm)
    if not np.any(finite):
        return left_uv
    threshold = float(np.quantile(displacement_norm[finite], min(max(q, 0.0), 0.95)))
    keep = finite & (displacement_norm >= threshold)
    return left_uv[keep] if int(keep.sum()) >= int(opts.pair_roi_min_common_tracks) else left_uv


def _clip_uv_to_frame(uv: np.ndarray, height: int, width: int) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    if len(uv) == 0:
        return uv
    keep = (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width)) & (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
    keep &= np.isfinite(uv).all(axis=1)
    return uv[keep]


def _remove_pair_roi_robust_outliers(uv: np.ndarray, mad_scale: float) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    if len(uv) < 6:
        return uv
    center = np.median(uv, axis=0)
    dist = np.linalg.norm(uv - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    if mad <= 1.0e-12:
        return uv
    robust_sigma = 1.4826 * mad
    keep = dist <= med + float(mad_scale) * robust_sigma
    return uv[keep] if int(keep.sum()) >= 3 else uv


def _build_pair_track_support_mask(
    uv: np.ndarray,
    height: int,
    width: int,
    opts: PairwiseDICOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    meta: dict[str, Any] = {
        "n_triangles_raw": 0,
        "n_triangles_valid": 0,
        "holes_filled": 0,
    }
    if len(uv) < 3:
        return np.zeros((height, width), dtype=bool), meta

    support_mode = str(opts.pair_roi_support or "convex_hull").lower()
    if support_mode in {"alpha", "alpha_shape", "delaunay_alpha"}:
        mask, n_raw, n_valid = _build_pair_alpha_support_mask(uv, height, width, opts)
        meta["n_triangles_raw"] = int(n_raw)
        meta["n_triangles_valid"] = int(n_valid)
        if int(mask.sum()) == 0:
            mask = _rasterize_polygon(_convex_hull(uv), height, width)
            meta["alpha_fallback"] = "convex_hull"
    else:
        mask = _rasterize_polygon(_convex_hull(uv), height, width)

    filled, holes_filled = _fill_internal_holes(mask)
    meta["holes_filled"] = int(holes_filled)
    return filled, meta


def _build_pair_alpha_support_mask(
    uv: np.ndarray,
    height: int,
    width: int,
    opts: PairwiseDICOptions,
) -> tuple[np.ndarray, int, int]:
    d_nn = _median_nn_distance(uv)
    if d_nn <= 0.0:
        return np.zeros((height, width), dtype=bool), 0, 0
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return np.zeros((height, width), dtype=bool), 0, 0
    try:
        tri = Delaunay(uv)
    except Exception:
        return np.zeros((height, width), dtype=bool), 0, 0
    tri_indices = np.asarray(tri.simplices)
    vertices = uv[tri_indices]
    e0 = vertices[:, 1] - vertices[:, 0]
    e1 = vertices[:, 2] - vertices[:, 1]
    e2 = vertices[:, 0] - vertices[:, 2]
    l0 = np.linalg.norm(e0, axis=-1)
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    area = 0.5 * np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0])
    circumradius = (l0 * l1 * l2) / (4.0 * np.maximum(area, 1.0e-12))
    threshold = float(opts.pair_roi_alpha_radius_scale) * d_nn
    valid = np.isfinite(circumradius) & (circumradius <= threshold)
    valid_tris = tri_indices[valid]
    if len(valid_tris) == 0:
        return np.zeros((height, width), dtype=bool), int(len(tri_indices)), 0
    return _rasterize_triangles(uv, valid_tris, height, width), int(len(tri_indices)), int(len(valid_tris))


def _fill_internal_holes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask, 0
    background = ~mask
    labels = _connected_components(background)
    if labels.size == 0:
        return mask, 0
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    filled = mask.copy()
    holes = 0
    for label_id in range(1, int(labels.max()) + 1):
        if label_id in border_labels:
            continue
        hole = labels == label_id
        if np.any(hole):
            filled[hole] = True
            holes += 1
    return filled, holes


def _camera_name_to_index(name: str) -> int:
    text = str(name)
    if text.startswith("cam_"):
        text = text[4:]
    if text.isdigit():
        return int(text)
    raise ValueError(f"Cannot infer camera index from name: {name!r}.")


def _save_pair_roi(
    out_dir: Path,
    master_name: str,
    paired_name: str,
    roi: np.ndarray,
    master_mask: np.ndarray,
    common_uv_master: np.ndarray,
    common_uv_paired: np.ndarray,
    master_reference: np.ndarray | None,
    paired_reference: np.ndarray | None,
    meta: Mapping[str, Any],
) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = out_dir / "roi"
    overplay_dir = out_dir / "overplay"
    roi_dir.mkdir(parents=True, exist_ok=True)
    overplay_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mask_{master_name}_{paired_name}"
    roi_u8 = (np.asarray(roi, dtype=bool).astype(np.uint8) * 255)
    left_u8 = (np.asarray(master_mask, dtype=bool).astype(np.uint8) * 255)
    Image.fromarray(roi_u8, mode="L").save(roi_dir / f"{stem}.png")
    np.save(roi_dir / f"{stem}.npy", np.asarray(roi, dtype=bool))
    overlay = np.zeros((*roi_u8.shape, 3), dtype=np.uint8)
    overlay[left_u8 > 0] = np.array([60, 120, 80], dtype=np.uint8)
    overlay[roi_u8 > 0] = np.array([240, 60, 40], dtype=np.uint8)
    Image.fromarray(overlay, mode="RGB").save(overplay_dir / f"{stem}_mask_overlay.png")
    _save_pair_overplay_triplet(
        overplay_dir / f"{stem}_overplay.png",
        master_reference,
        paired_reference,
        roi,
        common_uv_master,
        common_uv_paired,
        master_name,
        paired_name,
    )
    payload = dict(meta)
    payload["pair"] = f"{master_name}-{paired_name}"
    payload["mask_png"] = f"roi/{stem}.png"
    payload["mask_npy"] = f"roi/{stem}.npy"
    payload["mask_overlay_png"] = f"overplay/{stem}_mask_overlay.png"
    payload["overplay_png"] = f"overplay/{stem}_overplay.png"
    (roi_dir / f"{stem}_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_pair_mask_outputs(mask_root: Path) -> None:
    for folder, patterns in (
        ("roi", ("mask_cam_*_cam_*.png", "mask_cam_*_cam_*.npy", "mask_cam_*_cam_*_meta.json")),
        ("overplay", ("mask_cam_*_cam_*_mask_overlay.png", "mask_cam_*_cam_*_overplay.png")),
    ):
        root = mask_root / folder
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    path.unlink()
    for name in ("overview_roi.png", "overview_overplay.png"):
        path = mask_root / name
        if path.is_file():
            path.unlink()


def _image_to_rgb_canvas(image: np.ndarray | None, fallback_shape: tuple[int, int]) -> np.ndarray:
    h, w = fallback_shape
    if image is None:
        return np.full((h, w, 3), 35, dtype=np.uint8)
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray[..., 0]
    vmax = float(np.max(gray)) if gray.size else 0.0
    if vmax <= 1.0:
        gray = gray * 255.0
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray_u8[..., None], 3, axis=2)


def _draw_points_on_canvas(canvas: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], max_points: int = 5000) -> "Image.Image":
    from PIL import Image, ImageDraw

    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    step = max(1, len(uv) // max_points)
    for x, y in uv[::step]:
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return img


def _save_pair_overplay_triplet(
    path: Path,
    master_image: np.ndarray | None,
    paired_image: np.ndarray | None,
    roi: np.ndarray,
    master_uv: np.ndarray,
    paired_uv: np.ndarray,
    master_name: str,
    paired_name: str,
) -> None:
    from PIL import Image, ImageDraw

    roi_bool = np.asarray(roi, dtype=bool)
    h, w = roi_bool.shape
    master_canvas = _image_to_rgb_canvas(master_image, (h, w))
    master_roi_canvas = master_canvas.copy()
    master_roi_canvas[roi_bool] = (0.72 * master_roi_canvas[roi_bool] + 0.28 * np.asarray([40, 180, 80])).astype(np.uint8)
    panel1 = Image.fromarray(master_roi_canvas, mode="RGB")
    panel2 = _draw_points_on_canvas(master_canvas.copy(), master_uv, (255, 40, 220))
    paired_canvas = _image_to_rgb_canvas(paired_image, (h, w))
    panel3 = _draw_points_on_canvas(paired_canvas, paired_uv, (255, 170, 0))

    triplet = Image.new("RGB", (w * 3, h), "white")
    triplet.paste(panel1, (0, 0))
    triplet.paste(panel2, (w, 0))
    triplet.paste(panel3, (w * 2, 0))
    draw = ImageDraw.Draw(triplet)
    draw.text((14, 12), f"{master_name}->{paired_name}: ROI on master", fill=(255, 255, 255))
    draw.text((w + 14, 12), f"{master_name}: matched obs {len(master_uv)}", fill=(255, 255, 255))
    draw.text((2 * w + 14, 12), f"{paired_name}: matched obs {len(paired_uv)}", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    triplet.save(path)


def _save_mask_overviews(
    mask_root: Path,
    roi_items: Sequence[tuple[str, Path]],
    overplay_items: Sequence[tuple[str, Path]],
) -> None:
    if not roi_items:
        return
    _save_roi_overview(mask_root / "overview_roi.png", roi_items)
    _save_overplay_overview(mask_root / "overview_overplay.png", overplay_items)


def _save_roi_overview(path: Path, items: Sequence[tuple[str, Path]]) -> None:
    from PIL import Image, ImageDraw

    thumbs: list[tuple[str, Image.Image]] = []
    for label, item_path in items:
        if item_path.exists():
            img = Image.open(item_path).convert("RGB")
            img.thumbnail((260, 195))
            thumbs.append((label, img.copy()))
    if not thumbs:
        return
    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    cell_w, cell_h = 280, 225
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = c * cell_w + (cell_w - img.width) // 2
        y = r * cell_h + 24
        draw.text((c * cell_w + 10, r * cell_h + 6), label, fill=(20, 20, 20))
        canvas.paste(img, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_overplay_overview(path: Path, items: Sequence[tuple[str, Path]]) -> None:
    from PIL import Image, ImageDraw

    rows: list[tuple[str, Image.Image]] = []
    for label, item_path in items:
        if item_path.exists():
            img = Image.open(item_path).convert("RGB")
            img.thumbnail((960, 240))
            rows.append((label, img.copy()))
    if not rows:
        return
    cell_w, cell_h = 980, 270
    canvas = Image.new("RGB", (cell_w, len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(rows):
        y0 = idx * cell_h
        draw.text((10, y0 + 6), label, fill=(20, 20, 20))
        canvas.paste(img, ((cell_w - img.width) // 2, y0 + 26))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _compute_standard_subset_fields(
    compute_subset,
    reference: np.ndarray,
    paths: Mapping[str, Path],
    solve_roi: np.ndarray,
    output_roi: np.ndarray,
    subset_cfg: Mapping[str, Any],
    out_dir: Path,
    visualization_dir: Path,
    width: int,
    height: int,
) -> None:
    output_roi = np.asarray(output_roi, dtype=bool)
    for field_name, image_key, title in _PAIR_FIELD_DEFS:
        deformed = _read_gray_image(paths[image_key])
        result = compute_subset(reference, deformed, config=dict(subset_cfg), roi=np.asarray(solve_roi, dtype=np.uint8))
        xy = np.column_stack([np.asarray(result["x"], dtype=np.float64), np.asarray(result["y"], dtype=np.float64)])
        uv = np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])
        corr = np.asarray(result.get("correlation", np.ones(len(xy))), dtype=np.float64)
        valid = np.asarray(result.get("valid", np.ones(len(xy), dtype=bool)), dtype=bool)
        valid &= _points_inside_roi(xy, output_roi)
        _write_displacement_field(out_dir / field_name, xy, uv, corr, valid, id_name="id")
        stem = Path(field_name).stem
        _render_field_components(visualization_dir, stem, xy[valid], uv[valid], width, height, title, dense=False)


def _compute_standard_mesh_fields(
    compute_mesh,
    reference: np.ndarray,
    paths: Mapping[str, Path],
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
    mesh_cfg: Mapping[str, Any],
    out_dir: Path,
    visualization_dir: Path,
    width: int,
    height: int,
    roi: np.ndarray | None = None,
) -> None:
    samples = _dense_mesh_samples(nodes, elements, element_type, width, height)
    for field_name, image_key, title in _PAIR_FIELD_DEFS:
        deformed = _read_gray_image(paths[image_key])
        result = compute_mesh(reference, deformed, nodes, elements, element_type=element_type, config=dict(mesh_cfg), roi=roi)
        uv = np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])
        corr = np.asarray(result.get("correlation", np.ones(len(uv))), dtype=np.float64)
        valid = np.asarray(result.get("valid", np.ones(len(uv), dtype=bool)), dtype=bool)
        _write_displacement_field(out_dir / field_name, nodes, uv, corr, valid, id_name="node_id")
        stem = Path(field_name).stem
        dense_xy, dense_uv = _write_dense_displacement_field(out_dir / f"{stem}_dense.csv", samples, uv)
        _render_scalar_field(
            visualization_dir / f"{stem}_dense_mag.png",
            dense_xy,
            np.linalg.norm(dense_uv, axis=1),
            width,
            height,
            f"{element_type} dense {title}",
            "|d|",
        )


def _points_inside_roi(xy: np.ndarray, roi: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape((-1, 2))
    roi = np.asarray(roi, dtype=bool)
    if len(xy) == 0:
        return np.zeros((0,), dtype=bool)
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < roi.shape[1]) & (y >= 0) & (y < roi.shape[0])
    out = np.zeros(len(xy), dtype=bool)
    out[inside] = roi[y[inside], x[inside]]
    return out


def _write_displacement_field(
    path: Path,
    xy: np.ndarray,
    uv: np.ndarray,
    correlation: np.ndarray,
    valid: np.ndarray,
    *,
    id_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id_name, "x", "y", "u", "v", "correlation", "valid"])
        for i, ((x, y), (u, v), corr, ok) in enumerate(zip(xy, uv, correlation, valid), start=1):
            writer.writerow([i, float(x), float(y), float(u), float(v), float(corr), int(bool(ok))])


def _write_nodes(path: Path, nodes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# node_id,x,y\n")
        for i, (x, y) in enumerate(nodes, start=1):
            f.write(f"{i},{x:.12g},{y:.12g}\n")


def _write_elements(path: Path, elements: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# element_id,node_ids...\n")
        for i, elem in enumerate(elements, start=1):
            f.write(f"{i},{','.join(str(int(v) + 1) for v in elem)}\n")


def _color_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    t = np.zeros_like(values)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _render_scalar_field(
    path: Path,
    xy: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    title: str,
    label: str,
) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.min(finite)), float(np.max(finite)))
    colors = _color_map(np.asarray(values, dtype=np.float64), vmin, vmax)
    canvas = Image.new("RGB", (int(width) + 120, int(height)), "white")
    draw = ImageDraw.Draw(canvas)
    radius = 2 if len(xy) < 20000 else 1
    for (x, y), color in zip(np.asarray(xy, dtype=np.float64), colors):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cx, cy = int(round(x)), int(round(y))
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue
        fill = tuple(int(v) for v in np.clip(color * 255.0, 0, 255))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    draw.text((14, 12), f"{title} {label} min={vmin:.4g} max={vmax:.4g} px", fill=(20, 20, 20))
    _draw_field_colorbar(draw, width + 22, 42, max(80, height - 96), vmin, vmax, f"{label} px")
    canvas.save(path)


def _draw_field_colorbar(draw, x0: int, y0: int, height: int, vmin: float, vmax: float, label: str) -> None:
    bar_w = 18
    for i in range(height):
        value = vmax - (vmax - vmin) * (i / max(1, height - 1))
        color = _color_map(np.asarray([value], dtype=np.float64), vmin, vmax)[0]
        fill = tuple(int(v) for v in np.clip(color * 255.0, 0, 255))
        draw.line((x0, y0 + i, x0 + bar_w, y0 + i), fill=fill)
    draw.rectangle((x0, y0, x0 + bar_w, y0 + height), outline=(40, 40, 40))
    draw.text((x0 + bar_w + 8, y0 - 2), f"max {vmax:.4g}", fill=(20, 20, 20))
    draw.text((x0 + bar_w + 8, y0 + height - 12), f"min {vmin:.4g}", fill=(20, 20, 20))
    draw.text((x0, y0 + height + 10), label, fill=(20, 20, 20))


def _render_field_components(
    out_dir: Path,
    stem: str,
    xy: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
    title: str,
    *,
    dense: bool,
) -> None:
    suffix = "_dense" if dense else ""
    _render_scalar_field(out_dir / f"{stem}{suffix}_u.png", xy, uv[:, 0], width, height, f"{title} U", "u")
    _render_scalar_field(out_dir / f"{stem}{suffix}_v.png", xy, uv[:, 1], width, height, f"{title} V", "v")
    _render_scalar_field(out_dir / f"{stem}{suffix}_mag.png", xy, np.linalg.norm(uv, axis=1), width, height, f"{title} Mag", "|d|")


def _structured_meshes_from_roi(roi: np.ndarray, target_element_size: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    roi = np.asarray(roi, dtype=np.uint8) > 0
    ys, xs = np.nonzero(roi)
    if xs.size == 0:
        raise ValueError("Cannot generate mesh from an empty ROI mask.")
    nx = max(3, int(np.ceil((float(xs.max()) - float(xs.min())) / max(float(target_element_size), 1.0))) + 1)
    ny = max(3, int(np.ceil((float(ys.max()) - float(ys.min())) / max(float(target_element_size), 1.0))) + 1)
    x_values = np.linspace(float(xs.min()), float(xs.max()), nx)
    y_values = np.linspace(float(ys.min()), float(ys.max()), ny)
    nodes = np.asarray([(x, y) for y in y_values for x in x_values], dtype=np.float64)
    h, w = roi.shape
    q4: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            elem = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
            corners = [(int(round(nodes[k, 0])), int(round(nodes[k, 1]))) for k in elem]
            if all(0 <= x < w and 0 <= y < h and roi[y, x] for x, y in corners):
                q4.append(elem)
    q4_arr = np.asarray(q4, dtype=np.int64).reshape((-1, 4))
    t3_arr = np.asarray(
        [[a, b, c] for a, b, c, d in q4_arr] + [[a, c, d] for a, b, c, d in q4_arr],
        dtype=np.int64,
    ).reshape((-1, 3))
    q8_nodes, q8_arr = _q8_from_q4(nodes, q4_arr)
    return {"T3": (nodes, t3_arr), "Q4": (nodes, q4_arr), "Q8": (q8_nodes, q8_arr)}


def _q8_from_q4(nodes: np.ndarray, q4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_nodes = nodes.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}
    q8: list[list[int]] = []
    for a, b, c, d in q4:
        mids = []
        for i, j in ((a, b), (b, c), (c, d), (d, a)):
            key = tuple(sorted((int(i), int(j))))
            if key not in edge_midpoints:
                edge_midpoints[key] = len(out_nodes)
                out_nodes.append(((nodes[i] + nodes[j]) * 0.5).tolist())
            mids.append(edge_midpoints[key])
        q8.append([int(a), int(b), int(c), int(d), *mids])
    return np.asarray(out_nodes, dtype=np.float64), np.asarray(q8, dtype=np.int64).reshape((-1, 8))


def _render_mesh(path: Path, roi: np.ndarray, nodes: np.ndarray, elements: np.ndarray, element_type: str) -> None:
    from PIL import Image, ImageDraw

    mask = np.asarray(roi, dtype=np.uint8) > 0
    h, w = mask.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[mask] = np.array([238, 242, 246], dtype=np.uint8)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    for elem in elements:
        ids = elem[:4] if element_type in {"Q4", "Q8"} else elem
        pts = [tuple(np.round(nodes[int(idx)]).astype(int)) for idx in ids]
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=(42, 83, 140), width=1)
    radius = 1 if len(nodes) > 10000 else 2
    for x, y in nodes:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(214, 40, 40))
    draw.text((14, 12), f"{element_type} mesh: {len(nodes)} nodes, {len(elements)} elements", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _shape_t3(xi: float, eta: float) -> np.ndarray:
    return np.asarray([1.0 - xi - eta, xi, eta], dtype=np.float64)


def _shape_q4(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )


def _shape_q8(xi: float, eta: float) -> np.ndarray:
    return np.asarray(
        [
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ],
        dtype=np.float64,
    )


def _natural_t3(point: np.ndarray, tri: np.ndarray) -> tuple[float, float] | None:
    a, b, c = tri
    try:
        xi_eta = np.linalg.solve(np.column_stack([b - a, c - a]), point - a)
    except np.linalg.LinAlgError:
        return None
    xi, eta = float(xi_eta[0]), float(xi_eta[1])
    return (xi, eta) if xi >= -1.0e-8 and eta >= -1.0e-8 and xi + eta <= 1.0 + 1.0e-8 else None


def _natural_q4(point: np.ndarray, corners: np.ndarray) -> tuple[float, float] | None:
    xi, eta = 0.0, 0.0
    for _ in range(12):
        n = _shape_q4(xi, eta)
        residual = n @ corners - point
        if np.linalg.norm(residual) < 1.0e-8:
            return xi, eta
        dxi = 0.25 * np.asarray([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        deta = 0.25 * np.asarray([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
        try:
            delta = np.linalg.solve(np.column_stack([dxi @ corners, deta @ corners]), residual)
        except np.linalg.LinAlgError:
            return None
        xi -= float(delta[0])
        eta -= float(delta[1])
    return (xi, eta) if -1.0 <= xi <= 1.0 and -1.0 <= eta <= 1.0 else None


def _dense_mesh_samples(nodes: np.ndarray, elements: np.ndarray, element_type: str, width: int, height: int) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]]:
    samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for elem_id, elem in enumerate(elements, start=1):
        xy = nodes[elem]
        corners = xy[:4] if element_type in {"Q4", "Q8"} else xy
        xmin = max(0, int(np.floor(np.min(corners[:, 0]))))
        xmax = min(width - 1, int(np.ceil(np.max(corners[:, 0]))))
        ymin = max(0, int(np.floor(np.min(corners[:, 1]))))
        ymax = min(height - 1, int(np.ceil(np.max(corners[:, 1]))))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                point = np.asarray([float(x), float(y)], dtype=np.float64)
                natural = _natural_t3(point, corners) if element_type == "T3" else _natural_q4(point, corners)
                if natural is None:
                    continue
                xi, eta = natural
                shape = _shape_t3(xi, eta) if element_type == "T3" else _shape_q4(xi, eta) if element_type == "Q4" else _shape_q8(xi, eta)
                samples[(x, y)] = (np.asarray(elem, dtype=np.int64), shape, elem_id)
    return samples


def _write_dense_displacement_field(path: Path, samples: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray, int]], nodal_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(samples.items(), key=lambda item: (item[0][1], item[0][0]))
    xy = np.asarray([key for key, _ in items], dtype=np.float64).reshape((-1, 2))
    uv = np.asarray([shape @ nodal_uv[elem] for _, (elem, shape, _) in items], dtype=np.float64).reshape((-1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "u", "v", "element_id", "correlation", "valid"])
        for i, ((x, y), (u, v), (_, _, elem_id)) in enumerate(zip(xy, uv, [item[1] for item in items]), start=1):
            writer.writerow([i, float(x), float(y), float(u), float(v), int(elem_id), 1.0, 1])
    return xy, uv


def _coerce_pair_options(options: CameraPairSelectionOptions | Mapping[str, Any] | None) -> CameraPairSelectionOptions:
    if options is None:
        return CameraPairSelectionOptions()
    if isinstance(options, CameraPairSelectionOptions):
        return options
    valid_fields = set(CameraPairSelectionOptions.__dataclass_fields__)
    return CameraPairSelectionOptions(**{k: v for k, v in dict(options).items() if k in valid_fields})


def _manual_camera_pairs(manual: Sequence[Sequence[int | str]], cam_names: Sequence[str]) -> list[tuple[int, int]]:
    name_to_id = {name: idx for idx, name in enumerate(cam_names)}
    pairs: list[tuple[int, int]] = []
    for item in manual:
        if len(item) != 2:
            raise ValueError(f"Manual camera pair must contain two entries, got {item!r}.")
        a = _camera_pair_entry_to_id(item[0], name_to_id, len(cam_names))
        b = _camera_pair_entry_to_id(item[1], name_to_id, len(cam_names))
        if a == b:
            raise ValueError(f"Manual camera pair cannot repeat one camera: {item!r}.")
        pairs.append((a, b))
    return _deduplicate_pairs(pairs)


def _camera_pair_entry_to_id(value: int | str, name_to_id: Mapping[str, int], camera_count: int) -> int:
    if isinstance(value, int):
        idx = int(value)
    else:
        text = str(value)
        if text in name_to_id:
            idx = int(name_to_id[text])
        elif text.isdigit():
            idx = int(text)
        else:
            raise ValueError(f"Unknown camera pair entry: {value!r}.")
    if idx < 0 or idx >= camera_count:
        raise ValueError(f"Camera pair index out of range: {value!r}.")
    return idx


def _adjacent_pairs(order: Sequence[int], wrap: bool) -> list[tuple[int, int]]:
    n = len(order)
    pairs: list[tuple[int, int]] = []
    for i in range(n - _PAIR_NEIGHBOR_ORDER):
        pairs.append((int(order[i]), int(order[i + _PAIR_NEIGHBOR_ORDER])))
    if wrap and n > 2:
        pairs.append((int(order[-1]), int(order[0])))
    return _deduplicate_pairs(pairs)


def _auto_spatial_pairs(
    cam_names: Sequence[str],
    centers: np.ndarray | None,
    shared: np.ndarray,
    opts: CameraPairSelectionOptions,
) -> tuple[list[tuple[int, int]], list[int], float | None, bool, list[dict[str, Any]]]:
    if centers is None or centers.shape != (len(cam_names), 3) or len(cam_names) < 2:
        order = _camera_name_order(cam_names)
        return _adjacent_pairs(order, False), order, None, False, []

    order, circularity = _spatial_camera_order(centers)
    adjacent_distances = [
        float(np.linalg.norm(centers[order[i + 1]] - centers[order[i]]))
        for i in range(len(order) - 1)
    ]
    adjacent_shared = [int(shared[order[i], order[i + 1]]) for i in range(len(order) - 1)]
    median_adjacent = float(np.median(adjacent_distances)) if adjacent_distances else 0.0
    median_shared = float(np.median(adjacent_shared)) if adjacent_shared else 0.0
    wrap_distance = float(np.linalg.norm(centers[order[-1]] - centers[order[0]])) if len(order) > 2 else np.inf
    wrap_shared = int(shared[order[-1], order[0]]) if len(order) > 2 else 0
    wrap_ratio = wrap_distance / max(median_adjacent, 1.0e-12) if np.isfinite(wrap_distance) else np.inf
    min_shared_for_wrap = max(
        int(opts.auto_min_shared_tracks),
        int(round(median_shared * float(opts.auto_wrap_min_shared_ratio))),
    )
    is_circular = (
        bool(opts.wrap)
        and circularity >= float(opts.auto_circularity_threshold)
        and wrap_ratio <= float(opts.auto_wrap_distance_ratio)
        and wrap_shared >= min_shared_for_wrap
    )

    rejected: list[dict[str, Any]] = []
    pairs: list[tuple[int, int]] = []
    max_distance = median_adjacent * float(opts.auto_max_neighbor_distance_ratio) if median_adjacent > 0 else np.inf
    min_shared = int(opts.auto_min_shared_tracks)
    candidates = _adjacent_pairs(order, is_circular)
    for a, b in candidates:
        distance = float(np.linalg.norm(centers[a] - centers[b]))
        shared_count = int(shared[a, b])
        reasons = []
        if distance > max_distance:
            reasons.append("distance_too_large")
        if shared_count < min_shared:
            reasons.append("shared_tracks_too_few")
        if reasons:
            rejected.append(
                {
                    "pair": [cam_names[a], cam_names[b]],
                    "camera_ids": [int(a), int(b)],
                    "distance": distance,
                    "shared_tracks": shared_count,
                    "reasons": reasons,
                }
            )
        else:
            pairs.append((a, b))
    return _deduplicate_pairs(pairs), order, circularity, is_circular, rejected


def _spatial_camera_order(centers: np.ndarray) -> tuple[list[int], float]:
    centroid = np.mean(centers, axis=0)
    centered = centers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis0 = vh[0]
    axis1 = vh[1] if vh.shape[0] > 1 else np.asarray([0.0, 1.0, 0.0])
    projected = np.column_stack([centered @ axis0, centered @ axis1])
    radial = np.linalg.norm(projected, axis=1)
    circularity = float(np.min(radial) / max(np.max(radial), 1.0e-12)) if radial.size else 0.0
    if circularity >= 0.45 and len(centers) >= 4:
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        return [int(idx) for idx in np.argsort(angles)], circularity
    scores = projected[:, 0]
    return [int(idx) for idx in np.argsort(scores)], circularity


def _pair_selection_result(
    mode: str,
    pairs: Sequence[tuple[int, int]],
    cam_names: Sequence[str],
    centers: np.ndarray | None,
    shared: np.ndarray,
    opts: CameraPairSelectionOptions,
    *,
    spatial_order: Sequence[int],
    circularity: float | None = None,
    is_circular: bool = False,
    rejected_pairs: list[dict[str, Any]] | None = None,
) -> CameraPairSelectionResult:
    pairs = _deduplicate_pairs(pairs)
    pair_distances: dict[str, float] = {}
    shared_counts: dict[str, int] = {}
    for a, b in pairs:
        key = f"{cam_names[a]}-{cam_names[b]}"
        shared_counts[key] = int(shared[a, b]) if shared.size else 0
        if centers is not None and centers.shape[0] > max(a, b):
            pair_distances[key] = float(np.linalg.norm(centers[a] - centers[b]))
    return CameraPairSelectionResult(
        mode=mode,
        pairs=[(int(a), int(b)) for a, b in pairs],
        pair_names=[(cam_names[a], cam_names[b]) for a, b in pairs],
        spatial_order=[int(idx) for idx in spatial_order],
        spatial_order_names=[cam_names[idx] for idx in spatial_order],
        circularity=circularity,
        is_circular=bool(is_circular),
        neighbor_order=_PAIR_NEIGHBOR_ORDER,
        reconstruction_policy="pair_level_stereo_surface",
        pair_distances=pair_distances,
        shared_track_counts=shared_counts,
        rejected_pairs=rejected_pairs or [],
        thresholds={
            "auto_circularity_threshold": float(opts.auto_circularity_threshold),
            "auto_wrap_distance_ratio": float(opts.auto_wrap_distance_ratio),
            "auto_wrap_min_shared_ratio": float(opts.auto_wrap_min_shared_ratio),
            "auto_max_neighbor_distance_ratio": float(opts.auto_max_neighbor_distance_ratio),
            "auto_min_shared_tracks": int(opts.auto_min_shared_tracks),
        },
    )


def _deduplicate_pairs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for a, b in pairs:
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(a), int(b)))
    return out


def _build_masks_with_cpp_backend(
    cameras: Sequence[Any],
    observations: Mapping[int, np.ndarray],
    shapes: Sequence[tuple[int, int]],
    opts: MultiviewMaskOptions,
) -> list[CameraMaskResult] | None:
    try:
        from . import _traditional_dic as backend
    except Exception:
        return None
    if not hasattr(backend, "core") or not hasattr(backend.core, "build_observation_masks"):
        return None

    cam_indices: list[int] = []
    uv_chunks: list[np.ndarray] = []
    for cam_idx in range(len(cameras)):
        uv = np.asarray(observations.get(cam_idx, np.zeros((0, 2))), dtype=np.float64).reshape((-1, 2))
        if len(uv) == 0:
            continue
        cam_indices.extend([cam_idx] * len(uv))
        uv_chunks.append(uv)
    observation_uv = np.concatenate(uv_chunks, axis=0) if uv_chunks else np.zeros((0, 2), dtype=np.float64)
    widths = [int(width) for height, width in shapes]
    heights = [int(height) for height, width in shapes]
    option_dict = {
        "outlier_k": opts.outlier_k,
        "outlier_knn_scale": opts.outlier_knn_scale,
        "component_radius_scale": opts.component_radius_scale,
        "edge_scale": opts.edge_scale,
        "radius_scale": opts.radius_scale,
        "min_hole_area": opts.min_hole_area,
        "tiny_hole_fill_area": opts.tiny_hole_fill_area,
    }
    try:
        raw_results = backend.core.build_observation_masks(
            widths,
            heights,
            np.asarray(cam_indices, dtype=np.int32),
            observation_uv,
            option_dict,
        )
    except Exception:
        return None

    results: list[CameraMaskResult] = []
    for cam_idx, raw in enumerate(raw_results):
        results.append(
            CameraMaskResult(
                camera_index=int(raw["camera_index"]),
                camera_label=_camera_label(cameras[cam_idx], cam_idx),
                mask=np.asarray(raw["mask"], dtype=np.uint8).astype(bool),
                hull_mask=np.asarray(raw["hull_mask"], dtype=np.uint8).astype(bool),
                supported_mask=np.asarray(raw["supported_mask"], dtype=np.uint8).astype(bool),
                rejected_hole_mask=np.asarray(raw["rejected_hole_mask"], dtype=np.uint8).astype(bool),
                observations=np.asarray(raw["observations"], dtype=np.float64).reshape((-1, 2)),
                clean_observations=np.asarray(raw["clean_observations"], dtype=np.float64).reshape((-1, 2)),
                n_triangles_raw=int(raw["n_triangles_raw"]),
                n_triangles_valid=int(raw["n_triangles_valid"]),
                n_holes_detected=int(raw["n_holes_detected"]),
                n_holes_filled_as_speckle=int(raw["n_holes_filled_as_speckle"]),
                n_holes_rejected=int(raw["n_holes_rejected"]),
            )
        )
    return results


def _build_masks_with_python_fallback(
    calibration: Any,
    image_shapes: Mapping[int | str, tuple[int, int]] | Sequence[tuple[int, int]] | None = None,
    *,
    cameras: Sequence[Any] | None = None,
    observations: Mapping[int, np.ndarray] | None = None,
    shapes: Sequence[tuple[int, int]] | None = None,
    images: Mapping[int, np.ndarray] | None = None,
    reference_images: Mapping[int | str, str | Path | np.ndarray] | Sequence[str | Path | np.ndarray] | None = None,
    options: MultiviewMaskOptions,
) -> list[CameraMaskResult]:
    cameras = list(cameras) if cameras is not None else _extract_cameras(calibration)
    observations = observations if observations is not None else collect_observations_by_camera(calibration, len(cameras))
    images = dict(images) if images is not None else _load_reference_images(reference_images, len(cameras))
    shapes = list(shapes) if shapes is not None else _resolve_image_shapes(cameras, image_shapes, images)

    results: list[CameraMaskResult] = []
    for cam_idx, camera in enumerate(cameras):
        height, width = shapes[cam_idx]
        uv = observations.get(cam_idx, np.zeros((0, 2), dtype=np.float64))
        results.append(_build_single_mask(cam_idx, _camera_label(camera, cam_idx), height, width, uv, images.get(cam_idx), options))
    return results


def _coerce_mask_options(options: MultiviewMaskOptions | Mapping[str, Any] | None) -> MultiviewMaskOptions:
    if options is None:
        return MultiviewMaskOptions()
    if isinstance(options, MultiviewMaskOptions):
        return options
    valid_fields = set(MultiviewMaskOptions.__dataclass_fields__)
    return MultiviewMaskOptions(**{k: v for k, v in dict(options).items() if k in valid_fields})


def _extract_cameras(calibration: Any) -> list[Any]:
    if isinstance(calibration, Mapping):
        return list(calibration.get("cameras", []))
    return list(getattr(calibration, "cameras", []))


def _extract_sparse_points(calibration: Any) -> list[Any]:
    if isinstance(calibration, Mapping):
        return list(calibration.get("points3d", calibration.get("sparse_points", [])))
    return list(getattr(calibration, "sparse_points", []))


def _point_observations(point: Any) -> list[tuple[int, np.ndarray]]:
    raw = point.get("observations", []) if isinstance(point, Mapping) else getattr(point, "observations", [])
    out: list[tuple[int, np.ndarray]] = []
    for obs in raw:
        if isinstance(obs, Mapping):
            cam = int(obs.get("camera_index", obs.get("image_index", -1)))
            uv = np.asarray(obs.get("uv", obs.get("point", [np.nan, np.nan])), dtype=np.float64)
        else:
            cam = int(getattr(obs, "image_index", getattr(obs, "camera_index", -1)))
            uv = np.asarray(getattr(obs, "point"), dtype=np.float64)
        out.append((cam, uv.reshape(-1)[:2]))
    return out


def _resolve_image_shapes(
    cameras: Sequence[Any],
    image_shapes: Mapping[int | str, tuple[int, int]] | Sequence[tuple[int, int]] | None,
    images: Mapping[int, np.ndarray],
) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for idx, camera in enumerate(cameras):
        shape = _shape_from_mapping(image_shapes, idx, _camera_label(camera, idx))
        if shape is None and idx in images:
            shape = tuple(int(v) for v in images[idx].shape[:2])
        if shape is None:
            h, w = _camera_height_width(camera)
            if h > 0 and w > 0:
                shape = (h, w)
        if shape is None:
            raise ValueError(f"Image shape for camera {idx} is required.")
        shapes.append((int(shape[0]), int(shape[1])))
    return shapes


def _shape_from_mapping(
    image_shapes: Mapping[int | str, tuple[int, int]] | Sequence[tuple[int, int]] | None,
    idx: int,
    label: str,
) -> tuple[int, int] | None:
    if image_shapes is None:
        return None
    if isinstance(image_shapes, Mapping):
        value = image_shapes.get(idx, image_shapes.get(str(idx), image_shapes.get(label)))
        return tuple(value) if value is not None else None
    return tuple(image_shapes[idx]) if idx < len(image_shapes) else None


def _camera_height_width(camera: Any) -> tuple[int, int]:
    if isinstance(camera, Mapping):
        return int(camera.get("image_height", 0)), int(camera.get("image_width", 0))
    return int(getattr(camera, "image_height", 0)), int(getattr(camera, "image_width", 0))


def _camera_label(camera: Any, idx: int) -> str:
    label = camera.get("label", "") if isinstance(camera, Mapping) else getattr(camera, "label", "")
    label = str(label or f"cam_{idx}")
    return Path(label).stem.replace(" ", "_")


def _camera_name_order(cam_names: Sequence[str]) -> list[int]:
    return sorted(range(len(cam_names)), key=lambda idx: _natural_camera_key(cam_names[idx], idx))


def _natural_camera_key(name: str, fallback: int) -> tuple[str, int]:
    tail = str(name).rsplit("_", 1)[-1]
    return (str(name).rsplit("_", 1)[0], int(tail)) if tail.isdigit() else (str(name), fallback)


def _camera_centers(cameras: Sequence[Any]) -> np.ndarray | None:
    centers = []
    for camera in cameras:
        center = _camera_center(camera)
        if center is None:
            return None
        centers.append(center)
    return np.asarray(centers, dtype=np.float64).reshape((-1, 3))


def _camera_center(camera: Any) -> np.ndarray | None:
    if isinstance(camera, Mapping):
        for key in ("camera_center", "center", "camera_center_world"):
            if key in camera:
                arr = np.asarray(camera[key], dtype=np.float64).reshape(-1)
                if arr.size == 3:
                    return arr
        if "R" in camera and "t" in camera:
            R = np.asarray(camera["R"], dtype=np.float64).reshape((3, 3))
            t = np.asarray(camera["t"], dtype=np.float64).reshape(3)
            return -R.T @ t
        if "R_world_to_camera" in camera and "t_world_to_camera" in camera:
            R = np.asarray(camera["R_world_to_camera"], dtype=np.float64).reshape((3, 3))
            t = np.asarray(camera["t_world_to_camera"], dtype=np.float64).reshape(3)
            return -R.T @ t
        return None
    if hasattr(camera, "camera_center"):
        try:
            return np.asarray(camera.camera_center(), dtype=np.float64).reshape(3)
        except Exception:
            pass
    if hasattr(camera, "R") and hasattr(camera, "t"):
        R = np.asarray(getattr(camera, "R"), dtype=np.float64).reshape((3, 3))
        t = np.asarray(getattr(camera, "t"), dtype=np.float64).reshape(3)
        return -R.T @ t
    return None


def _shared_track_counts(calibration: Any, camera_count: int) -> np.ndarray:
    counts = np.zeros((camera_count, camera_count), dtype=np.int32)
    for point in _extract_sparse_points(calibration):
        cams = sorted({cam for cam, _ in _point_observations(point) if 0 <= cam < camera_count})
        for i, cam_a in enumerate(cams):
            for cam_b in cams[i + 1 :]:
                counts[cam_a, cam_b] += 1
                counts[cam_b, cam_a] += 1
    return counts


def _load_reference_images(
    reference_images: Mapping[int | str, str | Path | np.ndarray] | Sequence[str | Path | np.ndarray] | None,
    camera_count: int,
) -> dict[int, np.ndarray]:
    if reference_images is None:
        return {}
    out: dict[int, np.ndarray] = {}
    if isinstance(reference_images, Mapping):
        items = reference_images.items()
    else:
        items = enumerate(reference_images[:camera_count])
    for key, value in items:
        idx = int(key) if str(key).isdigit() else _camera_index_from_label(str(key))
        if idx is not None and 0 <= idx < camera_count:
            out[idx] = _load_image_array(value)
    return out


def _camera_index_from_label(label: str) -> int | None:
    tail = label.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _load_image_array(value: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    path = Path(value)
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))


def save_multiview_masks(
    masks: Sequence[CameraMaskResult],
    output_dir: str | Path,
    *,
    reference_images: Mapping[int, np.ndarray] | None = None,
    options: MultiviewMaskOptions | None = None,
) -> dict[str, str]:
    """Write multiview mask, overlay, debug, and metadata artifacts."""

    root = Path(output_dir)
    mask_dir = root / "mask"
    overlay_dir = root / "overlay"
    debug_dir = root / "debug"
    for path in (mask_dir, overlay_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)

    opts = options or MultiviewMaskOptions()
    meta: dict[str, Any] = {
        "mode": "auto_from_multiview_observations",
        "config": asdict(opts),
        "cameras": [],
    }
    for result in masks:
        stem = result.camera_label
        np.save(mask_dir / f"{stem}_mask.npy", result.mask.astype(bool))
        _write_gray(mask_dir / f"{stem}_mask.png", result.mask)
        _write_gray(debug_dir / f"{stem}_hull.png", result.hull_mask)
        _write_gray(debug_dir / f"{stem}_delaunay_supported.png", result.supported_mask)
        _write_gray(debug_dir / f"{stem}_rejected_holes.png", result.rejected_hole_mask)
        image = reference_images.get(result.camera_index) if reference_images else None
        if image is not None:
            _write_rgb(overlay_dir / f"{stem}_overlay.png", _make_overlay(image, result, opts))
        rows, cols = np.where(result.mask)
        meta["cameras"].append(
            {
                "camera_index": int(result.camera_index),
                "camera_label": result.camera_label,
                "mask_pixels": int(result.mask.sum()),
                "hull_pixels": int(result.hull_mask.sum()),
                "supported_pixels": int(result.supported_mask.sum()),
                "rejected_hole_pixels": int(result.rejected_hole_mask.sum()),
                "u_min": float(cols.min()) if cols.size else 0.0,
                "u_max": float(cols.max()) if cols.size else 0.0,
                "v_min": float(rows.min()) if rows.size else 0.0,
                "v_max": float(rows.max()) if rows.size else 0.0,
                "n_observations": int(len(result.observations)),
                "n_points_after_outlier_filter": int(len(result.clean_observations)),
                "n_triangles_raw": int(result.n_triangles_raw),
                "n_triangles_valid": int(result.n_triangles_valid),
                "n_holes_detected": int(result.n_holes_detected),
                "n_holes_filled_as_speckle": int(result.n_holes_filled_as_speckle),
                "n_holes_rejected": int(result.n_holes_rejected),
            }
        )

    if reference_images:
        _save_summary_grid(root / "mask_summary.png", masks, reference_images, opts)
    meta_path = root / "auto_roi_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "mask_dir": str(mask_dir),
        "overlay_dir": str(overlay_dir),
        "debug_dir": str(debug_dir),
        "meta_json": str(meta_path),
        "summary_png": str(root / "mask_summary.png") if reference_images else "",
    }


def _build_single_mask(
    cam_idx: int,
    label: str,
    height: int,
    width: int,
    uv: np.ndarray,
    image: np.ndarray | None,
    opts: MultiviewMaskOptions,
) -> CameraMaskResult:
    uv = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv_in = uv[in_frame]
    uv_clean = _remove_feature_outliers(uv_in, opts)
    empty = np.zeros((height, width), dtype=bool)
    if len(uv_clean) < 3:
        return CameraMaskResult(cam_idx, label, empty, empty.copy(), empty.copy(), empty.copy(), uv_in, uv_clean)

    hull = _convex_hull(uv_clean)
    hull_mask = _rasterize_polygon(hull, height, width)
    supported, n_tri_raw, n_tri_valid = _build_delaunay_support_mask(uv_clean, height, width, opts)
    supported &= hull_mask
    mask, rejected, hole_stats = _classify_and_fill_holes(image, hull_mask, supported, opts)
    return CameraMaskResult(
        cam_idx,
        label,
        mask,
        hull_mask,
        supported,
        rejected,
        uv_in,
        uv_clean,
        n_triangles_raw=n_tri_raw,
        n_triangles_valid=n_tri_valid,
        n_holes_detected=hole_stats["detected"],
        n_holes_filled_as_speckle=hole_stats["filled"],
        n_holes_rejected=hole_stats["rejected"],
    )


def _remove_feature_outliers(uv: np.ndarray, opts: MultiviewMaskOptions) -> np.ndarray:
    if len(uv) <= max(3, opts.outlier_k + 1):
        return uv
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return uv
    tree = cKDTree(uv)
    dists, _ = tree.query(uv, k=opts.outlier_k + 1)
    nn = dists[:, 1]
    kth = dists[:, -1]
    median_nn = float(np.median(nn[nn > 0])) if np.any(nn > 0) else float(np.median(kth))
    if not np.isfinite(median_nn) or median_nn <= 0:
        return uv
    uv_density = uv[kth <= opts.outlier_knn_scale * median_nn]
    if len(uv_density) < 3:
        uv_density = uv
    return _keep_largest_radius_component(uv_density, opts.component_radius_scale * median_nn)


def _keep_largest_radius_component(uv: np.ndarray, radius: float) -> np.ndarray:
    if len(uv) < 3 or radius <= 0:
        return uv
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return uv
    pairs = list(cKDTree(uv).query_pairs(radius))
    if not pairs:
        return uv
    parent = np.arange(len(uv))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(len(uv))])
    labels, counts = np.unique(roots, return_counts=True)
    keep = roots == labels[np.argmax(counts)]
    return uv[keep] if np.count_nonzero(keep) >= 3 else uv


def _build_delaunay_support_mask(
    uv: np.ndarray,
    height: int,
    width: int,
    opts: MultiviewMaskOptions,
) -> tuple[np.ndarray, int, int]:
    d_nn = _median_nn_distance(uv)
    if d_nn <= 0:
        return np.zeros((height, width), dtype=bool), 0, 0
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return _rasterize_polygon(_convex_hull(uv), height, width), 0, 0
    try:
        tri = Delaunay(uv)
    except Exception:
        return np.zeros((height, width), dtype=bool), 0, 0
    valid = _filter_triangles(uv, np.asarray(tri.simplices), d_nn, opts)
    valid_tris = np.asarray(tri.simplices)[valid]
    if len(valid_tris) == 0:
        return np.zeros((height, width), dtype=bool), int(len(tri.simplices)), 0
    return _rasterize_triangles(uv, valid_tris, height, width), int(len(tri.simplices)), int(len(valid_tris))


def _median_nn_distance(uv: np.ndarray) -> float:
    if len(uv) < 2:
        return 0.0
    try:
        from scipy.spatial import cKDTree
    except Exception:
        diffs = uv[:, None, :] - uv[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        dist[dist <= 0] = np.inf
        nearest = np.min(dist, axis=1)
        finite = nearest[np.isfinite(nearest)]
        return float(np.median(finite)) if finite.size else 0.0
    dists, _ = cKDTree(uv).query(uv, k=2)
    positive = dists[:, 1][dists[:, 1] > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _filter_triangles(
    uv: np.ndarray,
    tri_indices: np.ndarray,
    d_nn: float,
    opts: MultiviewMaskOptions,
) -> np.ndarray:
    vertices = uv[tri_indices]
    e0 = vertices[:, 1] - vertices[:, 0]
    e1 = vertices[:, 2] - vertices[:, 1]
    e2 = vertices[:, 0] - vertices[:, 2]
    l0 = np.linalg.norm(e0, axis=-1)
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    l_max = np.maximum(np.maximum(l0, l1), l2)
    area = 0.5 * np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0])
    radius = (l0 * l1 * l2) / (4.0 * np.maximum(area, 1e-12))
    return (l_max < opts.edge_scale * d_nn) & (radius < opts.radius_scale * d_nn)


def _classify_and_fill_holes(
    image: np.ndarray | None,
    hull_mask: np.ndarray,
    supported_mask: np.ndarray,
    opts: MultiviewMaskOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    candidate_holes = hull_mask & ~supported_mask
    final = supported_mask.copy()
    rejected = np.zeros_like(hull_mask, dtype=bool)
    counts = {"detected": 0, "filled": 0, "rejected": 0}
    labels = _connected_components(candidate_holes)
    ref_texture = _texture_metrics(image, supported_mask)
    for label_id in range(1, int(labels.max()) + 1):
        hole = labels == label_id
        area = int(np.count_nonzero(hole))
        if area < opts.min_hole_area:
            continue
        counts["detected"] += 1
        if area <= opts.tiny_hole_fill_area or _is_speckle_like(_texture_metrics(image, hole), ref_texture, opts):
            final[hole] = True
            counts["filled"] += 1
        else:
            rejected[hole] = True
            counts["rejected"] += 1
    final &= hull_mask
    return final, rejected, counts


def _connected_components(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        current = 0
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(x, y)]
                labels[y, x] = current
                while stack:
                    sx, sy = stack.pop()
                    for ny in range(max(0, sy - 1), min(height, sy + 2)):
                        for nx in range(max(0, sx - 1), min(width, sx + 2)):
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((nx, ny))
        return labels


def _texture_metrics(image: np.ndarray | None, mask: np.ndarray) -> dict[str, float]:
    if image is None or np.count_nonzero(mask) == 0:
        return {"std": 0.0, "lap_std": 0.0, "grad_mean": 0.0}
    gray = _as_gray(image).astype(np.float32)
    values = gray[mask]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.gradient(gx)[1] + np.gradient(gy)[0]
    return {
        "std": float(np.std(values)),
        "lap_std": float(np.std(lap[mask])),
        "grad_mean": float(np.mean(grad[mask])),
    }


def _is_speckle_like(hole: dict[str, float], ref: dict[str, float], opts: MultiviewMaskOptions) -> bool:
    std_ok = hole["std"] >= max(opts.min_speckle_std, opts.speckle_std_ratio * ref["std"])
    lap_ok = hole["lap_std"] >= max(opts.min_speckle_lap, opts.speckle_lap_ratio * ref["lap_std"])
    grad_ok = hole["grad_mean"] >= opts.speckle_grad_ratio * ref["grad_mean"]
    return bool(std_ok and lap_ok and grad_ok)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _rasterize_triangles(uv: np.ndarray, tri_indices: np.ndarray, height: int, width: int) -> np.ndarray:
    vertices = np.asarray(uv[tri_indices], dtype=np.float64)
    try:
        import cv2

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, np.round(vertices).astype(np.int32), 1)
        return mask.astype(bool)
    except Exception:
        mask = np.zeros((height, width), dtype=bool)
        for tri in vertices:
            mask |= _rasterize_polygon(tri, height, width)
        return mask


def _rasterize_polygon(vertices: np.ndarray, height: int, width: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 2))
    if len(vertices) < 3:
        return np.zeros((height, width), dtype=bool)
    try:
        import cv2

        out = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(out, [np.round(vertices).astype(np.int32)], 1)
        return out.astype(bool)
    except Exception:
        try:
            from PIL import Image, ImageDraw

            image = Image.new("L", (width, height), 0)
            ImageDraw.Draw(image).polygon([tuple(p) for p in vertices], outline=1, fill=1)
            return np.asarray(image, dtype=np.uint8).astype(bool)
        except Exception:
            return _rasterize_polygon_numpy(vertices, height, width)


def _rasterize_polygon_numpy(vertices: np.ndarray, height: int, width: int) -> np.ndarray:
    xmin = max(0, int(np.floor(np.min(vertices[:, 0]))))
    xmax = min(width - 1, int(np.ceil(np.max(vertices[:, 0]))))
    ymin = max(0, int(np.floor(np.min(vertices[:, 1]))))
    ymax = min(height - 1, int(np.ceil(np.max(vertices[:, 1]))))
    out = np.zeros((height, width), dtype=bool)
    if xmin > xmax or ymin > ymax:
        return out
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
    inside = np.zeros(xs.shape, dtype=bool)
    x0 = vertices[-1, 0]
    y0 = vertices[-1, 1]
    for x1, y1 in vertices:
        crosses = ((y1 > ys) != (y0 > ys)) & (xs < (x0 - x1) * (ys - y1) / ((y0 - y1) + 1e-12) + x1)
        inside ^= crosses
        x0, y0 = x1, y1
    out[ymin : ymax + 1, xmin : xmax + 1] = inside
    return out


def _as_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    return np.mean(arr[..., :3], axis=2)


def _make_overlay(image: np.ndarray, result: CameraMaskResult, opts: MultiviewMaskOptions) -> np.ndarray:
    gray = _as_gray(image).astype(np.float64)
    base = np.stack([gray, gray, gray], axis=-1)
    color = np.zeros_like(base)
    color[result.mask] = [0, 180, 0]
    color[result.rejected_hole_mask] = [220, 0, 0]
    alpha = float(opts.overlay_alpha)
    return np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _save_summary_grid(
    path: Path,
    masks: Sequence[CameraMaskResult],
    reference_images: Mapping[int, np.ndarray],
    opts: MultiviewMaskOptions,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(masks)
    cols = min(4, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), dpi=160, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, result in zip(axes_flat, masks):
        image = reference_images.get(result.camera_index)
        if image is None:
            overlay = np.stack([result.mask.astype(np.uint8) * 255] * 3, axis=-1)
        else:
            overlay = _make_overlay(image, result, opts)
        ax.imshow(overlay)
        ax.set_title(f"{result.camera_label}: {int(result.mask.sum()) // 1000}K px", fontsize=9)
        ax.axis("off")
    for ax in axes_flat[len(masks) :]:
        ax.axis("off")
    fig.suptitle("Multiview ROI masks: green=ROI, red=rejected holes", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def _write_gray(path: Path, mask: np.ndarray) -> None:
    arr = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    try:
        import cv2

        cv2.imwrite(str(path), arr)
    except Exception:
        from PIL import Image

        Image.fromarray(arr, mode="L").save(path)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    try:
        import cv2

        cv2.imwrite(str(path), image[..., ::-1])
    except Exception:
        from PIL import Image

        Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def multiview(reference_images, deformed_images=None, calibration=None, solver="subset"):
    """
    Run multiview 3D DIC orchestration.

    Current implementation checkpoint:
        mask generation is available through generate_masks_from_calibration().
        The full N-camera 2D-DIC assembly and C++ reconstruction binding will be
        connected in the next stage.
    """
    raise NotImplementedError(
        "Full multiview DIC orchestration is not implemented yet. "
        "Use generate_masks_from_calibration() for the current mask-generation stage."
    )
