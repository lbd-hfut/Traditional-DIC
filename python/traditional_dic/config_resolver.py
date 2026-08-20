"""Authoritative, side-effect-free workflow configuration resolution.

F2 deliberately normalizes workflow configuration without changing the
low-level Python API defaults.  A resolved workflow receives one deterministic
mapping, explicit provenance, and capability validation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from ._runtime import default_config_path, resolve_config_reference, runtime_root


CONFIG_WORKFLOW_KINDS = frozenset({"subset_2d", "mesh_2d", "stereo_3d", "multiview_3d", "calibration"})
DEFAULT_REPOSITORY_ROOT = runtime_root()


class ConfigResolutionError(ValueError):
    """Structured fail-closed configuration error."""

    def __init__(self, code: str, message: str, *, path: str | None = None, details: Mapping[str, Any] | None = None) -> None:
        self.code = str(code)
        self.path = path
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class ConfigIssue:
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class ResolvedConfig:
    workflow_kind: str
    values: Mapping[str, Any]
    source_files: tuple[str, ...]
    provenance: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    warnings: tuple[ConfigIssue, ...] = ()
    validation: Mapping[str, Any] = field(default_factory=lambda: {"valid": True})

    def backend_config(self) -> dict[str, Any]:
        """Return a detached mapping suitable for existing Python APIs."""
        return copy.deepcopy(dict(self.values))

    def to_dict(self, *, repository_root: str | Path | None = None) -> dict[str, Any]:
        root = Path(repository_root or DEFAULT_REPOSITORY_ROOT).resolve()
        return {
            "workflow_kind": self.workflow_kind,
            "source_files": [_portable_path(path, root) for path in self.source_files],
            "values": _portable(copy.deepcopy(dict(self.values)), root),
            "provenance": _portable(copy.deepcopy(dict(self.provenance)), root),
            "capabilities": _portable(copy.deepcopy(dict(self.capabilities)), root),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "validation": _portable(copy.deepcopy(dict(self.validation)), root),
        }

    def to_json(self, *, repository_root: str | Path | None = None, indent: int = 2) -> str:
        return json.dumps(self.to_dict(repository_root=repository_root), indent=indent, sort_keys=True) + "\n"


_DEFAULT_FILES = {
    "subset_2d": "config/subset_2d.yaml",
    "mesh_2d": "config/mesh_2d.yaml",
    "calibration": "config/calibration.yaml",
    "stereo_3d": "config/stereo_3d.yaml",
    "multiview_3d": "config/multiview_3d.yaml",
}

_TOP_KEYS = {
    "subset_2d": {"solver", "subset", "shape_function", "optimization", "correlation", "interpolation", "initialization", "seed_selection", "reliability_propagation", "strain"},
    "mesh_2d": {"solver", "mesh", "mesh_generation", "optimization", "interpolation", "initialization", "strain"},
    "calibration": {"calibration", "board", "detection", "mono_calibration", "stereo_calibration", "self_calibration", "scale"},
    "stereo_3d": {"workflow", "solver", "configs", "reconstruction", "strain"},
    "multiview_3d": {"mode", "self_calibration", "camera_pair_selection", "maskGen", "maskgen", "pairwise_2d_dic", "pairwise_3d_dic", "scale", "surface_stitch", "pairwise_surface_stitch", "triangulation", "strain", "workflow", "configs"},
}

_SECTION_KEYS = {
    "solver": {"method", "element_types"},
    "subset": {"radius", "truncate_roi_subsets", "min_valid_sample_ratio", "min_valid_samples"},
    "shape_function": {"order"},
    "optimization": {"method", "max_iterations", "convergence_threshold", "regularization_alpha", "objective", "mirror_image_padding"},
    "correlation": {"criterion"},
    "interpolation": {"method", "degree", "border", "use_exact_prefilter", "precompute_local_blocks"},
    "mesh": {"element_type"},
    "mesh_generation": {"method", "element_type", "target_element_size", "min_element_size", "max_element_size", "fit_roi_boundary", "remove_outside_elements", "remove_outside_nodes", "allow_partial_elements", "min_element_quality", "max_aspect_ratio", "min_jacobian", "nodes_file", "elements_file"},
    "initialization": {"method", "integer_search", "subpixel_refinement", "boundary_interpolation_init", "boundary_direct_prior_seed", "fedic_fft", "pyramid", "sift_prior", "quality_control"},
    "seed_selection": {"method", "seed_count", "threads", "quality_metric", "max_znssd", "min_zncc", "max_ssd", "min_displacement_norm", "min_texture_std", "kmeans_iterations", "kmeans_sample_limit"},
    "reliability_propagation": {"spacing", "max_znssd"},
    "strain": {"enabled", "method", "radius", "min_samples", "measure", "min_face_area"},
    "workflow": {"calibrate", "compute_fields", "reconstruct", "visualize_calibration"},
    "configs": {"subset", "mesh", "calibration"},
    "reconstruction": {"quality_metric", "max_znssd", "min_correlation", "max_reprojection_error_px", "remove_rigid_body_motion"},
    "pairwise_2d_dic": {"solver", "run_subset", "run_mesh", "mesh_types", "subset_config", "mesh_config", "mesh_target_element_size", "pair_roi_mode", "pair_roi_disparity_quantile", "pair_roi_erode_pixels", "subset_solve_roi_mode", "image_dir", "reference_frame", "deformed_frame", "pair_roi_source"},
    "pairwise_3d_dic": {"solver", "quality_metric", "max_znssd", "min_correlation", "max_reprojection_error_px", "remove_rigid_body_motion", "write_surface_strain"},
    "surface_stitch": {"mode", "solver", "max_reprojection_error_px", "max_quality", "triangle_edge_scale", "min_gap_factor", "outlier_neighbor_count", "outlier_distance_sigma", "outlier_displacement_sigma", "outlier_face_edge_scale", "smooth_displacement_knn", "pairwise_3d_dir", "output_dir", "calibration_dir"},
    "self_calibration": {"backend", "method", "matching_window", "wrap_matching", "max_features", "match_ratio", "ransac_reprojection_threshold", "min_triangulation_angle_degrees", "min_inlier_matches", "initial_focal_length_factor", "initial_image1", "initial_image2", "abs_pose_min_num_inliers", "abs_pose_min_inlier_ratio", "abs_pose_max_error", "filter_max_reproj_error", "ba_local_num_images", "ignore_two_view_tracks", "refine_bundle", "share_intrinsics", "refine_focal_length", "refine_principal_point", "refine_extra_params"},
    "camera_pair_selection": {"mode", "wrap", "manual"},
    "maskGen": {"pair_roi_source", "feature_method", "max_features", "match_ratio", "mutual_check", "ransac_reprojection_threshold", "min_matches", "roi_support", "robust_outlier", "outlier_mad_scale", "alpha_radius_scale"},
    "scale": {"output_file", "use_meta_observations", "board_rows", "board_cols", "square_size", "max_reprojection_error", "trim_fraction", "min_common_corners", "allow_meta_camera_model_fallback"},
    "triangulation": {"minimum_views", "nonlinear_refinement"},
    "calibration": {"mode"},
    "board": {"type", "rows", "cols", "spacing", "inner_rows", "inner_cols", "square_size", "circle_spacing"},
    "detection": {"refine_corners", "normalize_image", "max_iterations", "epsilon"},
    "mono_calibration": {"model", "estimate_tangential_distortion", "estimate_k3", "max_iterations", "epsilon"},
    "stereo_calibration": {"model", "fix_intrinsics", "estimate_tangential_distortion", "estimate_k3", "max_iterations", "epsilon", "reject_outlier_pairs", "outlier_mad_factor", "left_right_error_ratio_threshold", "left_right_error_abs_threshold", "min_pairs_after_rejection"},
}

_NESTED_SECTION_KEYS = {
    "initialization.integer_search": {"subset_radius", "search_radius", "sift_enabled", "pyramid_enabled", "pyramid_scale", "pyramid_refinement_radius"},
    "initialization.subpixel_refinement": {"enabled", "shape_function", "optimizer", "subset_radius", "max_iterations", "convergence_threshold"},
    "initialization.fedic_fft": {"window_size", "search_radius", "mirror_boundary_fallback"},
    "initialization.pyramid": {"enabled", "num_levels", "scale_factor", "coarse_search_radius", "refinement_radius", "window_size"},
    "initialization.sift_prior": {"enabled", "max_features", "ratio_threshold", "robust_mad_factor", "interpolation_neighbors", "interpolation_radius"},
    "initialization.quality_control": {"enabled", "min_zncc", "max_znssd", "fedic_qfactor_enabled", "fedic_qfactor_std_factor", "neighbor_mad_factor", "max_neighbor_deviation", "interpolation_neighbors"},
}


def _portable_path(value: str | Path, root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _portable(value: Any, root: Path) -> Any:
    if isinstance(value, Path):
        return _portable_path(value, root)
    if isinstance(value, Mapping):
        return {str(key): _portable(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item, root) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _deep_set(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = mapping
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ConfigResolutionError("INVALID_OVERRIDE_PATH", f"override path traverses a scalar: {dotted}", path=dotted)
        current = current[part]
    current[parts[-1]] = copy.deepcopy(value)


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(_flatten_leaves(child, path))
    return output


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", f"{path} must be a mapping", path=path)
    return dict(value)


def _check_keys(config: Mapping[str, Any], workflow: str) -> None:
    unknown = sorted(set(config) - _TOP_KEYS[workflow])
    if unknown:
        raise ConfigResolutionError("UNKNOWN_CONFIG_KEY", f"unknown {workflow} configuration key(s): {unknown}", path=unknown[0])
    for section, allowed in _SECTION_KEYS.items():
        value = config.get(section)
        if isinstance(value, Mapping):
            unknown_nested = sorted(set(value) - allowed)
            if unknown_nested:
                raise ConfigResolutionError("UNKNOWN_CONFIG_KEY", f"unknown key(s) under {section}: {unknown_nested}", path=f"{section}.{unknown_nested[0]}")
    for section_path, allowed in _NESTED_SECTION_KEYS.items():
        value: Any = config
        for component in section_path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                value = None
                break
            value = value[component]
        if isinstance(value, Mapping):
            unknown_nested = sorted(set(value) - allowed)
            if unknown_nested:
                raise ConfigResolutionError("UNKNOWN_CONFIG_KEY", f"unknown key(s) under {section_path}: {unknown_nested}", path=f"{section_path}.{unknown_nested[0]}")


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", f"{path} must be boolean", path=path)
    return value


def _int(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", f"{path} must be an integer", path=path)
    if positive and value <= 0:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be > 0", path=path)
    if nonnegative and value < 0:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be >= 0", path=path)
    return value


def _number(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", f"{path} must be numeric", path=path)
    number = float(value)
    if positive and number <= 0:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be > 0", path=path)
    if nonnegative and number < 0:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be >= 0", path=path)
    if unit_interval and not 0.0 <= number <= 1.0:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be in [0, 1]", path=path)
    return number


def _enum(value: Any, path: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", f"{path} must be a string", path=path)
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", f"{path} must be one of {sorted(allowed)}", path=path)
    return normalized


def _normalize_subset(config: dict[str, Any]) -> dict[str, Any]:
    _check_keys(config, "subset_2d")
    if _enum(config.get("solver", "subset"), "solver", {"subset"}) != "subset":
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "subset_2d only supports the subset solver", path="solver")
    subset = _require_mapping(config.get("subset", {}), "subset")
    subset["radius"] = _int(subset.get("radius", 41), "subset.radius", positive=True)
    subset["truncate_roi_subsets"] = _bool(subset.get("truncate_roi_subsets", False), "subset.truncate_roi_subsets")
    subset["min_valid_sample_ratio"] = _number(subset.get("min_valid_sample_ratio", 0.5), "subset.min_valid_sample_ratio", unit_interval=True)
    subset["min_valid_samples"] = _int(subset.get("min_valid_samples", 12), "subset.min_valid_samples", positive=True)
    config["subset"] = subset
    shape = _require_mapping(config.get("shape_function", {}), "shape_function")
    order = shape.get("order", 1)
    if isinstance(order, str):
        aliases = {"first": 1, "first_order": 1, "second": 2, "second_order": 2}
        if order.strip().lower() not in aliases:
            raise ConfigResolutionError("INVALID_CONFIG_VALUE", "shape_function.order must be first or second order", path="shape_function.order")
        order = aliases[order.strip().lower()]
    shape["order"] = _int(order, "shape_function.order", positive=True)
    if shape["order"] not in {1, 2}:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", "shape_function.order must be 1 or 2", path="shape_function.order")
    config["shape_function"] = shape
    optimization = _require_mapping(config.get("optimization", {}), "optimization")
    method = optimization.get("method", "icgn")
    if not isinstance(method, str):
        raise ConfigResolutionError("INVALID_CONFIG_TYPE", "optimization.method must be a string", path="optimization.method")
    method = {"forward_gauss_newton": "forward_gauss_newton", "fgn": "forward_gauss_newton", "icgn": "icgn"}.get(method.strip().lower(), method.strip().lower())
    if method not in {"icgn", "forward_gauss_newton"}:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", "unsupported subset optimization method", path="optimization.method")
    optimization["method"] = method
    optimization["max_iterations"] = _int(optimization.get("max_iterations", 50), "optimization.max_iterations", positive=True)
    optimization["convergence_threshold"] = _number(optimization.get("convergence_threshold", 1.0e-6), "optimization.convergence_threshold", positive=True)
    config["optimization"] = optimization
    correlation = _require_mapping(config.get("correlation", {}), "correlation")
    correlation["criterion"] = _enum(correlation.get("criterion", "znssd"), "correlation.criterion", {"znssd", "ssd"})
    config["correlation"] = correlation
    interpolation = _require_mapping(config.get("interpolation", {}), "interpolation")
    interpolation["method"] = _enum(interpolation.get("method", "bspline"), "interpolation.method", {"bspline"})
    interpolation["degree"] = _int(interpolation.get("degree", 5), "interpolation.degree", positive=True)
    if interpolation["degree"] not in {1, 3, 5}:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", "interpolation.degree must be 1, 3, or 5", path="interpolation.degree")
    config["interpolation"] = interpolation
    initialization = _require_mapping(config.get("initialization", {}), "initialization")
    initialization["method"] = _enum(initialization.get("method", "integer_search"), "initialization.method", {"integer_search"})
    integer = _require_mapping(initialization.get("integer_search", {}), "initialization.integer_search")
    integer["subset_radius"] = _int(integer.get("subset_radius", 10), "initialization.integer_search.subset_radius", positive=True)
    integer["search_radius"] = _int(integer.get("search_radius", 30), "initialization.integer_search.search_radius", nonnegative=True)
    for key in ("sift_enabled", "pyramid_enabled"):
        integer[key] = _bool(integer.get(key, key == "pyramid_enabled"), f"initialization.integer_search.{key}")
    initialization["integer_search"] = integer
    subpixel = _require_mapping(initialization.get("subpixel_refinement", {}), "initialization.subpixel_refinement")
    subpixel["enabled"] = _bool(subpixel.get("enabled", True), "initialization.subpixel_refinement.enabled")
    subpixel["subset_radius"] = _int(subpixel.get("subset_radius", subset["radius"]), "initialization.subpixel_refinement.subset_radius", positive=True)
    subpixel["max_iterations"] = _int(subpixel.get("max_iterations", optimization["max_iterations"]), "initialization.subpixel_refinement.max_iterations", positive=True)
    subpixel["convergence_threshold"] = _number(subpixel.get("convergence_threshold", optimization["convergence_threshold"]), "initialization.subpixel_refinement.convergence_threshold", positive=True)
    initialization["subpixel_refinement"] = subpixel
    config["initialization"] = initialization
    seed = _require_mapping(config.get("seed_selection", {}), "seed_selection")
    seed["seed_count"] = _int(seed.get("seed_count", 16), "seed_selection.seed_count", positive=True)
    seed["max_znssd"] = _number(seed.get("max_znssd", 2.0), "seed_selection.max_znssd", nonnegative=True)
    seed["min_displacement_norm"] = _number(seed.get("min_displacement_norm", 0.0), "seed_selection.min_displacement_norm", nonnegative=True)
    seed["min_texture_std"] = _number(seed.get("min_texture_std", 0.02), "seed_selection.min_texture_std", nonnegative=True)
    config["seed_selection"] = seed
    propagation = _require_mapping(config.get("reliability_propagation", {}), "reliability_propagation")
    propagation["spacing"] = _int(propagation.get("spacing", 3), "reliability_propagation.spacing", positive=True)
    propagation["max_znssd"] = _number(propagation.get("max_znssd", 2.0), "reliability_propagation.max_znssd", nonnegative=True)
    config["reliability_propagation"] = propagation
    strain = _require_mapping(config.get("strain", {}), "strain")
    strain["enabled"] = _bool(strain.get("enabled", True), "strain.enabled")
    strain["radius"] = _number(strain.get("radius", 15.0), "strain.radius", positive=True)
    strain["min_samples"] = _int(strain.get("min_samples", 6), "strain.min_samples", positive=True)
    strain["measure"] = _enum(strain.get("measure", "green_lagrange"), "strain.measure", {"green_lagrange", "infinitesimal"})
    config["strain"] = strain
    config["solver"] = "subset"
    return config


def _normalize_mesh(config: dict[str, Any]) -> dict[str, Any]:
    _check_keys(config, "mesh_2d")
    if _enum(config.get("solver", "mesh"), "solver", {"mesh"}) != "mesh":
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "mesh_2d only supports the mesh solver", path="solver")
    mesh = _require_mapping(config.get("mesh", {}), "mesh")
    mesh["element_type"] = _enum(mesh.get("element_type", "q4"), "mesh.element_type", {"t3", "q4", "q8"}).upper()
    config["mesh"] = mesh
    generation = _require_mapping(config.get("mesh_generation", {}), "mesh_generation")
    generation["method"] = _enum(generation.get("method", "auto"), "mesh_generation.method", {"auto", "manual", "structured", "unstructured"})
    generation["element_type"] = _enum(generation.get("element_type", mesh["element_type"]), "mesh_generation.element_type", {"t3", "q4", "q8"}).upper()
    generation["target_element_size"] = _number(generation.get("target_element_size", 41.0), "mesh_generation.target_element_size", positive=True)
    for key in ("min_element_size", "max_element_size", "min_element_quality"):
        generation[key] = _number(generation.get(key, 0.0), f"mesh_generation.{key}", nonnegative=True)
    for key in ("fit_roi_boundary", "remove_outside_elements", "remove_outside_nodes", "allow_partial_elements"):
        generation[key] = _bool(generation.get(key, key != "allow_partial_elements"), f"mesh_generation.{key}")
    generation["max_aspect_ratio"] = _number(generation.get("max_aspect_ratio", 6.0), "mesh_generation.max_aspect_ratio", positive=True)
    generation["min_jacobian"] = _number(generation.get("min_jacobian", 0.05), "mesh_generation.min_jacobian", positive=True)
    config["mesh_generation"] = generation
    optimization = _require_mapping(config.get("optimization", {}), "optimization")
    method = _enum(optimization.get("method", "fedic_element_icgn"), "optimization.method", {"fedic_element_icgn", "fedic_element_fgn", "icgn", "fgn"})
    optimization["method"] = {"icgn": "fedic_element_icgn", "fgn": "fedic_element_fgn"}.get(method, method)
    optimization["objective"] = _enum(optimization.get("objective", "ssd"), "optimization.objective", {"ssd", "znssd"})
    optimization["max_iterations"] = _int(optimization.get("max_iterations", 50), "optimization.max_iterations", positive=True)
    optimization["convergence_threshold"] = _number(optimization.get("convergence_threshold", 1e-6), "optimization.convergence_threshold", positive=True)
    optimization["regularization_alpha"] = _number(optimization.get("regularization_alpha", 0.0), "optimization.regularization_alpha", nonnegative=True)
    optimization["mirror_image_padding"] = _bool(optimization.get("mirror_image_padding", True), "optimization.mirror_image_padding")
    config["optimization"] = optimization
    interpolation = _require_mapping(config.get("interpolation", {}), "interpolation")
    interpolation["method"] = _enum(interpolation.get("method", "bspline"), "interpolation.method", {"bspline"})
    interpolation["degree"] = _int(interpolation.get("degree", 5), "interpolation.degree", positive=True)
    if interpolation["degree"] not in {1, 3, 5}:
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", "interpolation.degree must be 1, 3, or 5", path="interpolation.degree")
    config["interpolation"] = interpolation
    initialization = _require_mapping(config.get("initialization", {}), "initialization")
    initialization["method"] = _enum(initialization.get("method", "fedic_fft"), "initialization.method", {"fedic_fft"})
    config["initialization"] = initialization
    strain = _require_mapping(config.get("strain", {}), "strain")
    strain["enabled"] = _bool(strain.get("enabled", True), "strain.enabled")
    strain["radius"] = _number(strain.get("radius", 15.0), "strain.radius", positive=True)
    strain["min_samples"] = _int(strain.get("min_samples", 6), "strain.min_samples", positive=True)
    strain["measure"] = _enum(strain.get("measure", "green_lagrange"), "strain.measure", {"green_lagrange", "infinitesimal"})
    config["strain"] = strain
    config["solver"] = "mesh"
    return config


def _normalize_calibration(config: dict[str, Any]) -> dict[str, Any]:
    _check_keys(config, "calibration")
    calibration = _require_mapping(config.get("calibration", {}), "calibration")
    calibration["mode"] = _enum(calibration.get("mode", "stereo"), "calibration.mode", {"mono", "stereo", "self_calibration", "multiview_scale"})
    config["calibration"] = calibration
    board = _require_mapping(config.get("board", {}), "board")
    board["type"] = _enum(board.get("type", "chessboard"), "board.type", {"chessboard", "symmetric_circles", "asymmetric_circles"})
    board["rows"] = _int(board.get("rows", board.get("inner_rows", 0)), "board.rows", positive=True)
    board["cols"] = _int(board.get("cols", board.get("inner_cols", 0)), "board.cols", positive=True)
    board["spacing"] = _number(board.get("spacing", board.get("square_size", 1.0)), "board.spacing", positive=True)
    config["board"] = board
    detection = _require_mapping(config.get("detection", {}), "detection")
    detection["refine_corners"] = _bool(detection.get("refine_corners", True), "detection.refine_corners")
    detection["normalize_image"] = _bool(detection.get("normalize_image", True), "detection.normalize_image")
    detection["max_iterations"] = _int(detection.get("max_iterations", 30), "detection.max_iterations", positive=True)
    detection["epsilon"] = _number(detection.get("epsilon", 1e-3), "detection.epsilon", positive=True)
    config["detection"] = detection
    return config


def _normalize_stereo(config: dict[str, Any], subset: dict[str, Any], calibration: dict[str, Any], *, warnings: list[ConfigIssue]) -> dict[str, Any]:
    _check_keys(config, "stereo_3d")
    solver = _require_mapping(config.get("solver", {}), "solver")
    method = str(solver.get("method", "subset")).strip().lower()
    if method != "subset":
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "stereo_3d only supports Subset-DIC correspondence", path="solver.method")
    config["solver"] = {"method": "subset"}
    config["correspondence_solver"] = "subset"
    config["subset_config"] = subset
    config["calibration_config"] = calibration
    configs = dict(config.get("configs", {}) or {})
    configs.pop("mesh", None)
    config["configs"] = configs
    reconstruction = _require_mapping(config.get("reconstruction", {}), "reconstruction")
    reconstruction["quality_metric"] = _enum(reconstruction.get("quality_metric", "znssd"), "reconstruction.quality_metric", {"znssd", "correlation"})
    reconstruction["max_reprojection_error_px"] = _number(reconstruction.get("max_reprojection_error_px", 5.0), "reconstruction.max_reprojection_error_px", nonnegative=True)
    config["reconstruction"] = reconstruction
    strain = _require_mapping(config.get("strain", {}), "strain")
    if str(strain.get("method", "triangular_cosserat")) != "triangular_cosserat":
        raise ConfigResolutionError("INVALID_CONFIG_VALUE", "stereo_3d.strain.method must be triangular_cosserat", path="strain.method")
    config["strain"] = strain
    return config


def _normalize_multiview(config: dict[str, Any], subset: dict[str, Any], *, warnings: list[ConfigIssue], allow_legacy_mesh: bool) -> dict[str, Any]:
    _check_keys(config, "multiview_3d")
    pairwise = _require_mapping(config.get("pairwise_2d_dic", {}), "pairwise_2d_dic")
    if str(pairwise.get("solver", "subset")).strip().lower() in {"mesh", "both"}:
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d pairwise correspondence is Subset-DIC only", path="pairwise_2d_dic.solver")
    if pairwise.get("run_mesh") is True and not allow_legacy_mesh:
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d no longer accepts pairwise Mesh-DIC", path="pairwise_2d_dic.run_mesh")
    if pairwise.get("run_mesh") is True:
        warnings.append(ConfigIssue("LEGACY_MESH_KEY_IGNORED", "pairwise_2d_dic.run_mesh=true is a legacy field and is normalized to false", "pairwise_2d_dic.run_mesh"))
    if pairwise.get("run_subset") is False:
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d requires Subset-DIC pairwise correspondence", path="pairwise_2d_dic.run_subset")
    pairwise["run_subset"] = True
    pairwise["run_mesh"] = False
    pairwise.pop("mesh_types", None)
    pairwise.pop("mesh_config", None)
    config["pairwise_2d_dic"] = pairwise
    config["pairwise_2d_solver"] = "subset"
    pair3d = _require_mapping(config.get("pairwise_3d_dic", {}), "pairwise_3d_dic")
    if str(pair3d.get("solver", "subset")).strip().lower() != "subset":
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d pairwise reconstruction requires subset", path="pairwise_3d_dic.solver")
    pair3d["solver"] = "subset"
    config["pairwise_3d_dic"] = pair3d
    stitch = _require_mapping(config.get("surface_stitch", {}), "surface_stitch")
    if str(stitch.get("solver", "subset")).strip().lower() != "subset":
        raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d stitching requires subset pairwise inputs", path="surface_stitch.solver")
    stitch["solver"] = "subset"
    config["surface_stitch"] = stitch
    config["subset_config"] = subset
    config.pop("solver", None)
    return config


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        raise ConfigResolutionError("MISSING_CONFIG", f"cannot load configuration {path}: {exc}") from exc


def _resolve_file(value: str | Path, root: Path) -> Path:
    return resolve_config_reference(value, root)


def _default_config_file(workflow_kind: str, root: Path) -> Path:
    candidate = root / _DEFAULT_FILES[workflow_kind]
    return candidate if candidate.exists() else default_config_path(_DEFAULT_FILES[workflow_kind])


def resolve_config(
    workflow_kind: str,
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    subset_config: Mapping[str, Any] | None = None,
    subset_config_path: str | Path | None = None,
    calibration_config: Mapping[str, Any] | None = None,
    calibration_config_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> ResolvedConfig:
    if workflow_kind not in CONFIG_WORKFLOW_KINDS:
        raise ConfigResolutionError("UNSUPPORTED_WORKFLOW", f"unsupported configuration workflow: {workflow_kind}")
    root = Path(repository_root or DEFAULT_REPOSITORY_ROOT).resolve()
    default_path = _default_config_file(workflow_kind, root)
    main_path = _resolve_file(config_path, root) if config_path is not None else default_path
    main = _load_mapping(main_path) if config is None else copy.deepcopy(dict(config))
    main_mapping_paths = set(_flatten_leaves(main)) if config is not None else set()
    _check_keys(main, workflow_kind)
    # A caller-supplied mapping is treated like a partial user configuration,
    # so it receives the same workflow YAML baseline as a custom config file.
    # This keeps precedence consistent across mapping, path, and CLI callers.
    default = _load_mapping(default_path) if config is not None or main_path.resolve() != default_path.resolve() else {}
    effective = _deep_merge(default, main)
    override_paths: list[str] = []
    nested_override_values: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        if not isinstance(key, str):
            raise ConfigResolutionError("INVALID_OVERRIDE_PATH", "override keys must be strings")
        alias = {"radius": "subset.radius", "search_radius": "initialization.integer_search.search_radius", "seed_count": "seed_selection.seed_count", "spacing": "reliability_propagation.spacing", "max_iterations": "optimization.max_iterations"}.get(key, key)
        if workflow_kind in {"stereo_3d", "multiview_3d"} and alias.startswith("subset."):
            nested_override_values[alias] = value
            override_paths.append(f"subset_config.{alias}")
            continue
        if workflow_kind == "stereo_3d" and alias == "solver":
            if str(value).strip().lower() != "subset":
                raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "stereo_3d only supports Subset-DIC", path=key)
            _deep_set(effective, "solver.method", "subset")
            override_paths.append("solver.method")
            continue
        if workflow_kind == "multiview_3d" and alias == "pairwise_2d_solver":
            if str(value).strip().lower() != "subset":
                raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", "multiview_3d only supports Subset-DIC", path=key)
            override_paths.append("pairwise_2d_solver")
            continue
        if workflow_kind in {"stereo_3d", "multiview_3d"} and alias in {"solver", "solver.method", "pairwise_solver", "pairwise_2d_solver"} and str(value).lower() != "subset":
            raise ConfigResolutionError("UNSUPPORTED_SOLVER_FOR_WORKFLOW", f"{workflow_kind} only supports Subset-DIC", path=key)
        _deep_set(effective, alias, value)
        override_paths.append(alias)
    _check_keys(effective, workflow_kind)
    source_files = [main_path]
    nested_subset: dict[str, Any] | None = None
    nested_calibration: dict[str, Any] | None = None
    subset_mapping_paths: set[str] = set()
    calibration_mapping_paths: set[str] = set()
    if workflow_kind in {"stereo_3d", "multiview_3d"}:
        configs = dict(effective.get("configs", {}) or {})
        subset_default_path = _default_config_file("subset_2d", root)
        subset_path = _resolve_file(subset_config_path or configs.get("subset", _DEFAULT_FILES["subset_2d"]), root)
        subset_raw = copy.deepcopy(dict(subset_config or _load_mapping(subset_path)))
        subset_mapping_paths = set(_flatten_leaves(subset_raw)) if subset_config is not None else set()
        for key, value in nested_override_values.items():
            _deep_set(subset_raw, key, value)
        nested_subset = _normalize_subset(_deep_merge(_load_mapping(subset_default_path) if subset_path.resolve() != subset_default_path.resolve() and subset_config is None else {}, subset_raw))
        source_files.append(subset_path)
        if workflow_kind == "stereo_3d":
            calibration_path = _resolve_file(calibration_config_path or configs.get("calibration", _DEFAULT_FILES["calibration"]), root)
            calibration_raw = copy.deepcopy(dict(calibration_config or _load_mapping(calibration_path)))
            calibration_mapping_paths = set(_flatten_leaves(calibration_raw)) if calibration_config is not None else set()
            nested_calibration = _normalize_calibration(calibration_raw)
            source_files.append(calibration_path)
    warnings: list[ConfigIssue] = []
    if workflow_kind == "subset_2d":
        effective = _normalize_subset(effective)
    elif workflow_kind == "mesh_2d":
        effective = _normalize_mesh(effective)
    elif workflow_kind == "calibration":
        effective = _normalize_calibration(effective)
    elif workflow_kind == "stereo_3d":
        effective = _normalize_stereo(effective, nested_subset or {}, nested_calibration or {}, warnings=warnings)
    else:
        canonical_main = config is None and main_path.resolve() == _default_config_file("multiview_3d", root).resolve()
        explicit_mesh_selection = any(
            path in {"pairwise_2d_dic.run_mesh", "pairwise_2d_dic.solver", "pairwise_2d_solver"}
            for path in override_paths
        )
        effective = _normalize_multiview(
            effective,
            nested_subset or {},
            warnings=warnings,
            allow_legacy_mesh=canonical_main and not explicit_mesh_selection,
        )
    leaves = _flatten_leaves(effective)
    provenance: dict[str, Any] = {}
    for path in leaves:
        if path in override_paths:
            source = f"override:{path}"
        elif path in main_mapping_paths:
            source = "mapping"
        elif path.startswith("subset_config.") and len(source_files) > 1:
            nested_path = path[len("subset_config."):]
            source = "mapping:subset_config" if nested_path in subset_mapping_paths else _portable_path(source_files[1], root)
        elif path.startswith("calibration_config.") and len(source_files) > 2:
            nested_path = path[len("calibration_config."):]
            source = "mapping:calibration_config" if nested_path in calibration_mapping_paths else _portable_path(source_files[2], root)
        else:
            source = _portable_path(main_path, root)
        provenance[path] = {"source": source}
    capabilities = {
        "subset_2d": True,
        "mesh_2d": True,
        "stereo_3d_solver": "subset",
        "multiview_3d_solver": "subset",
    }
    if workflow_kind == "subset_2d":
        capabilities = {"workflow": "subset_2d", "solver": "subset"}
    elif workflow_kind == "mesh_2d":
        capabilities = {"workflow": "mesh_2d", "solver": "mesh", "element_types": ["T3", "Q4", "Q8"]}
    elif workflow_kind == "calibration":
        capabilities = {"workflow": "calibration"}
    elif workflow_kind == "stereo_3d":
        capabilities = {"workflow": "stereo_3d", "correspondence_solver": "subset"}
    elif workflow_kind == "multiview_3d":
        capabilities = {"workflow": "multiview_3d", "pairwise_2d_solver": "subset", "pairwise_3d_solver": "subset"}
    return ResolvedConfig(workflow_kind, effective, tuple(dict.fromkeys(_portable_path(path, root) for path in source_files)), provenance, capabilities, tuple(warnings))


def inspect_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return {"valid": True, "config": resolve_config(*args, **kwargs).to_dict(repository_root=kwargs.get("repository_root"))}
    except ConfigResolutionError as exc:
        return {"valid": False, "errors": [{"code": exc.code, "message": str(exc), "path": exc.path}]}


def validate_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return inspect_config(*args, **kwargs)


__all__ = ["CONFIG_WORKFLOW_KINDS", "ConfigResolutionError", "ConfigIssue", "ResolvedConfig", "resolve_config", "inspect_config", "validate_config"]
