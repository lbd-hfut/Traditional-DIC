"""Stereo 3D-DIC assembly from precomputed 2D displacement fields.

The master coordinate is the left reference image (L0).  Three 2D fields are
required for every point:

* reference disparity: L0 -> R0
* left temporal displacement: L0 -> L1
* deformed cross-camera displacement: L0 -> R1

The deformed 3D point is then reconstructed from (L1, R1), not from a right
temporal DIC solve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

try:
    from . import _traditional_dic as _backend

    _has_backend = True
except ImportError:
    _has_backend = False


FIELD_REF_DISPARITY = "reference_disparity.csv"
FIELD_LEFT_TEMPORAL = "left_temporal.csv"
FIELD_DEFORMED_DISPARITY = "deformed_disparity.csv"


def _require_backend() -> None:
    if not _has_backend:
        raise ImportError(
            "C++ backend _traditional_dic not found. "
            "Build with -DTRADITIONAL_DIC_BUILD_PYTHON=ON and use the matching Python ABI."
        )


def _as_matrix3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    return arr


def _as_vector3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have 3 values")
    return arr


def camera_from_dict(data: Mapping) -> object:
    """Build a backend CameraModel from a plain dict/JSON object."""
    _require_backend()
    camera = _backend.calibration.CameraModel()
    camera.K = _as_matrix3(data.get("K", np.eye(3)), "K")
    camera.R = _as_matrix3(data.get("R", np.eye(3)), "R")
    camera.t = _as_vector3(data.get("t", [0.0, 0.0, 0.0]), "t")
    camera.distortion = [float(v) for v in data.get("distortion", [])]
    camera.image_width = int(data.get("image_width", 0))
    camera.image_height = int(data.get("image_height", 0))
    camera.rms_error = float(data.get("rms_error", 0.0))
    camera.label = str(data.get("label", ""))
    return camera


def load_camera_pair(path: str | Path) -> tuple[object, object, float]:
    """Load left/right CameraModel objects and world scale from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "left" not in data or "right" not in data:
        raise ValueError("camera JSON must contain 'left' and 'right' objects")
    return camera_from_dict(data["left"]), camera_from_dict(data["right"]), float(data.get("world_scale", 1.0))


def load_field_csv(path: str | Path) -> list[dict[str, float | int | bool]]:
    """Load a 2D displacement field CSV.

    Accepted columns are:
    x,y,u,v[,id|node_id|point_id][,correlation][,valid]

    If no id column exists, rows are matched by zero-based row index.
    """
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        names = set(reader.fieldnames)
        missing = {"x", "y", "u", "v"} - names
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        id_name = next((name for name in ("id", "node_id", "point_id", "track_id") if name in names), None)
        for row_index, row in enumerate(reader):
            valid_text = str(row.get("valid", "1")).strip().lower()
            valid = valid_text not in {"0", "false", "no", "nan", ""}
            point_id = int(float(row[id_name])) if id_name else row_index
            rows.append(
                {
                    "id": point_id,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "u": float(row["u"]),
                    "v": float(row["v"]),
                    "correlation": float(row.get("correlation", 1.0) or 1.0),
                    "valid": bool(valid),
                }
            )
    return rows


def _match_fields(*fields: Sequence[Mapping]) -> list[tuple[Mapping, ...]]:
    maps = [{int(row["id"]): row for row in field} for field in fields]
    ids = sorted(set(maps[0]).intersection(*[set(m) for m in maps[1:]]))
    return [tuple(m[point_id] for m in maps) for point_id in ids]


def _field_value_finite(row: Mapping) -> bool:
    values = (
        float(row["x"]),
        float(row["y"]),
        float(row["u"]),
        float(row["v"]),
        float(row.get("correlation", 1.0)),
    )
    return bool(np.all(np.isfinite(values)))


def _quality_metric_name(value: str) -> str:
    metric = str(value or "correlation").strip().lower()
    if metric in {"correlation", "zncc"}:
        return "correlation"
    if metric == "znssd":
        return "znssd"
    raise ValueError("quality_metric must be 'correlation' or 'znssd'")


def _quality_valid(
    ref_disp: Mapping,
    left_temp: Mapping,
    def_disp: Mapping,
    *,
    quality_metric: str,
    min_correlation: float,
    max_znssd: float,
) -> bool:
    ref_quality = float(ref_disp["correlation"])
    left_quality = float(left_temp["correlation"])
    def_quality = float(def_disp["correlation"])
    if quality_metric == "znssd":
        return max(ref_quality, left_quality, def_quality) <= max_znssd
    return min(ref_quality, left_quality, def_quality) >= min_correlation


def _matched_field_arrays(
    reference_disparity: Sequence[Mapping],
    left_temporal: Sequence[Mapping],
    deformed_disparity: Sequence[Mapping],
    *,
    quality_metric: str = "correlation",
    min_correlation: float = 0.6,
    max_znssd: float = 2.0,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    quality_metric = _quality_metric_name(quality_metric)
    matched = _match_fields(reference_disparity, left_temporal, deformed_disparity)
    xy = []
    ref = []
    left = []
    deformed = []
    valid = []
    corr_ref = []
    corr_left = []
    corr_def = []
    meta = []
    for ref_disp, left_temp, def_disp in matched:
        x = float(ref_disp["x"])
        y = float(ref_disp["y"])
        p_r0 = (x + float(ref_disp["u"]), y + float(ref_disp["v"]))
        p_l1 = (x + float(left_temp["u"]), y + float(left_temp["v"]))
        p_r1 = (x + float(def_disp["u"]), y + float(def_disp["v"]))
        row_valid = (
            bool(ref_disp["valid"] and left_temp["valid"] and def_disp["valid"])
            and _field_value_finite(ref_disp)
            and _field_value_finite(left_temp)
            and _field_value_finite(def_disp)
            and _quality_valid(
                ref_disp,
                left_temp,
                def_disp,
                quality_metric=quality_metric,
                min_correlation=min_correlation,
                max_znssd=max_znssd,
            )
        )
        xy.append((x, y))
        ref.append((float(ref_disp["u"]), float(ref_disp["v"])))
        left.append((float(left_temp["u"]), float(left_temp["v"])))
        deformed.append((float(def_disp["u"]), float(def_disp["v"])))
        valid.append(1 if row_valid else 0)
        corr_ref.append(float(ref_disp["correlation"]))
        corr_left.append(float(left_temp["correlation"]))
        corr_def.append(float(def_disp["correlation"]))
        meta.append(
            {
                "id": int(ref_disp["id"]),
                "x_l0": x,
                "y_l0": y,
                "x_r0": float(p_r0[0]),
                "y_r0": float(p_r0[1]),
                "x_l1": float(p_l1[0]),
                "y_l1": float(p_l1[1]),
                "x_r1": float(p_r1[0]),
                "y_r1": float(p_r1[1]),
                "corr_ref_disparity": float(ref_disp["correlation"]),
                "corr_left_temporal": float(left_temp["correlation"]),
                "corr_deformed_disparity": float(def_disp["correlation"]),
                "input_valid": row_valid,
            }
        )
    arrays = {
        "xy": np.asarray(xy, dtype=np.float64).reshape((-1, 2)),
        "reference_disparity": np.asarray(ref, dtype=np.float64).reshape((-1, 2)),
        "left_temporal": np.asarray(left, dtype=np.float64).reshape((-1, 2)),
        "deformed_disparity": np.asarray(deformed, dtype=np.float64).reshape((-1, 2)),
        "valid": np.asarray(valid, dtype=np.uint8),
        "corr_ref": np.asarray(corr_ref, dtype=np.float64),
        "corr_left_temporal": np.asarray(corr_left, dtype=np.float64),
        "corr_deformed_disparity": np.asarray(corr_def, dtype=np.float64),
    }
    if arrays["valid"].size:
        keep = arrays["valid"].astype(bool)
        arrays = {key: value[keep] for key, value in arrays.items()}
        meta = [m for m, is_valid in zip(meta, keep) if bool(is_valid)]
    return arrays, meta


def _make_observations(
    matched_rows: Sequence[tuple[Mapping, Mapping, Mapping]],
    *,
    quality_metric: str = "correlation",
    min_correlation: float = 0.6,
    max_znssd: float = 2.0,
) -> tuple[list, list, list[dict]]:
    _require_backend()
    quality_metric = _quality_metric_name(quality_metric)
    left_obs = []
    right_obs = []
    meta = []
    for ref_disp, left_temp, def_disp in matched_rows:
        point_id = int(ref_disp["id"])
        x = float(ref_disp["x"])
        y = float(ref_disp["y"])
        p_r0 = np.array([x + float(ref_disp["u"]), y + float(ref_disp["v"])], dtype=np.float64)
        p_l1 = np.array([x + float(left_temp["u"]), y + float(left_temp["v"])], dtype=np.float64)
        p_r1 = np.array([x + float(def_disp["u"]), y + float(def_disp["v"])], dtype=np.float64)

        valid = (
            bool(ref_disp["valid"] and left_temp["valid"] and def_disp["valid"])
            and _field_value_finite(ref_disp)
            and _field_value_finite(left_temp)
            and _field_value_finite(def_disp)
            and _quality_valid(
                ref_disp,
                left_temp,
                def_disp,
                quality_metric=quality_metric,
                min_correlation=min_correlation,
                max_znssd=max_znssd,
            )
        )
        corr_left = min(float(left_temp["correlation"]), float(ref_disp["correlation"]))
        corr_right = min(float(def_disp["correlation"]), float(ref_disp["correlation"]))

        left = _backend.reconstruction.PointObservation()
        left.camera_index = 0
        left.uv_ref = np.array([x, y], dtype=np.float64)
        left.uv_def = p_l1
        left.u_displacement = float(p_l1[0] - x)
        left.v_displacement = float(p_l1[1] - y)
        left.correlation = corr_left
        left.dic_valid = valid

        right = _backend.reconstruction.PointObservation()
        right.camera_index = 1
        right.uv_ref = p_r0
        right.uv_def = p_r1
        right.u_displacement = float(p_r1[0] - p_r0[0])
        right.v_displacement = float(p_r1[1] - p_r0[1])
        right.correlation = corr_right
        right.dic_valid = valid

        left_obs.append(left)
        right_obs.append(right)
        meta.append(
            {
                "id": point_id,
                "x_l0": x,
                "y_l0": y,
                "x_r0": float(p_r0[0]),
                "y_r0": float(p_r0[1]),
                "x_l1": float(p_l1[0]),
                "y_l1": float(p_l1[1]),
                "x_r1": float(p_r1[0]),
                "y_r1": float(p_r1[1]),
                "corr_ref_disparity": float(ref_disp["correlation"]),
                "corr_left_temporal": float(left_temp["correlation"]),
                "corr_deformed_disparity": float(def_disp["correlation"]),
                "input_valid": valid,
            }
        )
    return left_obs, right_obs, meta


def reconstruct_from_fields(
    reference_disparity: Sequence[Mapping],
    left_temporal: Sequence[Mapping],
    deformed_disparity: Sequence[Mapping],
    left_camera,
    right_camera,
    *,
    min_correlation: float = 0.6,
    quality_metric: str = "correlation",
    max_znssd: float = 2.0,
    max_reprojection_error_px: float = 2.0,
    world_scale: float = 1.0,
    remove_rigid_body_motion: bool = False,
) -> tuple[object, list[dict]]:
    """Reconstruct stereo 3D-DIC points from already computed 2D fields."""
    _require_backend()
    quality_metric = _quality_metric_name(quality_metric)
    opts = _backend.reconstruction.StereoDICOptions()
    opts.min_correlation = float(min_correlation) if quality_metric == "correlation" else -float("inf")
    opts.max_reprojection_error_px = float(max_reprojection_error_px)
    opts.world_scale = float(world_scale)
    opts.remove_rigid_body_motion = bool(remove_rigid_body_motion)

    if hasattr(_backend.reconstruction, "reconstruct_stereo_fields"):
        arrays, meta = _matched_field_arrays(
            reference_disparity,
            left_temporal,
            deformed_disparity,
            quality_metric=quality_metric,
            min_correlation=float(min_correlation),
            max_znssd=float(max_znssd),
        )
        result = _backend.reconstruction.reconstruct_stereo_fields(
            arrays["xy"],
            arrays["reference_disparity"],
            arrays["left_temporal"],
            arrays["deformed_disparity"],
            arrays["valid"],
            arrays["corr_ref"],
            arrays["corr_left_temporal"],
            arrays["corr_deformed_disparity"],
            left_camera,
            right_camera,
            opts,
        )
        return result, meta

    matched = _match_fields(reference_disparity, left_temporal, deformed_disparity)
    left_obs, right_obs, meta = _make_observations(
        matched,
        quality_metric=quality_metric,
        min_correlation=float(min_correlation),
        max_znssd=float(max_znssd),
    )

    result = _backend.reconstruction.StereoDIC(opts).reconstruct(left_obs, right_obs, left_camera, right_camera)
    return result, meta


def reconstruct_from_field_files(
    field_dir: str | Path,
    left_camera,
    right_camera,
    *,
    out_dir: Optional[str | Path] = None,
    deformation_out_dir: Optional[str | Path] = None,
    visualization_out_dir: Optional[str | Path] = None,
    deformation_visualization_out_dir: Optional[str | Path] = None,
    faces: Optional[Sequence[Sequence[int]]] = None,
    output_prefix: str = "",
    write_shape_maps: bool = True,
    write_deformation_maps: bool = True,
    write_surface_strain: bool = True,
    min_correlation: float = 0.6,
    quality_metric: str = "correlation",
    max_znssd: float = 2.0,
    max_reprojection_error_px: float = 2.0,
    world_scale: float = 1.0,
    remove_rigid_body_motion: bool = False,
    reference_field: str = FIELD_REF_DISPARITY,
    left_temporal_field: str = FIELD_LEFT_TEMPORAL,
    deformed_field: str = FIELD_DEFORMED_DISPARITY,
) -> object:
    """Read standard field files from a directory, reconstruct, and write 3D CSVs."""
    field_dir = Path(field_dir)
    out_dir = Path(out_dir) if out_dir is not None else field_dir
    visualization_out_dir = Path(visualization_out_dir) if visualization_out_dir is not None else out_dir
    result, meta = reconstruct_from_fields(
        load_field_csv(field_dir / reference_field),
        load_field_csv(field_dir / left_temporal_field),
        load_field_csv(field_dir / deformed_field),
        left_camera,
        right_camera,
        min_correlation=min_correlation,
        quality_metric=quality_metric,
        max_znssd=max_znssd,
        max_reprojection_error_px=max_reprojection_error_px,
        world_scale=world_scale,
        remove_rigid_body_motion=remove_rigid_body_motion,
    )
    save_stereo_points_csv(result, meta, out_dir / f"{output_prefix}stereo_3d_points.csv")
    save_summary_json(result, out_dir / f"{output_prefix}stereo_3d_summary.json")
    if write_shape_maps:
        save_shape_visualizations(
            result,
            meta,
            visualization_out_dir,
            int(getattr(left_camera, "image_width", 0)),
            int(getattr(left_camera, "image_height", 0)),
            prefix=output_prefix,
        )
    if deformation_out_dir is not None:
        deformation_out_dir = Path(deformation_out_dir)
        deformation_visualization_out_dir = (
            Path(deformation_visualization_out_dir)
            if deformation_visualization_out_dir is not None
            else deformation_out_dir
        )
        save_deformation_csv(result, meta, deformation_out_dir / f"{output_prefix}deformation_3d.csv")
        save_summary_json(result, deformation_out_dir / f"{output_prefix}deformation_3d_summary.json")
        if write_deformation_maps:
            save_deformation_visualizations(
                result,
                meta,
                deformation_visualization_out_dir,
                int(getattr(left_camera, "image_width", 0)),
                int(getattr(left_camera, "image_height", 0)),
                prefix=output_prefix,
            )
    if faces is not None and write_surface_strain:
        save_surface_strain_csv(result, faces, out_dir / f"{output_prefix}stereo_3d_strain_faces.csv")
    return result


def _result_arrays(result) -> dict[str, np.ndarray]:
    if hasattr(_backend.reconstruction, "stereo_result_arrays"):
        return {key: np.asarray(value) for key, value in _backend.reconstruction.stereo_result_arrays(result).items()}

    point_ref = []
    point_def = []
    displacement = []
    mag = []
    err_ref = []
    err_def = []
    corr = []
    valid = []
    for point in result.points:
        point_ref.append(np.asarray(point.point_ref_world, dtype=np.float64))
        point_def.append(np.asarray(point.point_def_world, dtype=np.float64))
        displacement.append(np.asarray(point.displacement_world, dtype=np.float64))
        mag.append(float(point.displacement_norm_world))
        err_ref.append(float(point.reprojection_error_ref))
        err_def.append(float(point.reprojection_error_def))
        corr.append(float(point.combined_correlation))
        valid.append(1 if bool(point.valid) else 0)
    return {
        "point_ref_world": np.asarray(point_ref, dtype=np.float64).reshape((-1, 3)),
        "point_def_world": np.asarray(point_def, dtype=np.float64).reshape((-1, 3)),
        "displacement_world": np.asarray(displacement, dtype=np.float64).reshape((-1, 3)),
        "displacement_norm_world": np.asarray(mag, dtype=np.float64),
        "reprojection_error_ref": np.asarray(err_ref, dtype=np.float64),
        "reprojection_error_def": np.asarray(err_def, dtype=np.float64),
        "combined_correlation": np.asarray(corr, dtype=np.float64),
        "valid": np.asarray(valid, dtype=np.uint8),
    }


def save_stereo_points_csv(result, meta: Sequence[Mapping], path: str | Path) -> None:
    """Write reference/deformed 3D coordinates and 3D displacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _result_arrays(result)
    point_ref = arrays["point_ref_world"]
    point_def = arrays["point_def_world"]
    displacement = arrays["displacement_world"]
    mag = arrays["displacement_norm_world"]
    err_ref = arrays["reprojection_error_ref"]
    err_def = arrays["reprojection_error_def"]
    corr = arrays["combined_correlation"]
    valid = arrays["valid"].astype(bool)
    header = [
        "id",
        "x_l0",
        "y_l0",
        "x_r0",
        "y_r0",
        "x_l1",
        "y_l1",
        "x_r1",
        "y_r1",
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
        "combined_correlation",
        "input_valid",
        "valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, m in enumerate(meta):
            m = meta[i]
            row = [
                m["id"],
                m["x_l0"],
                m["y_l0"],
                m["x_r0"],
                m["y_r0"],
                m["x_l1"],
                m["y_l1"],
                m["x_r1"],
                m["y_r1"],
                point_ref[i, 0],
                point_ref[i, 1],
                point_ref[i, 2],
                point_def[i, 0],
                point_def[i, 1],
                point_def[i, 2],
                displacement[i, 0],
                displacement[i, 1],
                displacement[i, 2],
                mag[i],
                err_ref[i],
                err_def[i],
                corr[i],
                int(bool(m["input_valid"])),
                int(bool(valid[i])),
            ]
            writer.writerow(row)


def save_summary_json(result, path: str | Path) -> None:
    arrays = _result_arrays(result)
    valid_mask = arrays["valid"].astype(bool)
    displacement = arrays["displacement_world"][valid_mask]
    ux = displacement[:, 0] if displacement.size else np.asarray([], dtype=np.float64)
    uy = displacement[:, 1] if displacement.size else np.asarray([], dtype=np.float64)
    uz = displacement[:, 2] if displacement.size else np.asarray([], dtype=np.float64)
    mag = arrays["displacement_norm_world"][valid_mask]
    summary = {
        "total_points": int(result.total_points),
        "valid_points": int(result.valid_points),
        "invalid_points": int(result.total_points - result.valid_points),
        "world_scale": float(result.world_scale),
        "mean_displacement_norm": float(result.mean_displacement_norm),
        "Ux_mean": float(ux.mean()) if ux.size else 0.0,
        "Uy_mean": float(uy.mean()) if uy.size else 0.0,
        "Uz_mean": float(uz.mean()) if uz.size else 0.0,
        "Umag_max": float(mag.max()) if mag.size else 0.0,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_deformation_csv(result, meta: Sequence[Mapping], path: str | Path) -> None:
    """Write the 3D displacement field in left-reference image coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _result_arrays(result)
    displacement = arrays["displacement_world"]
    mag = arrays["displacement_norm_world"]
    err_ref = arrays["reprojection_error_ref"]
    err_def = arrays["reprojection_error_def"]
    corr = arrays["combined_correlation"]
    valid = arrays["valid"].astype(bool)
    header = [
        "id",
        "x_l0",
        "y_l0",
        "x_l1",
        "y_l1",
        "Ux",
        "Uy",
        "Uz",
        "Umag",
        "reprojection_error_ref",
        "reprojection_error_def",
        "combined_correlation",
        "input_valid",
        "valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, m in enumerate(meta):
            writer.writerow(
                [
                    m["id"],
                    m["x_l0"],
                    m["y_l0"],
                    m["x_l1"],
                    m["y_l1"],
                    displacement[i, 0],
                    displacement[i, 1],
                    displacement[i, 2],
                    mag[i],
                    err_ref[i],
                    err_def[i],
                    corr[i],
                    int(bool(m["input_valid"])),
                    int(bool(valid[i])),
                ]
            )


def _color_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = np.zeros_like(values, dtype=np.float64)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _draw_colorbar(image, x0: int, y0: int, height: int, vmin: float, vmax: float, label: str) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
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


def _save_shape_image(
    path: Path,
    xy: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
    title: str,
    scalar_label: str = "Z",
) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    if width <= 0 or height <= 0:
        width = max(1, int(np.ceil(np.max(xy[:, 0]))) + 20) if xy.size else 640
        height = max(1, int(np.ceil(np.max(xy[:, 1]))) + 20) if xy.size else 480

    valid_values = values[np.isfinite(values)]
    if valid_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(valid_values)), float(np.max(valid_values))
    colors = np.asarray(np.clip(_color_map(values, float(vmin), float(vmax)) * 255.0, 0, 255), dtype=np.uint8)

    canvas = np.full((height, width + 120, 3), 255, dtype=np.uint8)
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & np.all(np.isfinite(colors), axis=1)
    cx = np.rint(xy[finite, 0]).astype(np.int32)
    cy = np.rint(xy[finite, 1]).astype(np.int32)
    point_colors = colors[finite]
    inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
    cx = cx[inside]
    cy = cy[inside]
    point_colors = point_colors[inside]

    radius = 2 if cx.size < 20000 else 1
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    for dx, dy in offsets:
        xx = cx + dx
        yy = cy + dy
        ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        canvas[yy[ok], xx[ok], :] = point_colors[ok]

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((14, 12), f"{title} {scalar_label} min={vmin:.4g} max={vmax:.4g}", fill=(20, 20, 20))
    _draw_colorbar(image, width + 22, 42, max(80, height - 96), vmin, vmax, scalar_label)
    image.save(path)


def save_shape_visualizations(
    result,
    meta: Sequence[Mapping],
    out_dir: str | Path,
    image_width: int = 0,
    image_height: int = 0,
    prefix: str = "",
) -> None:
    """Write simple Z-colored reference/deformed shape maps in left image coordinates."""
    arrays = _result_arrays(result)
    valid_mask = arrays["valid"].astype(bool)
    if not np.any(valid_mask):
        return
    meta_xy_ref = np.asarray([[m["x_l0"], m["y_l0"]] for m in meta], dtype=np.float64)
    meta_xy_def = np.asarray([[m["x_l1"], m["y_l1"]] for m in meta], dtype=np.float64)
    xy_ref = meta_xy_ref[valid_mask]
    xy_def = meta_xy_def[valid_mask]
    z_ref = arrays["point_ref_world"][valid_mask, 2]
    z_def = arrays["point_def_world"][valid_mask, 2]
    out_dir = Path(out_dir)
    _save_shape_image(out_dir / f"{prefix}shape_ref_z.png", xy_ref, z_ref, image_width, image_height, "Reference shape")
    _save_shape_image(out_dir / f"{prefix}shape_def_z.png", xy_def, z_def, image_width, image_height, "Deformed shape")


def save_deformation_visualizations(
    result,
    meta: Sequence[Mapping],
    out_dir: str | Path,
    image_width: int = 0,
    image_height: int = 0,
    prefix: str = "",
) -> None:
    """Write Ux/Uy/Uz/Umag maps in left reference image coordinates."""
    arrays = _result_arrays(result)
    valid_mask = arrays["valid"].astype(bool)
    if not np.any(valid_mask):
        return
    xy = np.asarray([[m["x_l0"], m["y_l0"]] for m in meta], dtype=np.float64)[valid_mask]
    displacement = arrays["displacement_world"][valid_mask]
    magnitude = arrays["displacement_norm_world"][valid_mask]
    out_dir = Path(out_dir)
    _save_shape_image(out_dir / f"{prefix}deformation_ux.png", xy, displacement[:, 0], image_width, image_height, "3D displacement", "Ux")
    _save_shape_image(out_dir / f"{prefix}deformation_uy.png", xy, displacement[:, 1], image_width, image_height, "3D displacement", "Uy")
    _save_shape_image(out_dir / f"{prefix}deformation_uz.png", xy, displacement[:, 2], image_width, image_height, "3D displacement", "Uz")
    _save_shape_image(out_dir / f"{prefix}deformation_umag.png", xy, magnitude, image_width, image_height, "3D displacement", "Umag")


def triangulate_faces_from_elements(elements: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    """Convert T3/Q4/Q8 element connectivity to triangular faces."""
    faces: list[tuple[int, int, int]] = []
    for elem in elements:
        ids = [int(v) for v in elem]
        if len(ids) == 3:
            faces.append((ids[0], ids[1], ids[2]))
        elif len(ids) >= 4:
            faces.append((ids[0], ids[1], ids[2]))
            faces.append((ids[0], ids[2], ids[3]))
        else:
            raise ValueError("Each element must contain at least 3 node indices")
    return faces


def save_surface_strain_csv(result, elements: Sequence[Sequence[int]], path: str | Path) -> None:
    """Compute and write per-triangle 3D surface strain for mesh results."""
    _require_backend()
    faces = triangulate_faces_from_elements(elements)
    points_ref = [p.point_ref_world for p in result.points]
    points_def = [p.point_def_world for p in result.points]
    valid_points = [bool(p.valid) for p in result.points]
    valid_faces = [all(valid_points[idx] for idx in face) for face in faces]
    strains = _backend.postprocess.compute_surface_strain(faces, points_ref, points_def, valid_faces)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "face_id",
                "n1",
                "n2",
                "n3",
                "area",
                "Epc1",
                "Epc2",
                "Eeq",
                "EShearMax",
                "epc1",
                "epc2",
                "eeq",
                "valid",
            ]
        )
        for i, (face, strain) in enumerate(zip(faces, strains), start=1):
            writer.writerow(
                [
                    i,
                    face[0] + 1,
                    face[1] + 1,
                    face[2] + 1,
                    strain.area,
                    strain.Epc1,
                    strain.Epc2,
                    strain.Eeq,
                    strain.EShearMax,
                    strain.epc1,
                    strain.epc2,
                    strain.eeq,
                    int(bool(strain.valid)),
                ]
            )


def stereo(*args, **kwargs):
    """Compatibility entry point for stereo 3D-DIC.

    New code should call :func:`reconstruct_from_fields` or
    :func:`reconstruct_from_field_files` explicitly.
    """
    if "field_dir" in kwargs:
        return reconstruct_from_field_files(**kwargs)
    raise NotImplementedError(
        "Use reconstruct_from_fields/reconstruct_from_field_files with precomputed "
        "L0->R0, L0->L1, and L0->R1 fields."
    )
