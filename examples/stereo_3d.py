"""Run stereo 3D-DIC from YAML configuration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import traditional_dic as tdic  # noqa: E402
from traditional_dic import calibration as calib  # noqa: E402
from traditional_dic.config import (  # noqa: E402
    load_config,
    mesh_generation_config,
    normalize_mesh_config,
    normalize_subset_config,
)
from traditional_dic.stereo import (  # noqa: E402
    FIELD_DEFORMED_DISPARITY,
    FIELD_LEFT_TEMPORAL,
    FIELD_REF_DISPARITY,
    load_camera_pair,
    reconstruct_from_field_files,
)


FIELD_DEFS = [
    (FIELD_REF_DISPARITY, "right_reference", "Reference disparity L0->R0"),
    (FIELD_LEFT_TEMPORAL, "left_deformed", "Temporal field L0->Llast"),
    (FIELD_DEFORMED_DISPARITY, "right_deformed", "Deformed stereo field L0->Rlast"),
]


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def case_path(case_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else case_root / p


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


def run_calibration(paths: dict[str, Path], calibration_cfg: dict[str, Any], out_dir: Path) -> Path:
    left_paths = sorted((paths["case_root"] / "calibrate1").glob("*.bmp"))
    right_paths = sorted((paths["case_root"] / "calibrate2").glob("*.bmp"))
    if not left_paths or len(left_paths) != len(right_paths):
        raise RuntimeError("calibrate1/calibrate2 must contain the same nonzero number of BMP images")

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


def compute_subset_fields(paths: dict[str, Path], subset_cfg: dict[str, Any], disp_dir: Path) -> None:
    reference = read_gray(paths["left_reference"])
    roi = read_mask(paths["roi"])
    height, width = reference.shape
    config = normalize_subset_config(subset_cfg)
    disp_dir.mkdir(parents=True, exist_ok=True)

    for field_name, image_key, title in FIELD_DEFS:
        print(f"Computing subset {title}")
        deformed = read_gray(paths[image_key])
        result = tdic.subset(reference, deformed, config=config, roi=roi)
        valid = np.asarray(result["valid"], dtype=bool)
        xy_all = np.column_stack([np.asarray(result["x"], dtype=np.float64), np.asarray(result["y"], dtype=np.float64)])
        uv_all = np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])
        write_result_field(disp_dir / field_name, xy_all, uv_all, result["correlation"], valid)
        render_field_components(disp_dir, Path(field_name).stem, xy_all[valid], uv_all[valid], width, height, title)


def write_nodes(path: Path, nodes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# node_id,x,y\n")
        for i, (x, y) in enumerate(nodes, start=1):
            f.write(f"{i},{x:.12g},{y:.12g}\n")


def write_elements(path: Path, elements: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# element_id,node_ids...\n")
        for i, elem in enumerate(elements, start=1):
            f.write(f"{i},{','.join(str(int(v) + 1) for v in elem)}\n")


def read_elements(path: Path) -> list[list[int]]:
    elements = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            elements.append([int(v) - 1 for v in parts[1:]])
    return elements


def grid_from_roi(roi_path: Path, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    roi = np.asarray(Image.open(roi_path).convert("L")) > 0
    height, width = roi.shape
    ys, xs = np.nonzero(roi)
    if xs.size == 0:
        raise ValueError(f"{roi_path} has no nonzero ROI pixels")
    x_values = np.linspace(float(xs.min()), float(xs.max()), nx)
    y_values = np.linspace(float(ys.min()), float(ys.max()), ny)
    nodes = np.asarray([(x, y) for y in y_values for x in x_values], dtype=np.float64)

    q4 = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
            pixel_corners = [(int(round(nodes[k, 0])), int(round(nodes[k, 1]))) for k in corners]
            if all(0 <= x < width and 0 <= y < height and roi[y, x] for x, y in pixel_corners):
                q4.append(corners)
    q4 = np.asarray(q4, dtype=np.int64)
    t3 = np.asarray([[a, b, c] for a, b, c, d in q4] + [[a, c, d] for a, b, c, d in q4], dtype=np.int64)
    return nodes, q4, t3, width, height


def q8_from_q4(nodes: np.ndarray, q4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_nodes = nodes.tolist()
    edge_midpoints: dict[tuple[int, int], int] = {}
    q8 = []
    for a, b, c, d in q4:
        mids = []
        for i, j in ((a, b), (b, c), (c, d), (d, a)):
            key = tuple(sorted((int(i), int(j))))
            if key not in edge_midpoints:
                edge_midpoints[key] = len(out_nodes)
                out_nodes.append(((nodes[i] + nodes[j]) * 0.5).tolist())
            mids.append(edge_midpoints[key])
        q8.append([a, b, c, d, mids[0], mids[1], mids[2], mids[3]])
    return np.asarray(out_nodes, dtype=np.float64), np.asarray(q8, dtype=np.int64)


def draw_mesh(path: Path, roi_path: Path, nodes: np.ndarray, elements: np.ndarray, etype: str) -> None:
    roi = Image.open(roi_path).convert("L")
    canvas = Image.new("RGB", roi.size, "white")
    mask = np.asarray(roi) > 0
    arr = np.asarray(canvas).copy()
    arr[mask] = np.array([238, 242, 246], dtype=np.uint8)
    canvas = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for elem in elements:
        pts = [tuple(np.round(nodes[idx]).astype(int)) for idx in (elem[:4] if etype in {"Q4", "Q8"} else elem)]
        draw.line(pts + [pts[0]], fill=(42, 83, 140), width=1)
    for x, y in nodes:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(214, 40, 40))
    draw.text((14, 12), f"{etype} mesh: {len(nodes)} nodes, {len(elements)} elements", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def shape_t3(xi: float, eta: float) -> np.ndarray:
    return np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)


def shape_q4(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )


def shape_q8(xi: float, eta: float) -> np.ndarray:
    return np.array(
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


def t3_natural(point: np.ndarray, tri: np.ndarray) -> tuple[float, float] | None:
    a, b, c = tri
    try:
        xi_eta = np.linalg.solve(np.column_stack([b - a, c - a]), point - a)
    except np.linalg.LinAlgError:
        return None
    xi, eta = float(xi_eta[0]), float(xi_eta[1])
    if xi >= -1.0e-8 and eta >= -1.0e-8 and xi + eta <= 1.0 + 1.0e-8:
        return xi, eta
    return None


def natural_q4(point: np.ndarray, corners: np.ndarray) -> tuple[float, float] | None:
    xi, eta = 0.0, 0.0
    for _ in range(12):
        n = shape_q4(xi, eta)
        residual = n @ corners - point
        if np.linalg.norm(residual) < 1.0e-8:
            return xi, eta
        dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
        try:
            delta = np.linalg.solve(np.column_stack([dxi @ corners, deta @ corners]), residual)
        except np.linalg.LinAlgError:
            return None
        xi -= float(delta[0])
        eta -= float(delta[1])
    if -1.0 - 1.0e-8 <= xi <= 1.0 + 1.0e-8 and -1.0 - 1.0e-8 <= eta <= 1.0 + 1.0e-8:
        return xi, eta
    return None


def dense_mesh_samples(nodes: np.ndarray, elements: np.ndarray, etype: str, width: int, height: int):
    samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for elem_id, elem in enumerate(elements, start=1):
        xy = nodes[elem]
        corners = xy[:4] if etype in {"Q4", "Q8"} else xy
        xmin = max(0, int(np.floor(np.min(corners[:, 0]))))
        xmax = min(width - 1, int(np.ceil(np.max(corners[:, 0]))))
        ymin = max(0, int(np.floor(np.min(corners[:, 1]))))
        ymax = min(height - 1, int(np.ceil(np.max(corners[:, 1]))))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                point = np.array([float(x), float(y)], dtype=np.float64)
                natural = t3_natural(point, corners) if etype == "T3" else natural_q4(point, corners)
                if natural is None:
                    continue
                xi, eta = natural
                shape = shape_t3(xi, eta) if etype == "T3" else shape_q4(xi, eta) if etype == "Q4" else shape_q8(xi, eta)
                samples[(x, y)] = (np.asarray(elem, dtype=np.int64), shape, elem_id)
    return samples


def write_dense_field(path: Path, samples: dict, nodal_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(samples.items(), key=lambda item: (item[0][1], item[0][0]))
    xy = np.asarray([key for key, _ in items], dtype=np.float64)
    uv = np.asarray([shape @ nodal_uv[elem] for _, (elem, shape, _) in items], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "u", "v", "element_id", "correlation", "valid"])
        for i, ((x, y), (u, v), (_, _, elem_id)) in enumerate(zip(xy, uv, [item[1] for item in items]), start=1):
            writer.writerow([i, x, y, float(u), float(v), elem_id, 1.0, 1])
    return xy, uv


def mesh_data_by_type(paths: dict[str, Path], mesh_cfg: dict[str, Any]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int, int]:
    generation = mesh_generation_config(mesh_cfg)
    roi = read_mask(paths["roi"])
    height, width = roi.shape
    ys, xs = np.nonzero(roi > 0)
    if xs.size == 0:
        raise ValueError(f"{paths['roi']} has no nonzero ROI pixels")
    target = float(generation.get("target_element_size", 35.0))
    nx = max(3, int(np.ceil((float(xs.max()) - float(xs.min())) / target)) + 1)
    ny = max(3, int(np.ceil((float(ys.max()) - float(ys.min())) / target)) + 1)
    base_nodes, q4, t3, _, _ = grid_from_roi(paths["roi"], nx, ny)
    data = {"T3": (base_nodes, t3), "Q4": (base_nodes, q4)}
    data["Q8"] = q8_from_q4(base_nodes, q4)
    return data, width, height


def compute_mesh_fields(paths: dict[str, Path], mesh_cfg: dict[str, Any], disp_root: Path, element_types: list[str]) -> None:
    reference = read_gray(paths["left_reference"])
    images = {key: read_gray(path) for _, key, _ in FIELD_DEFS for path in [paths[key]]}
    config = normalize_mesh_config(mesh_cfg)
    data_by_type, width, height = mesh_data_by_type(paths, mesh_cfg)

    for etype in element_types:
        disp_dir = disp_root / etype
        mesh_dir = disp_dir / "meshGen"
        nodes, elements = data_by_type[etype]
        write_nodes(mesh_dir / f"nodes_{etype}.txt", nodes)
        write_elements(mesh_dir / f"elements_{etype}.txt", elements)
        draw_mesh(mesh_dir / f"mesh_{etype}.png", paths["roi"], nodes, elements, etype)
        samples = dense_mesh_samples(nodes, elements, etype, width, height)

        for field_name, image_key, title in FIELD_DEFS:
            print(f"Computing mesh {etype} {title}")
            result = tdic.mesh(reference, images[image_key], nodes, elements, element_type=etype, config=config)
            uv = np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])
            corr = np.asarray(result.get("correlation", np.ones(len(uv))), dtype=np.float64)
            valid = np.asarray(result.get("valid", np.ones(len(uv), dtype=bool)), dtype=bool)
            write_result_field(disp_dir / field_name, nodes, uv, corr, valid)
            stem = Path(field_name).stem
            dense_xy, dense_uv = write_dense_field(disp_dir / f"{stem}_dense.csv", samples, uv)
            render_scalar_field(
                disp_dir / f"{stem}_dense_mag.png",
                dense_xy,
                np.linalg.norm(dense_uv, axis=1),
                width,
                height,
                f"{etype} dense {title}",
                "|d|",
            )


def reconstruct_subset(paths: dict[str, Path], output_dirs: dict[str, Path], recon_cfg: dict[str, Any], camera_path: Path) -> None:
    left_camera, right_camera, world_scale = load_camera_pair(camera_path)
    result = reconstruct_from_field_files(
        output_dirs["disp"] / "subset",
        left_camera,
        right_camera,
        out_dir=output_dirs["reconstruct"] / "subset",
        deformation_out_dir=output_dirs["deformation"] / "subset",
        min_correlation=float(recon_cfg.get("min_correlation", 0.0)),
        quality_metric=str(recon_cfg.get("quality_metric", "correlation")),
        max_znssd=float(recon_cfg.get("max_znssd", 2.0)),
        max_reprojection_error_px=float(recon_cfg.get("max_reprojection_error_px", 5.0)),
        world_scale=world_scale,
        remove_rigid_body_motion=bool(recon_cfg.get("remove_rigid_body_motion", False)),
    )
    print(f"Reconstructed subset {result.valid_points}/{result.total_points} valid")


def reconstruct_mesh(output_dirs: dict[str, Path], recon_cfg: dict[str, Any], camera_path: Path, element_types: list[str]) -> None:
    left_camera, right_camera, world_scale = load_camera_pair(camera_path)
    for etype in element_types:
        disp_dir = output_dirs["disp"] / "mesh" / etype
        elements_path = disp_dir / "meshGen" / f"elements_{etype}.txt"
        if not elements_path.exists():
            elements_path = disp_dir / f"elements_{etype}.txt"
        elements = read_elements(elements_path)
        result = reconstruct_from_field_files(
            disp_dir,
            left_camera,
            right_camera,
            out_dir=output_dirs["reconstruct"] / "mesh" / etype,
            deformation_out_dir=output_dirs["deformation"] / "mesh" / etype,
            faces=elements,
            write_shape_maps=False,
            write_deformation_maps=False,
            min_correlation=float(recon_cfg.get("min_correlation", 0.0)),
            quality_metric=str(recon_cfg.get("quality_metric", "correlation")),
            max_znssd=float(recon_cfg.get("max_znssd", 2.0)),
            max_reprojection_error_px=float(recon_cfg.get("max_reprojection_error_px", 5.0)),
            world_scale=world_scale,
            remove_rigid_body_motion=bool(recon_cfg.get("remove_rigid_body_motion", False)),
        )
        reconstruct_from_field_files(
            disp_dir,
            left_camera,
            right_camera,
            out_dir=output_dirs["reconstruct"] / "mesh" / etype,
            deformation_out_dir=output_dirs["deformation"] / "mesh" / etype,
            output_prefix="dense_",
            write_surface_strain=False,
            min_correlation=float(recon_cfg.get("min_correlation", 0.0)),
            quality_metric=str(recon_cfg.get("quality_metric", "correlation")),
            max_znssd=float(recon_cfg.get("max_znssd", 2.0)),
            max_reprojection_error_px=float(recon_cfg.get("max_reprojection_error_px", 5.0)),
            world_scale=world_scale,
            remove_rigid_body_motion=bool(recon_cfg.get("remove_rigid_body_motion", False)),
            reference_field="reference_disparity_dense.csv",
            left_temporal_field="left_temporal_dense.csv",
            deformed_field="deformed_disparity_dense.csv",
        )
        print(f"Reconstructed mesh {etype} {result.valid_points}/{result.total_points} valid")


def build_paths(stereo_cfg: dict[str, Any]) -> dict[str, Path]:
    case_cfg = stereo_cfg.get("case", {})
    case_root = resolve_path(case_cfg.get("root", "case/stereo_DIC/plate_center_load"))
    return {
        "case_root": case_root,
        "left_reference": case_path(case_root, case_cfg.get("left_reference", "cam1/00_L.bmp")),
        "right_reference": case_path(case_root, case_cfg.get("right_reference", "cam2/00_R.bmp")),
        "left_deformed": case_path(case_root, case_cfg.get("left_deformed", "cam1/04_L.bmp")),
        "right_deformed": case_path(case_root, case_cfg.get("right_deformed", "cam2/04_R.bmp")),
        "roi": case_path(case_root, case_cfg.get("roi", "ROI.bmp")),
    }


def build_output_dirs(paths: dict[str, Path], stereo_cfg: dict[str, Any]) -> dict[str, Path]:
    output = stereo_cfg.get("output", {})
    root = case_path(paths["case_root"], output.get("root", "result"))
    return {
        "root": root,
        "calibration": root / output.get("calibration", "calibration"),
        "disp": root / output.get("disp", "disp"),
        "reconstruct": root / output.get("reconstruct", "reconstruct"),
        "deformation": root / output.get("deformation", "deformation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-config", type=Path, default=PROJECT_ROOT / "config" / "stereo_3d.yaml")
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--solver", choices=["subset", "mesh"])
    parser.add_argument("--element", choices=["T3", "Q4", "Q8", "all"])
    parser.add_argument("--compute-fields", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    stereo_cfg = load_config(args.stereo_config)
    paths = build_paths(stereo_cfg)
    output_dirs = build_output_dirs(paths, stereo_cfg)
    calibration_cfg_path = args.calibration_config or resolve_path(
        stereo_cfg.get("configs", {}).get("calibration", "config/calibration.yaml")
    )
    calibration_cfg = load_config(calibration_cfg_path)

    workflow = stereo_cfg.get("workflow", {})
    solver_cfg = stereo_cfg.get("solver", {})
    solver = args.solver or str(solver_cfg.get("method", "subset")).lower()
    compute_fields = bool(args.compute_fields or workflow.get("compute_fields", False))
    calibrate = bool(workflow.get("calibrate", True)) and not args.skip_calibration
    reconstruct = bool(workflow.get("reconstruct", True))
    camera_path = output_dirs["calibration"] / "camera_pair.json"

    if calibrate:
        camera_path = run_calibration(paths, calibration_cfg, output_dirs["calibration"])
        if workflow.get("visualize_calibration", False):
            from case.visualize_calibration_results import visualize_saved_stereo_calibration_result

            visualize_saved_stereo_calibration_result(output_dirs["calibration"] / "stereo_calibration.json", output_dirs["calibration"])
    elif not camera_path.exists():
        raise FileNotFoundError(camera_path)

    if solver == "subset":
        if compute_fields:
            subset_cfg = load_config(resolve_path(stereo_cfg.get("configs", {}).get("subset", "config/subset_2d.yaml")))
            compute_subset_fields(paths, subset_cfg, output_dirs["disp"] / "subset")
        if reconstruct:
            reconstruct_subset(paths, output_dirs, stereo_cfg.get("reconstruction", {}), camera_path)
    elif solver == "mesh":
        if args.element and args.element != "all":
            element_types = [args.element]
        else:
            element_values = solver_cfg.get("element_types", ["T3", "Q4", "Q8"])
            element_types = [str(v).upper() for v in element_values]
        if compute_fields:
            mesh_cfg = load_config(resolve_path(stereo_cfg.get("configs", {}).get("mesh", "config/mesh_2d.yaml")))
            compute_mesh_fields(paths, mesh_cfg, output_dirs["disp"] / "mesh", element_types)
        if reconstruct:
            reconstruct_mesh(output_dirs, stereo_cfg.get("reconstruction", {}), camera_path, element_types)
    else:
        raise ValueError(f"Unsupported solver: {solver}")


if __name__ == "__main__":
    main()
