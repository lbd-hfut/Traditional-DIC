"""Regenerate visualizations for case results without rerunning DIC solvers.

The output policy mirrors the examples: structured data stays under
``case/.../result`` and PNG visualizations are written under the matching
``case/.../visualization`` path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from traditional_dic.visualization import (  # noqa: E402
    densify_2d_mesh_displacement_field,
    plot_2d_field_overlay,
    plot_3d_scatter_field,
    plot_3d_surface_field,
    visualization_dir_for_result,
)


FIELD_COMPONENTS_2D = ("mag", "u", "v")
FIELD_COMPONENTS_3D = ("umag", "ux", "uy", "uz")
OVERWRITE = False


def should_write(path: Path) -> bool:
    return OVERWRITE or not path.exists()


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_column(rows: list[Mapping[str, str]], *names: str, default: float = np.nan) -> np.ndarray:
    key = next((name for name in names if rows and name in rows[0]), None)
    if key is None:
        return np.full(len(rows), default, dtype=np.float64)
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, default)))
        except (TypeError, ValueError):
            out.append(default)
    return np.asarray(out, dtype=np.float64)


def bool_column(rows: list[Mapping[str, str]], name: str = "valid") -> np.ndarray:
    if not rows or name not in rows[0]:
        return np.ones(len(rows), dtype=bool)
    return np.asarray([str(row.get(name, "1")).strip().lower() not in {"", "0", "false", "nan", "no"} for row in rows], dtype=bool)


def field_from_csv(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(path)
    return {
        "x": numeric_column(rows, "x", "x_l0"),
        "y": numeric_column(rows, "y", "y_l0"),
        "u": numeric_column(rows, "u", "Ux"),
        "v": numeric_column(rows, "v", "Uy"),
        "valid": bool_column(rows),
    }


def read_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def find_case_roots() -> list[Path]:
    roots: list[Path] = []
    for family in ("mono_DIC", "stereo_DIC", "multi_DIC"):
        base = PROJECT_ROOT / "case" / family
        if not base.exists():
            continue
        roots.extend(path for path in base.iterdir() if path.is_dir() and (path / "result").exists())
    return sorted(roots, key=lambda p: str(p).lower())


def reference_for_result(case_root: Path, result_path: Path) -> Path | None:
    parts = result_path.relative_to(case_root / "result").parts
    if "multi_DIC" in case_root.parts and len(parts) >= 2:
        pair = next((part for part in parts if re.match(r"cam_\d+-cam_\d+", part)), "")
        if pair:
            left = pair.split("-", 1)[0]
            image = case_root / "images" / left / "001.bmp"
            if image.exists():
                return image
    if "stereo_DIC" in case_root.parts:
        for candidate in (case_root / "cam1" / "00_L.bmp", case_root / "001.bmp"):
            if candidate.exists():
                return candidate
    for candidate in (case_root / "001.bmp", case_root / "cam1" / "00_L.bmp"):
        if candidate.exists():
            return candidate
    return None


def write_overview(image_paths: Iterable[Path], out_path: Path, title_prefix: str = "") -> None:
    paths = [path for path in image_paths if path.exists()]
    if not paths:
        return
    cols = min(3, len(paths))
    rows = int(np.ceil(len(paths) / cols))
    tile_w, tile_h = 420, 460
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 24, tile_h - 58))
        x0 = (idx % cols) * tile_w
        y0 = (idx // cols) * tile_h
        label = f"{title_prefix}{path.stem}".strip()
        draw.text((x0 + 18, y0 + 18), label, fill=(20, 20, 20))
        canvas.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 50))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def visualize_2d_field_csv(case_root: Path, csv_path: Path) -> list[str]:
    reference_path = reference_for_result(case_root, csv_path)
    if reference_path is None:
        return []
    field = field_from_csv(csv_path)
    out_dir = visualization_dir_for_result(case_root, csv_path.parent)
    stem = csv_path.stem
    if stem == "displacements":
        stem = "subset_field"
    outputs = []
    for component in FIELD_COMPONENTS_2D:
        suffix = component
        out_path = out_dir / f"{stem}_{suffix}.png"
        label = "|d| px" if component == "mag" else f"{component} px"
        if should_write(out_path):
            plot_2d_field_overlay(
                reference_path,
                field,
                out_path,
                component=component,
                title=f"{csv_path.parent.name} {label}",
                label=label,
                cmap="jet",
                point_size=6.0,
            )
        outputs.append(str(out_path))
    overview_path = out_dir / "overview.png"
    if should_write(overview_path):
        write_overview([Path(p) for p in outputs], overview_path)
    return outputs


def parse_nodes(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append((float(parts[1]), float(parts[2])))
    return np.asarray(rows, dtype=np.float64)


def parse_elements(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 4:
                rows.append([int(v) - 1 for v in parts[1:]])
    return np.asarray(rows, dtype=np.int64)


def mesh_edge_ids(element_type: str) -> list[tuple[int, int]]:
    if element_type == "T3":
        return [(0, 1), (1, 2), (2, 0)]
    if element_type == "Q8":
        return [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
    return [(0, 1), (1, 2), (2, 3), (3, 0)]


def draw_mesh_edges(image: Image.Image, nodes: np.ndarray, elements: np.ndarray, element_type: str) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for elem in elements:
        for a, b in mesh_edge_ids(element_type):
            if a < len(elem) and b < len(elem):
                draw.line([tuple(nodes[elem[a]]), tuple(nodes[elem[b]])], fill=(128, 128, 128, 255), width=1)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def visualize_mesh_result(case_root: Path, final_u_path: Path) -> list[str]:
    element_dir = final_u_path.parent
    element_type = element_dir.name.upper()
    nodes_path = element_dir / f"nodes_{element_type}.txt"
    elements_path = element_dir / f"elements_{element_type}.txt"
    reference_path = reference_for_result(case_root, final_u_path)
    if not nodes_path.exists() or not elements_path.exists() or reference_path is None:
        return []
    nodes = parse_nodes(nodes_path)
    elements = parse_elements(elements_path)
    image = Image.open(reference_path).convert("L")
    mesh_preview = draw_mesh_edges(image.convert("RGB"), nodes, elements, element_type).convert("RGB")
    out_dir = visualization_dir_for_result(case_root, element_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / f"mesh_preview_{element_type}.png"
    if should_write(preview_path):
        mesh_preview.save(preview_path)

    field = field_from_csv(final_u_path)
    field["x"] = nodes[:, 0]
    field["y"] = nodes[:, 1]
    dense_field = densify_2d_mesh_displacement_field(
        nodes,
        elements,
        field["u"],
        field["v"],
        element_type,
        valid=field["valid"],
    )
    vis_dir = out_dir / "field_visualization"
    outputs = []
    for component in FIELD_COMPONENTS_2D:
        out_path = vis_dir / f"{element_type}_field_{component}_mesh_overlay.png"
        if should_write(out_path):
            plot_2d_field_overlay(
                reference_path,
                dense_field,
                out_path,
                component=component,
                title=f"{element_type} {component}",
                label=component,
                cmap="jet",
                point_size=8.0,
                mesh_nodes=nodes,
                mesh_elements=elements,
                mesh_element_type=element_type,
            )
        outputs.append(str(out_path))
    return [str(out_dir / f"mesh_preview_{element_type}.png"), *outputs]


def visualize_3d_csv(case_root: Path, csv_path: Path) -> list[str]:
    rows = read_csv_rows(csv_path)
    if not rows:
        return []
    valid = bool_column(rows)
    x = numeric_column(rows, "x_l0", "x")
    y = numeric_column(rows, "y_l0", "y")
    reference_path = reference_for_result(case_root, csv_path)
    out_dir = visualization_dir_for_result(case_root, csv_path.parent)
    outputs = []
    values_by_name = {
        "ux": numeric_column(rows, "Ux"),
        "uy": numeric_column(rows, "Uy"),
        "uz": numeric_column(rows, "Uz"),
        "umag": numeric_column(rows, "Umag"),
    }
    if "X0" in rows[0] and "Y0" in rows[0] and "Z0" in rows[0]:
        points = np.column_stack((numeric_column(rows, "X0"), numeric_column(rows, "Y0"), numeric_column(rows, "Z0")))
        for name, values in values_by_name.items():
            out_path = out_dir / f"{csv_path.stem}_{name}_scatter.png"
            if should_write(out_path):
                plot_3d_scatter_field(
                    points[valid],
                    values[valid],
                    out_path,
                    title=f"{csv_path.parent.name} {name}",
                    label=name,
                    cmap="jet",
                    max_points=25000,
                )
            outputs.append(str(out_path))
    if reference_path is not None and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        for name, values in values_by_name.items():
            field = {"x": x, "y": y, "u": values, "v": np.zeros_like(values), "valid": valid & np.isfinite(values)}
            out_path = out_dir / f"{csv_path.stem}_{name}.png"
            if should_write(out_path):
                plot_2d_field_overlay(reference_path, field, out_path, component="u", title=f"{csv_path.parent.name} {name}", label=name, cmap="jet")
            outputs.append(str(out_path))
        if "Z0" in rows[0]:
            z0 = numeric_column(rows, "Z0")
            field = {"x": x, "y": y, "u": z0, "v": np.zeros_like(z0), "valid": valid & np.isfinite(z0)}
            out_path = out_dir / f"{csv_path.stem}_shape_ref_z.png"
            if should_write(out_path):
                plot_2d_field_overlay(reference_path, field, out_path, component="u", title=f"{csv_path.parent.name} Z0", label="Z0", cmap="jet")
            outputs.append(str(out_path))
    return outputs


def visualize_stitched_surface(case_root: Path, points_path: Path) -> list[str]:
    faces_path = points_path.parent / "stitched_faces.csv"
    if not faces_path.exists():
        return []
    point_rows = read_csv_rows(points_path)
    face_rows = read_csv_rows(faces_path)
    valid = bool_column(point_rows)
    reference = np.column_stack((numeric_column(point_rows, "X0"), numeric_column(point_rows, "Y0"), numeric_column(point_rows, "Z0")))
    deformed = np.column_stack((numeric_column(point_rows, "X1"), numeric_column(point_rows, "Y1"), numeric_column(point_rows, "Z1")))
    displacement = deformed - reference
    faces = np.column_stack((numeric_column(face_rows, "n1"), numeric_column(face_rows, "n2"), numeric_column(face_rows, "n3"))).astype(np.int64) - 1
    bad = np.where(~valid)[0]
    if bad.size:
        reference[bad] = np.nan
        deformed[bad] = np.nan
        displacement[bad] = np.nan
    out_dir = visualization_dir_for_result(case_root, points_path.parent)
    outputs = []
    ref_path = out_dir / "stitched_reference_scene.png"
    if should_write(ref_path):
        plot_3d_surface_field(reference, faces, reference[:, 2], ref_path, title="Stitched reference surface", label="Z", cmap="viridis")
    outputs.append(str(ref_path))
    def_path = out_dir / "stitched_deformed_scene.png"
    if should_write(def_path):
        plot_3d_surface_field(deformed, faces, deformed[:, 2], def_path, title="Stitched deformed surface", label="Z", cmap="viridis")
    outputs.append(str(def_path))
    values = {
        "ux": displacement[:, 0],
        "uy": displacement[:, 1],
        "uz": displacement[:, 2],
        "umag": np.linalg.norm(displacement, axis=1),
    }
    for name, value in values.items():
        out_path = out_dir / f"stitched_displacement_{name}.png"
        if should_write(out_path):
            plot_3d_surface_field(reference, faces, value, out_path, title=f"Stitched displacement {name.upper()}", label=name.upper(), cmap="jet")
        outputs.append(str(out_path))
    return outputs


def mirror_existing_result_pngs(case_root: Path, patterns: Iterable[str]) -> list[str]:
    outputs = []
    for pattern in patterns:
        for src in (case_root / "result").glob(pattern):
            dst = visualization_dir_for_result(case_root, src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if should_write(dst):
                shutil.copy2(src, dst)
            outputs.append(str(dst))
    return outputs


def visualize_stereo_calibration(case_root: Path) -> list[str]:
    calibration_json = case_root / "result" / "calibration" / "stereo_calibration.json"
    if not calibration_json.exists():
        return []
    from case.visualize_calibration_results import visualize_saved_stereo_calibration_result

    out_dir = visualization_dir_for_result(case_root, calibration_json.parent)
    if OVERWRITE or not any(out_dir.glob("*.png")):
        visualize_saved_stereo_calibration_result(calibration_json, out_dir)
    return [str(path) for path in out_dir.glob("*.png")]


def visualize_multiview_calibration(case_root: Path) -> list[str]:
    calibration_path = case_root / "result" / "calibration" / "calibration_result_scaled.json"
    if not calibration_path.exists():
        calibration_path = case_root / "result" / "calibration" / "calibration_result.json"
    if not calibration_path.exists():
        return []
    from examples.multiview_3d import _save_calibration_visualization

    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    image_paths = []
    for camera in data.get("cameras", []):
        label = str(camera.get("label", f"cam_{len(image_paths)}"))
        image = case_root / "images" / label / "001.bmp"
        if image.exists():
            image_paths.append(image)
    if len(image_paths) < 2:
        return []
    out_dir = visualization_dir_for_result(case_root, case_root / "result" / "calibration")
    if OVERWRITE or not (out_dir / "visualization_outputs.json").exists():
        outputs = _save_calibration_visualization(data, case_root, case_root / "result" / "calibration", image_paths)
    else:
        outputs = json.loads((out_dir / "visualization_outputs.json").read_text(encoding="utf-8"))
    return [str(value) for key, value in outputs.items() if str(value).lower().endswith(".png")]


def visualize_case(case_root: Path) -> dict[str, Any]:
    result_root = case_root / "result"
    outputs: list[str] = []
    warnings: list[str] = []

    for csv_path in sorted(result_root.rglob("displacements.csv"), key=natural_key):
        outputs.extend(visualize_2d_field_csv(case_root, csv_path))
    for csv_path in sorted(result_root.rglob("*disparity.csv"), key=natural_key):
        outputs.extend(visualize_2d_field_csv(case_root, csv_path))
    for csv_path in sorted(result_root.rglob("left_temporal.csv"), key=natural_key):
        outputs.extend(visualize_2d_field_csv(case_root, csv_path))
    for csv_path in sorted(result_root.rglob("final_U.csv"), key=natural_key):
        outputs.extend(visualize_mesh_result(case_root, csv_path))
    for csv_path in sorted(result_root.rglob("*3d_points.csv"), key=natural_key):
        outputs.extend(visualize_3d_csv(case_root, csv_path))
    for csv_path in sorted(result_root.rglob("deformation_3d.csv"), key=natural_key):
        outputs.extend(visualize_3d_csv(case_root, csv_path))
    for points_path in sorted(result_root.rglob("stitched_points.csv"), key=natural_key):
        outputs.extend(visualize_stitched_surface(case_root, points_path))

    outputs.extend(visualize_stereo_calibration(case_root))
    outputs.extend(visualize_multiview_calibration(case_root))
    outputs.extend(
        mirror_existing_result_pngs(
            case_root,
            (
                "mask/**/*.png",
            ),
        )
    )

    summary = {
        "case_root": str(case_root),
        "output_count": len(outputs),
        "outputs": sorted(set(outputs)),
        "warnings": warnings,
    }
    summary_path = case_root / "visualization" / "visualization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global OVERWRITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, action="append", help="Specific case root to visualize. May be repeated.")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "tests" / "case_visualization_summary.json")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even when visualization outputs already exist.")
    args = parser.parse_args()
    OVERWRITE = bool(args.overwrite)

    case_roots = [path.resolve() for path in args.case_root] if args.case_root else find_case_roots()
    summaries = [visualize_case(case_root) for case_root in case_roots]
    payload = {
        "case_count": len(summaries),
        "total_outputs": int(sum(item["output_count"] for item in summaries)),
        "cases": summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": payload["case_count"], "total_outputs": payload["total_outputs"]}, indent=2))


if __name__ == "__main__":
    main()
