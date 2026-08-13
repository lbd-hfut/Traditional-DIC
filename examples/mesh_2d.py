"""Run a complete 2D Mesh-DIC workflow through the Python API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import traditional_dic as tdic  # noqa: E402
from traditional_dic import io as dic_io  # noqa: E402
from traditional_dic.config import load_config, mesh_generation_config, mesh_method_tag, normalize_mesh_config  # noqa: E402
from traditional_dic.postprocess import save_least_squares_strain_csv  # noqa: E402
from traditional_dic.visualization import (  # noqa: E402
    densify_2d_mesh_displacement_field,
    plot_2d_field_overlay,
    visualization_dir_for_result,
)


def read_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("F")
    arr = np.array(image, dtype=np.float32, copy=True)
    max_value = float(np.max(arr))
    if max_value > 0.0:
        arr /= max_value
    return arr


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def read_nodes(path: Path) -> np.ndarray:
    nodes = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                nodes.append((float(parts[1]), float(parts[2])))
    return np.asarray(nodes, dtype=np.float64)


def read_elements(path: Path, nn: int) -> np.ndarray:
    elements = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= nn + 1:
                elements.append([int(v) - 1 for v in parts[1 : nn + 1]])
    return np.asarray(elements, dtype=np.int64)


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
            one_based = [str(int(v) + 1) for v in elem]
            f.write(f"{i},{','.join(one_based)}\n")


def write_dense_displacement_csv(path: Path, dense_field: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(dense_field["x"], dtype=np.float64)
    y = np.asarray(dense_field["y"], dtype=np.float64)
    u = np.asarray(dense_field["u"], dtype=np.float64)
    v = np.asarray(dense_field["v"], dtype=np.float64)
    valid = np.asarray(dense_field.get("valid", np.ones_like(x, dtype=bool)), dtype=bool)
    mag = np.hypot(u, v)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "u", "v", "mag", "valid"])
        for i, values in enumerate(zip(x, y, u, v, mag, valid), start=1):
            sx, sy, su, sv, smag, ok = values
            writer.writerow([i, sx, sy, su, sv, smag, int(bool(ok))])


def generate_roi_mesh(roi: np.ndarray, element_type: str, element_size: float) -> dict[str, np.ndarray]:
    """Generate a structured T3/Q4/Q8 mesh, retaining elements inside the ROI."""
    ys, xs = np.nonzero(roi > 0)
    if len(xs) == 0:
        raise ValueError("ROI mask contains no valid pixels")
    step = max(8, int(round(element_size)))
    x_values = list(range(int(xs.min()), int(xs.max()) + 1, step))
    y_values = list(range(int(ys.min()), int(ys.max()) + 1, step))
    if x_values[-1] != int(xs.max()):
        x_values.append(int(xs.max()))
    if y_values[-1] != int(ys.max()):
        y_values.append(int(ys.max()))

    points: list[tuple[float, float]] = []
    point_ids: dict[tuple[float, float], int] = {}
    elements: list[list[int]] = []

    def add_point(x: float, y: float) -> int:
        point = (float(x), float(y))
        if point not in point_ids:
            point_ids[point] = len(points)
            points.append(point)
        return point_ids[point]

    def in_roi(x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < roi.shape[1] and 0 <= iy < roi.shape[0] and roi[iy, ix] > 0

    for j in range(len(y_values) - 1):
        for i in range(len(x_values) - 1):
            x0, x1 = x_values[i], x_values[i + 1]
            y0, y1 = y_values[j], y_values[j + 1]
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            if element_type == "T3":
                for triangle in ((corners[0], corners[1], corners[2]), (corners[0], corners[2], corners[3])):
                    if all(in_roi(*point) for point in triangle):
                        elements.append([add_point(*point) for point in triangle])
            elif element_type == "Q4":
                if all(in_roi(*point) for point in corners):
                    elements.append([add_point(*point) for point in corners])
            else:
                midsides = ((0.5 * (x0 + x1), y0), (x1, 0.5 * (y0 + y1)),
                            (0.5 * (x0 + x1), y1), (x0, 0.5 * (y0 + y1)))
                q8_points = corners + midsides
                if all(in_roi(*point) for point in q8_points):
                    elements.append([add_point(*point) for point in q8_points])

    if not elements:
        raise ValueError(f"No {element_type} elements fit inside the ROI")
    return {"nodes": np.asarray(points, dtype=np.float64), "elements": np.asarray(elements, dtype=np.int64)}


def generate_meshes_from_roi(roi: np.ndarray, generation: dict) -> dict:
    """Use the annulus generator when applicable, otherwise a general ROI mesh."""
    try:
        return tdic.generate_annulus_meshes_from_mask(
            roi,
            target_element_size=float(generation.get("target_element_size", 35.0)),
            min_element_size=float(generation.get("min_element_size", 18.0)),
            max_element_size=float(generation.get("max_element_size", 55.0)),
            min_element_quality=float(generation.get("min_element_quality", 0.1)),
            config=generation,
        )
    except RuntimeError as error:
        if "one hole" not in str(error):
            raise
    target = float(generation.get("target_element_size", 35.0))
    meshes = {element_type: generate_roi_mesh(roi, element_type, target) for element_type in ("T3", "Q4", "Q8")}
    meshes["summary"] = {"method": "structured_roi", "target_element_size": target}
    return meshes


def save_mesh_preview(path: Path, nodes: np.ndarray, elements: np.ndarray, etype: str, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), "white")
    image = draw_mesh_edges(image, nodes, elements, etype).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y in nodes:
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(40, 120, 60))
    image.save(path)


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


def color_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = np.zeros_like(values, dtype=np.float64)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def result_uv(result: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(result["u"], dtype=np.float64), np.asarray(result["v"], dtype=np.float64)])


def render_mesh_field(
    reference: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    result: dict[str, np.ndarray],
    etype: str,
    width: int,
    height: int,
    component: str,
) -> tuple[Image.Image, float, float]:
    accum = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    values = []
    samples = []
    uv = result_uv(result)

    if etype == "T3":
        shape = shape_t3
        div = 14
        grid = [(i / div, j / div) for i in range(div + 1) for j in range(div + 1 - i)]
    else:
        shape = shape_q8 if etype == "Q8" else shape_q4
        coords = np.linspace(-1.0, 1.0, 17)
        grid = [(xi, eta) for xi in coords for eta in coords]

    for elem in elements:
        xy = nodes[elem]
        elem_uv = uv[elem]
        for xi, eta in grid:
            n = shape(xi, eta)
            p = n @ xy
            d = n @ elem_uv
            value = d[0] if component == "u" else d[1] if component == "v" else math.hypot(d[0], d[1])
            samples.append((int(round(p[0])), int(round(p[1])), value))
            values.append(value)

    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(values_arr, [1.0, 99.0])
        if component == "mag":
            vmin = 0.0

    for x, y, value in samples:
        color = color_map(np.asarray([value]), float(vmin), float(vmax))[0]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < width and 0 <= yy < height:
                    accum[yy, xx] += color
                    weight[yy, xx] += 1.0

    base = np.asarray(reference, dtype=np.float64)
    if base.ndim != 2:
        base = np.mean(base[..., :3], axis=2)
    if np.nanmax(base) > 1.0:
        base = base / max(float(np.nanmax(base)), 1.0)
    rgb = np.repeat(np.clip(base, 0.0, 1.0)[..., None], 3, axis=2)
    mask = weight > 0
    field_rgb = accum[mask] / weight[mask, None]
    rgb[mask] = 0.2 * rgb[mask] + 0.8 * field_rgb
    image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), "RGB")
    return draw_mesh_edges(image, nodes, elements, etype), float(vmin), float(vmax)


def draw_mesh_edges(image: Image.Image, nodes: np.ndarray, elements: np.ndarray, etype: str) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    edge_color = (128, 128, 128, 255)
    if etype == "T3":
        edge_ids = [(0, 1), (1, 2), (2, 0)]
    elif etype == "Q4":
        edge_ids = [(0, 1), (1, 2), (2, 3), (3, 0)]
    else:
        edge_ids = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
    for elem in elements:
        for a, b in edge_ids:
            draw.line([tuple(nodes[elem[a]]), tuple(nodes[elem[b]])], fill=edge_color, width=1)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def write_legend(path: Path, component: str, vmin: float, vmax: float) -> None:
    path.write_text(
        f"component={component}\ncolor_min={vmin}\ncolor_max={vmax}\n"
        "colormap=blue-green-red percentile clipped\nmesh_edges=gray,width=1\n",
        encoding="utf-8",
    )


def save_overview(out_dir: Path, element_types: list[str], *, dense: bool = False) -> None:
    tile_w, tile_h = 360, 390
    overview = Image.new("RGB", (tile_w * 3, tile_h * len(element_types)), "white")
    draw = ImageDraw.Draw(overview)
    name_prefix = "dense_field" if dense else "field"
    for row, etype in enumerate(element_types):
        for col, component in enumerate(("mag", "u", "v")):
            path = out_dir / etype / f"{etype}_{name_prefix}_{component}_mesh_overlay.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w - 24, tile_h - 50))
            x0, y0 = col * tile_w, row * tile_h
            label = "|U|" if component == "mag" else component
            title = f"{etype} dense {label}" if dense else f"{etype} {label}"
            draw.text((x0 + 16, y0 + 14), title, fill=(20, 20, 20))
            overview.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 42))
    overview.save(out_dir / ("dense_overview.png" if dense else "overview.png"))


def run_element(
    reference: np.ndarray,
    deformed: np.ndarray,
    generated_meshes: dict,
    out_root: Path,
    visualization_root: Path,
    etype: str,
    args,
    solver_config: dict,
    effective_config: dict,
    roi: np.ndarray | None = None,
) -> None:
    out_dir = out_root / etype
    element_visualization_dir = visualization_root / etype
    element_visualization_dir.mkdir(parents=True, exist_ok=True)
    mesh_data = generated_meshes[etype]
    nodes = np.asarray(mesh_data["nodes"], dtype=np.float64)
    elements = np.asarray(mesh_data["elements"], dtype=np.int64)
    write_nodes(out_dir / f"nodes_{etype}.txt", nodes)
    write_elements(out_dir / f"elements_{etype}.txt", elements)
    save_mesh_preview(element_visualization_dir / f"mesh_preview_{etype}.png", nodes, elements, etype, reference.shape[1], reference.shape[0])

    result = tdic.mesh(
        reference,
        deformed,
        nodes,
        elements,
        element_type=etype,
        config=solver_config,
        roi=roi,
    )
    dic_io.save_displacement_csv(result, out_dir / "final_U.csv")
    dense_field = densify_2d_mesh_displacement_field(
        nodes,
        elements,
        result["u"],
        result["v"],
        etype,
        valid=result.get("valid"),
        samples_per_axis=args.dense_samples_per_axis,
    )
    write_dense_displacement_csv(out_dir / "dense_U.csv", dense_field)
    strain_cfg = dict(solver_config.get("strain", {}) or {})
    if bool(strain_cfg.get("enabled", True)):
        valid = np.asarray(dense_field.get("valid", np.ones_like(dense_field["x"], dtype=bool)), dtype=bool)
        points = np.column_stack((np.asarray(dense_field["x"])[valid], np.asarray(dense_field["y"])[valid]))
        displacement = np.column_stack((np.asarray(dense_field["u"])[valid], np.asarray(dense_field["v"])[valid]))
        save_least_squares_strain_csv(out_dir / "dense_strain.csv", points, displacement,
                                      radius=float(strain_cfg["radius"]),
                                      min_samples=int(strain_cfg.get("min_samples", 6)),
                                      green_lagrange=str(strain_cfg.get("measure", "green_lagrange")) == "green_lagrange")

    height, width = reference.shape
    for component in ("mag", "u", "v"):
        label = "|U| px" if component == "mag" else f"{component} px"
        plot_2d_field_overlay(
            reference,
            dense_field,
            element_visualization_dir / f"{etype}_dense_field_{component}_mesh_overlay.png",
            component=component,
            title=f"{etype} dense {component}",
            label=label,
            cmap="jet",
            alpha=0.9,
            point_size=2.0,
            mesh_nodes=nodes,
            mesh_elements=elements,
            mesh_element_type=etype,
        )

    summary = {
        "nodes": int(nodes.shape[0]),
        "elements": int(elements.shape[0]),
        "dense_samples": int(len(dense_field["x"])),
        "mag_mean": float(np.mean(result["mag"])) if len(result["mag"]) else 0.0,
        "mag_max": float(np.max(result["mag"])) if len(result["mag"]) else 0.0,
        "solver": effective_config.get("mesh", {}).get("optimization_method", ""),
        "criterion": effective_config.get("mesh", {}).get("photometric_objective", ""),
        "initialization": effective_config.get("initialization", {}).get("method", ""),
        "nodes_file": f"nodes_{etype}.txt",
        "elements_file": f"elements_{etype}.txt",
        "nodal_displacement_file": "final_U.csv",
        "dense_displacement_file": "dense_U.csv",
        "mesh_preview": str(element_visualization_dir / f"mesh_preview_{etype}.png"),
        "generation_summary": generated_meshes.get("summary", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {etype} Mesh-DIC results to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "config" / "case_paths.yaml")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "mesh_2d.yaml")
    parser.add_argument("--element", choices=["T3", "Q4", "Q8", "all"], default="all")
    parser.add_argument(
        "--initialization",
        choices=["fedic_fft"],
        help="Override the mesh nodal-initialization route from the YAML configuration.",
    )
    parser.add_argument(
        "--optimization",
        choices=["fedic_element_icgn", "fedic_element_fgn"],
        help="Override the Mesh-DIC global optimization route from the YAML configuration.",
    )
    parser.add_argument(
        "--objective",
        choices=["ssd", "znssd"],
        help="Override the Mesh-DIC photometric objective from the YAML configuration.",
    )
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--search-radius", type=int, default=20)
    parser.add_argument("--regularization-alpha", type=float)
    parser.add_argument("--dense-samples-per-axis", type=int, default=25)
    parser.add_argument("--init-quality-control", action="store_true")
    parser.add_argument("--init-min-zncc", type=float)
    parser.add_argument("--init-max-znssd", type=float)
    parser.add_argument("--init-fedic-qfactor", action="store_true")
    parser.add_argument("--init-fedic-qfactor-std-factor", type=float)
    parser.add_argument("--init-neighbor-mad-factor", type=float)
    parser.add_argument("--init-max-neighbor-deviation", type=float)
    parser.add_argument("--init-interpolation-neighbors", type=int)
    args = parser.parse_args()

    paths_cfg = dict(load_config(args.paths_config).get("mono_2d", {}) or {})
    case_root = Path(paths_cfg["case_root"])
    if not case_root.is_absolute():
        case_root = PROJECT_ROOT / case_root
    images = sorted((case_root / str(paths_cfg.get("images_dir", "."))).glob("*"))
    images = [path for path in images if path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    if len(images) < 3:
        raise ValueError("mono_2d.images_dir must contain reference, at least one deformed image, and ROI")
    output_cfg = dict(paths_cfg.get("output", {}) or {})
    solver_output = dict(dict(output_cfg.get("solver_roots", {}) or {}).get("mesh", {}) or {})
    result_root = case_root / str(solver_output.get("result_root", "result/mesh"))
    visualization_root = case_root / str(solver_output.get("visualization_root", "visualization/mesh"))
    reference = read_gray(images[0])
    roi = read_mask(images[-1])
    raw_config = load_config(args.config) if args.config else {}
    if args.initialization is not None:
        raw_config.setdefault("initialization", {})["method"] = args.initialization
    if args.optimization is not None:
        raw_config.setdefault("optimization", {})["method"] = args.optimization
    if args.objective is not None:
        raw_config.setdefault("optimization", {})["objective"] = args.objective
    if args.regularization_alpha is not None:
        raw_config.setdefault("optimization", {})["regularization_alpha"] = float(args.regularization_alpha)
    if args.init_quality_control:
        qc_config = raw_config.setdefault("initialization", {}).setdefault("quality_control", {})
        qc_config["enabled"] = True
        if args.init_min_zncc is not None:
            qc_config["min_zncc"] = float(args.init_min_zncc)
        if args.init_max_znssd is not None:
            qc_config["max_znssd"] = float(args.init_max_znssd)
        if args.init_fedic_qfactor:
            qc_config["fedic_qfactor_enabled"] = True
        if args.init_fedic_qfactor_std_factor is not None:
            qc_config["fedic_qfactor_std_factor"] = float(args.init_fedic_qfactor_std_factor)
        if args.init_neighbor_mad_factor is not None:
            qc_config["neighbor_mad_factor"] = float(args.init_neighbor_mad_factor)
        if args.init_max_neighbor_deviation is not None:
            qc_config["max_neighbor_deviation"] = float(args.init_max_neighbor_deviation)
        if args.init_interpolation_neighbors is not None:
            qc_config["interpolation_neighbors"] = int(args.init_interpolation_neighbors)
    api_config = normalize_mesh_config(raw_config)
    generation = mesh_generation_config(raw_config)
    element_types = ["T3", "Q4", "Q8"] if args.element == "all" else [args.element]
    generated_meshes = generate_meshes_from_roi(roi, generation)
    for deformed_path in images[1:-1]:
        method_tag = mesh_method_tag(raw_config)
        suffix = f"_{method_tag}" if method_tag else ""
        out_dir = result_root / f"{deformed_path.stem}{suffix}"
        visualization_dir = visualization_root / f"{deformed_path.stem}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mesh_generation_summary.json").write_text(json.dumps(generated_meshes.get("summary", {}), indent=2), encoding="utf-8")
        deformed = read_gray(deformed_path)
        for etype in element_types:
            run_element(reference, deformed, generated_meshes, out_dir, visualization_dir, etype, args, raw_config, api_config, roi=roi)
        save_overview(visualization_dir, element_types, dense=True)
        print(f"Wrote dense Mesh-DIC overview to {visualization_dir / 'dense_overview.png'}")


if __name__ == "__main__":
    main()
