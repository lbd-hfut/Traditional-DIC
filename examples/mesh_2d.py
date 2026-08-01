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
from traditional_dic.config import load_config, mesh_generation_config, normalize_mesh_config  # noqa: E402
from traditional_dic.visualization import visualization_dir_for_result  # noqa: E402


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


def save_overview(out_dir: Path, element_types: list[str]) -> None:
    tile_w, tile_h = 360, 390
    overview = Image.new("RGB", (tile_w * 3, tile_h * len(element_types)), "white")
    draw = ImageDraw.Draw(overview)
    for row, etype in enumerate(element_types):
        for col, component in enumerate(("mag", "u", "v")):
            path = out_dir / etype / "field_visualization" / f"{etype}_field_{component}_mesh_overlay.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w - 24, tile_h - 50))
            x0, y0 = col * tile_w, row * tile_h
            label = "|U|" if component == "mag" else component
            draw.text((x0 + 16, y0 + 14), f"{etype} {label}", fill=(20, 20, 20))
            overview.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 42))
    overview.save(out_dir / "overview.png")


def run_element(
    reference: np.ndarray,
    deformed: np.ndarray,
    generated_meshes: dict,
    out_root: Path,
    visualization_root: Path,
    etype: str,
    args,
    api_config: dict,
) -> None:
    out_dir = out_root / etype
    element_visualization_dir = visualization_root / etype
    vis_dir = element_visualization_dir / "field_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
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
        config=api_config,
    )
    dic_io.save_displacement_csv(result, out_dir / "final_U.csv")

    height, width = reference.shape
    for component in ("mag", "u", "v"):
        image, vmin, vmax = render_mesh_field(reference, nodes, elements, result, etype, width, height, component)
        image.save(vis_dir / f"{etype}_field_{component}_mesh_overlay.png")
        write_legend(vis_dir / f"{etype}_field_{component}_legend.txt", component, vmin, vmax)

    summary = {
        "nodes": int(nodes.shape[0]),
        "elements": int(elements.shape[0]),
        "mag_mean": float(np.mean(result["mag"])) if len(result["mag"]) else 0.0,
        "mag_max": float(np.max(result["mag"])) if len(result["mag"]) else 0.0,
        "solver": api_config.get("mesh", {}).get("solver", ""),
        "criterion": api_config.get("mesh", {}).get("criterion", ""),
        "initialization": api_config.get("initialization", {}).get("method", ""),
        "nodes_file": f"nodes_{etype}.txt",
        "elements_file": f"elements_{etype}.txt",
        "mesh_preview": str(element_visualization_dir / f"mesh_preview_{etype}.png"),
        "generation_summary": generated_meshes.get("summary", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {etype} Mesh-DIC results to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    ring_root = PROJECT_ROOT / "case" / "mono_DIC" / "ring"
    parser.add_argument("--reference", type=Path, default=ring_root / "001.bmp")
    parser.add_argument("--deformed", type=Path, default=ring_root / "002.bmp")
    parser.add_argument("--roi", type=Path, default=ring_root / "003.bmp")
    parser.add_argument("--out-dir", type=Path, default=ring_root / "result" / "mesh")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "mesh_2d.yaml")
    parser.add_argument("--element", choices=["T3", "Q4", "Q8", "all"], default="all")
    parser.add_argument("--solver", choices=["icgn", "fgn", "forward_gn", "forward_gauss_newton"], default="icgn")
    parser.add_argument("--initialization", choices=["integer_search", "sift"], default="sift")
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--search-radius", type=int, default=20)
    parser.add_argument("--regularization-alpha", type=float, default=0.0)
    args = parser.parse_args()

    reference = read_gray(args.reference)
    deformed = read_gray(args.deformed)
    roi = read_mask(args.roi)
    raw_config = load_config(args.config) if args.config else {}
    api_config = normalize_mesh_config(raw_config)
    generation = mesh_generation_config(raw_config)
    if api_config.get("mesh", {}).get("criterion") == "znssd":
        print("Warning: Mesh ZNSSD global solver is not implemented; use SSD for solved mesh displacement.")
    element_types = ["T3", "Q4", "Q8"] if args.element == "all" else [args.element]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = visualization_dir_for_result(args.reference.parent, args.out_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    generated_meshes = tdic.generate_annulus_meshes_from_mask(
        roi,
        target_element_size=float(generation.get("target_element_size", 35.0)),
        min_element_size=float(generation.get("min_element_size", 18.0)),
        max_element_size=float(generation.get("max_element_size", 55.0)),
        min_element_quality=float(generation.get("min_element_quality", 0.1)),
        config=generation,
    )
    (args.out_dir / "mesh_generation_summary.json").write_text(
        json.dumps(generated_meshes.get("summary", {}), indent=2),
        encoding="utf-8",
    )

    for etype in element_types:
        run_element(reference, deformed, generated_meshes, args.out_dir, visualization_dir, etype, args, api_config)
    save_overview(visualization_dir, element_types)
    print(f"Wrote Mesh-DIC overview to {visualization_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
