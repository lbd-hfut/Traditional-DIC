import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_nodes(path):
    nodes = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                nodes.append((float(parts[1]), float(parts[2])))
    return np.asarray(nodes, dtype=np.float64)


def read_elements(path, nn):
    elements = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= nn + 1:
                elements.append([int(v) - 1 for v in parts[1 : nn + 1]])
    return elements


def read_u(path):
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            rows.append((float(r["u"]), float(r["v"])))
    return np.asarray(rows, dtype=np.float64)


def shape_t3(xi, eta):
    return np.array([1.0 - xi - eta, xi, eta])


def shape_q4(xi, eta):
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


def shape_q8(xi, eta):
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
        ]
    )


def color_map(values, vmin, vmax):
    t = np.zeros_like(values, dtype=np.float64)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def blend_pixel(img, weight, x, y, color):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] += color
        weight[y, x] += 1.0


def render_field(nodes, elements, U, etype, width, height, component):
    accum = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    values = []
    samples = []

    if etype == "T3":
        shape = shape_t3
        grid = []
        div = 14
        for i in range(div + 1):
            for j in range(div + 1 - i):
                grid.append((i / div, j / div))
    else:
        shape = shape_q8 if etype == "Q8" else shape_q4
        div = 16
        coords = np.linspace(-1.0, 1.0, div + 1)
        grid = [(xi, eta) for xi in coords for eta in coords]

    for elem in elements:
        xy = nodes[elem]
        uv = U[elem]
        for xi, eta in grid:
            N = shape(xi, eta)
            p = N @ xy
            disp = N @ uv
            val = disp[0] if component == "u" else disp[1] if component == "v" else math.hypot(disp[0], disp[1])
            samples.append((int(round(p[0])), int(round(p[1])), val))
            values.append(val)

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(values, [1.0, 99.0])
        if component == "mag":
            vmin = 0.0
    for x, y, val in samples:
        color = color_map(np.asarray([val]), vmin, vmax)[0]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                blend_pixel(accum, weight, x + dx, y + dy, color)

    mask = weight > 0
    out = np.ones((height, width, 3), dtype=np.float64)
    out[mask] = accum[mask] / weight[mask, None]
    return (np.clip(out * 255.0, 0, 255).astype(np.uint8), float(vmin), float(vmax))


def draw_mesh_edges(image, nodes, elements, etype):
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    edge_color = (145, 145, 145, 95)
    if etype == "T3":
        edge_ids = [(0, 1), (1, 2), (2, 0)]
    elif etype == "Q4":
        edge_ids = [(0, 1), (1, 2), (2, 3), (3, 0)]
    else:
        edge_ids = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
    for elem in elements:
        for a, b in edge_ids:
            p0 = tuple(nodes[elem[a]])
            p1 = tuple(nodes[elem[b]])
            draw.line([p0, p1], fill=edge_color, width=1)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def write_legend(path, component, vmin, vmax):
    Path(path).write_text(
        f"component={component}\n"
        f"color_min={vmin}\n"
        f"color_max={vmax}\n"
        "colormap=blue-green-red percentile clipped\n"
        "mesh_edges=rgba(145,145,145,0.37)\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--elements", required=True)
    parser.add_argument("--u", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--etype", required=True, choices=["T3", "Q4", "Q8"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    args = parser.parse_args()

    nn = {"T3": 3, "Q4": 4, "Q8": 8}[args.etype]
    nodes = read_nodes(args.nodes)
    elements = read_elements(args.elements, nn)
    U = read_u(args.u)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for component in ["u", "v", "mag"]:
        rgb, vmin, vmax = render_field(nodes, elements, U, args.etype, args.width, args.height, component)
        image = Image.fromarray(rgb, "RGB")
        image = draw_mesh_edges(image, nodes, elements, args.etype)
        image.save(out_dir / f"{args.etype}_field_{component}_mesh_overlay.png")
        write_legend(out_dir / f"{args.etype}_field_{component}_legend.txt", component, vmin, vmax)


if __name__ == "__main__":
    main()
