"""Run a complete 2D Subset-DIC workflow through the Python API."""

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
from traditional_dic.config import load_config, normalize_subset_config  # noqa: E402


def read_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("F")
    arr = np.array(image, dtype=np.float32, copy=True)
    max_value = float(np.max(arr))
    if max_value > 0.0:
        arr /= max_value
    return arr


def read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def save_subset_csv(result: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "u", "v", "du_dx", "du_dy", "dv_dx", "dv_dy", "correlation", "valid"])
        n = len(result["x"])
        for i in range(n):
            writer.writerow(
                [
                    result["x"][i],
                    result["y"][i],
                    result["u"][i],
                    result["v"][i],
                    result["du_dx"][i],
                    result["du_dy"][i],
                    result["dv_dx"][i],
                    result["dv_dy"][i],
                    result["correlation"][i],
                    int(result["valid"][i]),
                ]
            )


def color_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    t = np.zeros_like(values, dtype=np.float64)
    if vmax > vmin:
        t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def render_points(result: dict[str, np.ndarray], component: str, width: int, height: int) -> tuple[Image.Image, float, float]:
    valid = np.asarray(result["valid"], dtype=bool)
    x = np.asarray(result["x"], dtype=np.float64)[valid]
    y = np.asarray(result["y"], dtype=np.float64)[valid]
    u = np.asarray(result["u"], dtype=np.float64)[valid]
    v = np.asarray(result["v"], dtype=np.float64)[valid]
    values = u if component == "u" else v if component == "v" else np.hypot(u, v)

    if values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(values, [1.0, 99.0])
        if component == "mag":
            vmin = 0.0

    image = np.ones((height, width, 3), dtype=np.float64)
    colors = color_map(values, float(vmin), float(vmax))
    radius = 2
    for xi, yi, color in zip(x, y, colors):
        cx = int(round(xi))
        cy = int(round(yi))
        for yy in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for xx in range(max(0, cx - radius), min(width, cx + radius + 1)):
                image[yy, xx] = color
    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), "RGB"), float(vmin), float(vmax)


def write_stats(result: dict[str, np.ndarray], path: Path) -> None:
    valid = np.asarray(result["valid"], dtype=bool)
    u = np.asarray(result["u"], dtype=np.float64)[valid]
    v = np.asarray(result["v"], dtype=np.float64)[valid]
    q = np.asarray(result["correlation"], dtype=np.float64)[valid]
    mag = np.hypot(u, v)
    stats = {
        "total_points": int(len(result["x"])),
        "valid_points": int(np.sum(valid)),
        "invalid_points": int(len(result["x"]) - np.sum(valid)),
        "u_mean": float(u.mean()) if u.size else 0.0,
        "u_std": float(u.std()) if u.size else 0.0,
        "v_mean": float(v.mean()) if v.size else 0.0,
        "v_std": float(v.std()) if v.size else 0.0,
        "mag_mean": float(mag.mean()) if mag.size else 0.0,
        "mag_max": float(mag.max()) if mag.size else 0.0,
        "znssd_mean": float(q.mean()) if q.size else 0.0,
        "znssd_max": float(q.max()) if q.size else 0.0,
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_legend(path: Path, component: str, vmin: float, vmax: float) -> None:
    path.write_text(
        f"component={component}\ncolor_min={vmin}\ncolor_max={vmax}\n"
        "colormap=blue-green-red percentile clipped\n",
        encoding="utf-8",
    )


def save_overview(out_dir: Path) -> None:
    labels = [("|U|", "mag"), ("u", "u"), ("v", "v")]
    tile_w, tile_h = 420, 460
    overview = Image.new("RGB", (tile_w * 3, tile_h), "white")
    draw = ImageDraw.Draw(overview)
    for idx, (label, name) in enumerate(labels):
        path = out_dir / f"subset_field_{name}.png"
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 24, tile_h - 58))
        x0 = idx * tile_w
        draw.text((x0 + 18, 18), f"Subset {label}", fill=(20, 20, 20))
        overview.paste(image, (x0 + (tile_w - image.width) // 2, 50))
    overview.save(out_dir / "overview.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    ring_root = PROJECT_ROOT / "case" / "mono_DIC" / "ring"
    parser.add_argument("--reference", type=Path, default=ring_root / "001.bmp")
    parser.add_argument("--deformed", type=Path, default=ring_root / "002.bmp")
    parser.add_argument("--roi", type=Path, default=ring_root / "003.bmp")
    parser.add_argument("--out-dir", type=Path, default=ring_root / "result" / "subset")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "subset_2d.yaml")
    parser.add_argument("--radius", type=int, default=37)
    parser.add_argument("--spacing", type=int, default=3)
    parser.add_argument("--search-radius", type=int, default=30)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = read_gray(args.reference)
    deformed = read_gray(args.deformed)
    roi = read_mask(args.roi)
    config = normalize_subset_config(load_config(args.config)) if args.config else None

    result = tdic.subset(
        reference,
        deformed,
        config=config,
        roi=roi,
        radius=args.radius,
        seed_subset_radius=args.radius,
        search_radius=args.search_radius,
        seed_count=args.seed_count,
        propagation_spacing=args.spacing,
        max_iterations=args.max_iterations,
    )

    save_subset_csv(result, args.out_dir / "displacements.csv")
    write_stats(result, args.out_dir / "stats.json")
    height, width = reference.shape
    for component in ("mag", "u", "v"):
        image, vmin, vmax = render_points(result, component, width, height)
        image.save(args.out_dir / f"subset_field_{component}.png")
        write_legend(args.out_dir / f"subset_field_{component}_legend.txt", component, vmin, vmax)
    save_overview(args.out_dir)

    print(f"Wrote Subset-DIC results to {args.out_dir}")


if __name__ == "__main__":
    main()
