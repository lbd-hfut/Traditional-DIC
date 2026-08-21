"""Normalized 2D Subset-DIC workflow facade implementation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import traditional_dic as tdic
from ..case import ResolvedCase
from ..config import subset_method_tag
from ..config_resolver import ResolvedConfig
from ..postprocess import save_least_squares_strain_csv
from ..visualization import plot_2d_field_overlay
from .common import WorkflowRunResult, execute_with_contract, make_context, output_paths, require_workflow

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


def write_stats(result: dict[str, np.ndarray], path: Path, metric: str = "znssd") -> None:
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
        f"{metric}_mean": float(q.mean()) if q.size else 0.0,
        f"{metric}_max": float(q.max()) if q.size else 0.0,
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_legend(path: Path, component: str, vmin: float, vmax: float) -> None:
    path.write_text(
        f"component={component}\ncolor_min={vmin}\ncolor_max={vmax}\n"
        "colormap=jet percentile clipped\n",
        encoding="utf-8",
    )


def field_limits(result: dict[str, np.ndarray], component: str) -> tuple[float, float]:
    valid = np.asarray(result["valid"], dtype=bool)
    u = np.asarray(result["u"], dtype=np.float64)[valid]
    v = np.asarray(result["v"], dtype=np.float64)[valid]
    values = u if component == "u" else v if component == "v" else np.hypot(u, v)
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1.0, 99.0])
    if component == "mag":
        vmin = 0.0
    return float(vmin), float(vmax)


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



def _run_subset_2d_impl(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
) -> WorkflowRunResult:
    """Run Subset-DIC using only resolved F1/F2 contracts."""
    context = make_context(
        resolved_case,
        resolved_config,
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
    )
    require_workflow(context, "subset_2d")
    case = context.resolved_case
    config = context.resolved_config.backend_config()
    reference_path = case.frame("reference").path
    deformed_paths = [frame.path for frame in case.frames if frame.role == "deformed"]
    roi_path = case.roi.path or case.frame("roi").path
    result_root, visualization_dir_root = output_paths(
        context,
        "subset",
        default_result="result/subset",
        default_visualization="visualization/subset",
    )
    reference = read_gray(reference_path)
    roi = read_mask(roi_path)
    artifacts: dict[str, list[str]] = {"displacements": [], "strains": [], "stats": [], "figures": []}
    for deformed_path in deformed_paths:
        name = deformed_path.stem
        method_tag = subset_method_tag(config)
        suffix = f"_{method_tag}" if method_tag else ""
        out_dir = result_root / f"{name}{suffix}"
        visualization_dir = visualization_dir_root / f"{name}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        result = tdic.subset(reference, read_gray(deformed_path), config=config, roi=roi)
        displacement_path = out_dir / "displacements.csv"
        save_subset_csv(result, displacement_path)
        artifacts["displacements"].append(str(displacement_path))
        strain_cfg = dict(config.get("strain", {}) or {})
        if bool(strain_cfg.get("enabled", False)):
            valid = np.asarray(result["valid"], dtype=bool)
            points = np.column_stack((np.asarray(result["x"])[valid], np.asarray(result["y"])[valid]))
            displacement = np.column_stack((np.asarray(result["u"])[valid], np.asarray(result["v"])[valid]))
            strain_path = out_dir / "strain.csv"
            save_least_squares_strain_csv(
                strain_path,
                points,
                displacement,
                radius=float(strain_cfg["radius"]),
                min_samples=int(strain_cfg.get("min_samples", 6)),
                green_lagrange=str(strain_cfg.get("measure", "green_lagrange")) == "green_lagrange",
            )
            artifacts["strains"].append(str(strain_path))
        criterion = str(config.get("correlation", {}).get("criterion", "znssd")).strip().lower()
        stats_path = out_dir / "stats.json"
        write_stats(result, stats_path, metric="ssd" if criterion == "ssd" else "znssd")
        artifacts["stats"].append(str(stats_path))
        for component in ("mag", "u", "v"):
            vmin, vmax = field_limits(result, component)
            label = "|U| px" if component == "mag" else f"{component} px"
            figure_path = visualization_dir / f"subset_field_{component}.png"
            plot_2d_field_overlay(
                reference,
                result,
                figure_path,
                component=component,
                title=f"Subset {label}",
                label=label,
                cmap="jet",
            )
            write_legend(visualization_dir / f"subset_field_{component}_legend.txt", component, vmin, vmax)
            artifacts["figures"].append(str(figure_path))
        save_overview(visualization_dir)
        artifacts["figures"].append(str(visualization_dir / "overview.png"))
    return WorkflowRunResult("subset_2d", result_root, artifacts)


def run_subset_2d(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
    run_id: str | None = None,
) -> WorkflowRunResult:
    """Run Subset-DIC and write the additive normalized F4 run contract."""
    return execute_with_contract(
        "subset_2d",
        resolved_case,
        resolved_config,
        solver_root="subset",
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
        run_id=run_id,
        runner=lambda: _run_subset_2d_impl(
            resolved_case,
            resolved_config,
            repository_root=repository_root,
            output_root=output_root,
            visualization_root=visualization_root,
        ),
    )
