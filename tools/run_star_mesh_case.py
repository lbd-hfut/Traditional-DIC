"""Run or visualize one Mesh-DIC optimizer case for the mono_DIC/star inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import traditional_dic as tdic  # noqa: E402
from traditional_dic import io as dic_io  # noqa: E402
from traditional_dic.config import load_config, mesh_generation_config, normalize_mesh_config  # noqa: E402
from traditional_dic.visualization import densify_2d_mesh_displacement_field, plot_2d_field_overlay  # noqa: E402
from examples.mesh_2d import write_dense_displacement_csv, write_elements, write_nodes  # noqa: E402


def read_gray(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    maximum = float(np.max(image))
    return image / maximum if maximum > 0.0 else image


def read_dense_field(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "x": np.asarray(data["x"], dtype=np.float64),
        "y": np.asarray(data["y"], dtype=np.float64),
        "u": np.asarray(data["u"], dtype=np.float64),
        "v": np.asarray(data["v"], dtype=np.float64),
        "valid": np.asarray(data["valid"], dtype=bool),
    }


def generate_roi_mesh(roi: np.ndarray, element_type: str, element_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a structured mesh, retaining only elements fully inside the ROI."""
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
        key = (float(x), float(y))
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(key)
        return point_ids[key]

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
    return np.asarray(points, dtype=np.float64), np.asarray(elements, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    star_root = PROJECT_ROOT / "case" / "mono_DIC" / "star"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("solve", "visualize"))
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--element", choices=("T3", "Q4", "Q8"), required=True)
    parser.add_argument("--optimizer", choices=("icgn", "fgn"))
    parser.add_argument("--objective", choices=("ssd", "znssd"))
    parser.add_argument("--reference", type=Path, default=star_root / "001.bmp")
    parser.add_argument("--deformed", type=Path, default=star_root / "002.bmp")
    parser.add_argument("--roi", type=Path, default=star_root / "003.bmp")
    parser.add_argument("--result-root", type=Path, default=star_root / "result" / "mesh")
    parser.add_argument("--visualization-root", type=Path, default=star_root / "visualization" / "mesh")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "mesh_2d.yaml")
    parser.add_argument("--target-element-size", type=float, default=35.0)
    parser.add_argument("--max-iterations", type=int, default=1500)
    return parser.parse_args()


def solve(args: argparse.Namespace) -> None:
    if args.optimizer is None or args.objective is None:
        raise ValueError("solve requires --optimizer and --objective")
    output_dir = args.result_root / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    optimization = raw_config.setdefault("optimization", {})
    optimization["method"] = "fedic_element_icgn" if args.optimizer == "icgn" else "fedic_element_fgn"
    optimization["objective"] = args.objective
    optimization["max_iterations"] = args.max_iterations
    api_config = normalize_mesh_config(raw_config)

    reference = read_gray(args.reference)
    deformed = read_gray(args.deformed)
    roi = np.asarray(Image.open(args.roi).convert("L"), dtype=np.uint8)
    generation = mesh_generation_config(raw_config)
    target_element_size = args.target_element_size
    nodes, elements = generate_roi_mesh(roi, args.element, target_element_size)
    write_nodes(output_dir / f"nodes_{args.element}.txt", nodes)
    write_elements(output_dir / f"elements_{args.element}.txt", elements)

    # Pass the YAML-shaped mapping. traditional_dic.mesh normalizes it exactly once.
    result = tdic.mesh(reference, deformed, nodes, elements, args.element, config=raw_config)
    dic_io.save_displacement_csv(result, output_dir / "final_U.csv")
    dense = densify_2d_mesh_displacement_field(
        nodes, elements, result["u"], result["v"], args.element, valid=result.get("valid"), samples_per_axis=25
    )
    write_dense_displacement_csv(output_dir / "dense_U.csv", dense)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "element": args.element,
                "optimizer": args.optimizer,
                "objective": args.objective,
                "mesh_config": api_config,
                "mesh_generation": {
                    "method": "structured_roi",
                    "target_element_size": target_element_size,
                    "nodes": int(len(nodes)),
                    "elements": int(len(elements)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"nodes={len(nodes)} elements={len(elements)} dense_samples={len(dense['x'])}")


def visualize(args: argparse.Namespace) -> None:
    result_dir = args.result_root / args.case_name
    output_dir = args.visualization_root / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_gray(args.reference)
    dense = read_dense_field(result_dir / "dense_U.csv")
    nodes = np.loadtxt(result_dir / f"nodes_{args.element}.txt", delimiter=",", comments="#", usecols=(1, 2))
    elements = np.loadtxt(result_dir / f"elements_{args.element}.txt", delimiter=",", comments="#", dtype=np.int64)
    elements = np.atleast_2d(elements)[:, 1:] - 1
    for component in ("mag", "u", "v"):
        plot_2d_field_overlay(
            reference,
            dense,
            output_dir / f"{args.case_name}_{component}.png",
            component=component,
            title=f"{args.case_name} {component}",
            label="|U| px" if component == "mag" else f"{component} px",
            cmap="jet",
            alpha=0.9,
            point_size=2.0,
            mesh_nodes=nodes,
            mesh_elements=elements,
            mesh_element_type=args.element,
        )


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "solve":
        solve(arguments)
    else:
        visualize(arguments)
