"""Generate calibration-like sparse products and ROI masks for CylinderDIC.

This mirrors ``case/multi_calibrartion/visualize_colmap_like_results.py`` but
writes into ``case/multi_DIC/CylinderDIC/result`` and immediately runs the
multiview mask builder on true per-camera observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.traditional_dic.multiview import (  # noqa: E402
    generate_masks_from_calibration,
    save_pair_selection_report,
    select_camera_pairs,
)


def natural_key(name: str) -> tuple[str, int]:
    head = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return head, int(digits) if digits else -1


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t.reshape(3)


def make_sparse_cylinder_points(config: dict[str, Any], count_theta: int = 180, count_y: int = 80) -> np.ndarray:
    radius = float(config.get("cylinder_radius", 80.0))
    height = float(config.get("cylinder_height", 120.0))
    theta = np.linspace(0.0, 2.0 * np.pi, count_theta, endpoint=False)
    y = np.linspace(-height / 2.0, height / 2.0, count_y)
    tt, yy = np.meshgrid(theta, y)
    return np.column_stack([radius * np.cos(tt.ravel()), yy.ravel(), radius * np.sin(tt.ravel())]).astype(np.float64)


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_cam = R @ points.T + t.reshape(3, 1)
    z = p_cam[2]
    valid_z = z > 1e-9
    uv = np.column_stack([K[0, 0] * (p_cam[0] / z) + K[0, 2], K[1, 1] * (p_cam[1] / z) + K[1, 2]])
    return uv, valid_z


def build_colmap_like_products(case_root: Path, meta: dict[str, Any], points: np.ndarray) -> dict[str, Any]:
    cameras = sorted(meta["cameras"], key=lambda item: natural_key(item["camera_name"]))
    cam_names = [cam["camera_name"] for cam in cameras]
    K_list = [np.asarray(cam["K"], dtype=np.float64) for cam in cameras]
    R_list = [np.asarray(cam["R_world_to_camera"], dtype=np.float64) for cam in cameras]
    t_list = [np.asarray(cam["t_world_to_camera"], dtype=np.float64).reshape(3) for cam in cameras]
    centers = np.asarray([camera_center(R, t) for R, t in zip(R_list, t_list)], dtype=np.float64)
    image_paths = [case_root / "images" / name / "001.bmp" for name in cam_names]

    cam_indices: list[int] = []
    point_indices: list[int] = []
    uv_values: list[list[float]] = []
    width = int(meta["config"].get("image_width", 1440))
    height = int(meta["config"].get("image_height", 1080))

    normals = points.copy()
    normals[:, 1] = 0.0
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    for cam_idx, (K, R, t, center) in enumerate(zip(K_list, R_list, t_list, centers)):
        uv, valid_z = project_points(points, K, R, t)
        view_vec = center.reshape(1, 3) - points
        view_vec /= np.maximum(np.linalg.norm(view_vec, axis=1, keepdims=True), 1e-12)
        front_visible = np.sum(normals * view_vec, axis=1) > 0.05
        inside = (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
        visible = np.where(valid_z & front_visible & inside)[0]
        cam_indices.extend([cam_idx] * len(visible))
        point_indices.extend(visible.tolist())
        uv_values.extend(uv[visible].tolist())

    observations = {
        "cam_indices": np.asarray(cam_indices, dtype=np.int32),
        "point_indices": np.asarray(point_indices, dtype=np.int32),
        "uv": np.asarray(uv_values, dtype=np.float64),
    }
    return {
        "cam_names": cam_names,
        "image_paths": image_paths,
        "K": np.asarray(K_list),
        "R": np.asarray(R_list),
        "t": np.asarray(t_list),
        "centers": centers,
        "points": points,
        "observations": observations,
        "image_width": width,
        "image_height": height,
    }


def calibration_dict(data: dict[str, Any]) -> dict[str, Any]:
    obs_by_point: list[list[dict[str, Any]]] = [[] for _ in range(len(data["points"]))]
    observations = data["observations"]
    for cam_idx, point_idx, uv in zip(observations["cam_indices"], observations["point_indices"], observations["uv"]):
        obs_by_point[int(point_idx)].append({"camera_index": int(cam_idx), "uv": [float(uv[0]), float(uv[1])]})
    cameras = [
        {
            "label": name,
            "K": data["K"][idx].tolist(),
            "R": data["R"][idx].tolist(),
            "t": data["t"][idx].tolist(),
            "image_width": int(data["image_width"]),
            "image_height": int(data["image_height"]),
        }
        for idx, name in enumerate(data["cam_names"])
    ]
    points3d = [
        {"point3d_id": idx, "xyz": point.tolist(), "observations": obs}
        for idx, (point, obs) in enumerate(zip(data["points"], obs_by_point))
        if obs
    ]
    return {"cameras": cameras, "points3d": points3d, "mean_reprojection_error": 0.0}


def axis_limits(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.vstack([points, centers]) if len(points) else centers
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    pad = 0.08 * max(float(np.max(hi - lo)), 1.0)
    return lo - pad, hi + pad


def apply_limits(ax: Any, limits: tuple[np.ndarray, np.ndarray]) -> None:
    lo, hi = limits
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass


def plot_sparse_scene(data: dict[str, Any], output_dir: Path, dpi: int = 160) -> Path:
    points = data["points"]
    centers = data["centers"]
    limits = axis_limits(points, centers)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sample_idx = np.random.RandomState(0).choice(len(points), min(len(points), 30000), replace=False)
    sample = points[sample_idx]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.0, c="0.25", alpha=0.55)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=45, marker="^")
    scale = 0.08 * max(limits[1] - limits[0])
    for idx, center in enumerate(centers):
        ax.text(center[0], center[1], center[2], data["cam_names"][idx], fontsize=7)
        axes = scale * data["R"][idx].T
        for axis, color in zip(axes.T, ["r", "g", "b"]):
            ax.quiver(center[0], center[1], center[2], axis[0], axis[1], axis[2], color=color, linewidth=1)
    apply_limits(ax, limits)
    ax.set_title("Sparse points and camera poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    path = output_dir / "sparse_scene.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_camera_observations_3d(data: dict[str, Any], output_dir: Path, dpi: int = 160) -> Path:
    points = data["points"]
    centers = data["centers"]
    observations = data["observations"]
    limits = axis_limits(points, centers)
    fig = plt.figure(figsize=(16, 12))
    for idx, name in enumerate(data["cam_names"]):
        ax = fig.add_subplot(3, 4, idx + 1, projection="3d")
        mask = observations["cam_indices"] == idx
        observed = points[observations["point_indices"][mask]]
        if len(observed):
            ax.scatter(observed[:, 0], observed[:, 1], observed[:, 2], s=2, c="tab:red", alpha=0.75)
        apply_limits(ax, limits)
        ax.set_title(f"{name}: {len(observed)} obs", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    path = output_dir / "camera_observations_3d.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_camera_observations_2d(data: dict[str, Any], output_dir: Path, dpi: int = 160) -> Path:
    observations = data["observations"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for idx, name in enumerate(data["cam_names"]):
        ax = axes.ravel()[idx]
        image = mpimg.imread(data["image_paths"][idx])
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        uv = observations["uv"][observations["cam_indices"] == idx]
        if len(uv):
            draw_idx = np.random.RandomState(idx).choice(len(uv), min(len(uv), 1200), replace=False)
            ax.scatter(uv[draw_idx, 0], uv[draw_idx, 1], s=8, c="red", marker="o", linewidths=0, alpha=0.7)
        ax.set_title(f"{name}: {len(uv)} obs", fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
    for idx in range(len(data["cam_names"]), len(axes.ravel())):
        axes.ravel()[idx].axis("off")
    fig.tight_layout()
    path = output_dir / "camera_observations_2d.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_calibration_products(data: dict[str, Any], calib: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = data["observations"]
    np.savez_compressed(
        output_dir / "colmap_like_sparse_model.npz",
        cam_names=np.asarray(data["cam_names"]),
        K=data["K"],
        R=data["R"],
        t=data["t"],
        camera_centers_world=data["centers"],
        points3D=data["points"],
        observation_cam_indices=observations["cam_indices"],
        observation_point_indices=observations["point_indices"],
        observation_uv=observations["uv"],
    )
    np.savez_compressed(
        output_dir / "cameras.npz",
        cam_names=np.asarray(data["cam_names"]),
        image_paths=np.asarray([str(path) for path in data["image_paths"]]),
        K=data["K"],
        R=data["R"],
        t=data["t"],
        camera_centers_world=data["centers"],
    )
    np.savez_compressed(output_dir / "observations.npz", **observations)
    (output_dir / "calibration_result.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    summary = {
        "num_cameras": len(data["cam_names"]),
        "num_sparse_points": int(len(data["points"])),
        "num_observed_sparse_points": int(len(calib["points3d"])),
        "num_observations": int(len(observations["cam_indices"])),
        "per_camera_observations": {
            name: int(np.sum(observations["cam_indices"] == idx)) for idx, name in enumerate(data["cam_names"])
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs = {
        "sparse_scene": str(plot_sparse_scene(data, output_dir)),
        "camera_observations_3d": str(plot_camera_observations_3d(data, output_dir)),
        "camera_observations_2d": str(plot_camera_observations_2d(data, output_dir)),
    }
    (output_dir / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "case" / "multi_DIC" / "CylinderDIC")
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    meta = json.loads((case_root / "calibrate_images" / "chessboard_meta.json").read_text(encoding="utf-8"))
    data = build_colmap_like_products(case_root, meta, make_sparse_cylinder_points(meta["config"]))
    calib = calibration_dict(data)

    result_root = case_root / "result"
    calib_dir = result_root / "calibration"
    mask_dir = result_root / "mask"
    calibration_outputs = save_calibration_products(data, calib, calib_dir)
    pair_selection = select_camera_pairs(calib)
    pair_report = calib_dir / "pair_selection_report.json"
    save_pair_selection_report(pair_selection, pair_report)
    reference_images = {idx: path for idx, path in enumerate(data["image_paths"])}
    masks = generate_masks_from_calibration(calib, reference_images=reference_images, output_dir=mask_dir)
    summary = {
        "case_root": str(case_root),
        "calibration_dir": str(calib_dir),
        "mask_dir": str(mask_dir),
        "calibration_outputs": calibration_outputs,
        "pair_selection_report": str(pair_report),
        "selected_pairs": [[a, b] for a, b in pair_selection.pair_names],
        "num_masks": len(masks),
        "mask_pixels": {mask.camera_label: int(mask.mask.sum()) for mask in masks},
    }
    (result_root / "cylinder_multiview_mask_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
