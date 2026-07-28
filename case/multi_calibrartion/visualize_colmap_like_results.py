"""Visualize COLMAP-like sparse multiview calibration results for this case.

This follows the plotting style used by Multi-DIC's COLMAP product exporter:

- sparse_scene.png: sparse 3D points and camera poses.
- camera_observations_3d.png: per-camera observed sparse points in 3D.
- camera_observations_2d.png: per-camera projected sparse observations on images.

The copied case does not include a completed COLMAP sparse model, so this script
uses the simulator camera metadata and creates a deterministic sparse cylinder
surface as a test reconstruction target.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "calibrate_images" / "chessboard_meta.json"
OUT_DIR = ROOT / "visualization"


def natural_key(name: str) -> tuple:
    head = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return (head, int(digits) if digits else -1)


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t.reshape(3)


def axis_limits(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.vstack([points, centers]) if len(points) else centers
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    pad = 0.08 * max(float(np.max(hi - lo)), 1.0)
    return lo - pad, hi + pad


def apply_limits(ax, limits: tuple[np.ndarray, np.ndarray]) -> None:
    lo, hi = limits
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass


def make_sparse_cylinder_points(config: dict, count_theta: int = 180, count_y: int = 80) -> np.ndarray:
    radius = float(config.get("cylinder_radius", 80.0))
    height = float(config.get("cylinder_height", 120.0))
    theta = np.linspace(0.0, 2.0 * np.pi, count_theta, endpoint=False)
    y = np.linspace(-height / 2.0, height / 2.0, count_y)
    tt, yy = np.meshgrid(theta, y)
    points = np.column_stack(
        [
            radius * np.cos(tt.ravel()),
            yy.ravel(),
            radius * np.sin(tt.ravel()),
        ]
    )
    return points.astype(np.float64)


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_cam = R @ points.T + t.reshape(3, 1)
    z = p_cam[2]
    valid_z = z > 1e-9
    uv = np.column_stack(
        [
            K[0, 0] * (p_cam[0] / z) + K[0, 2],
            K[1, 1] * (p_cam[1] / z) + K[1, 2],
        ]
    )
    return uv, valid_z


def build_colmap_like_products(meta: dict, points: np.ndarray) -> dict:
    cameras = sorted(meta["cameras"], key=lambda item: natural_key(item["camera_name"]))
    cam_names = [cam["camera_name"] for cam in cameras]
    K_list = [np.asarray(cam["K"], dtype=np.float64) for cam in cameras]
    R_list = [np.asarray(cam["R_world_to_camera"], dtype=np.float64) for cam in cameras]
    t_list = [np.asarray(cam["t_world_to_camera"], dtype=np.float64).reshape(3) for cam in cameras]
    centers = np.asarray([camera_center(R, t) for R, t in zip(R_list, t_list)], dtype=np.float64)
    image_paths = [ROOT / "images" / name / "001.bmp" for name in cam_names]

    cam_indices = []
    point_indices = []
    uv_values = []
    width = int(meta["config"].get("image_width", 1440))
    height = int(meta["config"].get("image_height", 1080))
    radius = float(meta["config"].get("cylinder_radius", 80.0))

    normals = points.copy()
    normals[:, 1] = 0.0
    normal_norm = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_norm[:, None], 1e-12)

    for cam_idx, (K, R, t, center) in enumerate(zip(K_list, R_list, t_list, centers)):
        uv, valid_z = project_points(points, K, R, t)
        view_vec = center.reshape(1, 3) - points
        view_vec = view_vec / np.maximum(np.linalg.norm(view_vec, axis=1, keepdims=True), 1e-12)
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
        "R": R_list,
        "t": t_list,
        "centers": centers,
        "points": points,
        "observations": observations,
        "radius": radius,
    }


def save_products(data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "colmap_like_sparse_model.npz",
        cam_names=np.asarray(data["cam_names"]),
        K=data["K"],
        R=np.asarray(data["R"]),
        t=np.asarray(data["t"]),
        camera_centers_world=data["centers"],
        points3D=data["points"],
        observation_cam_indices=data["observations"]["cam_indices"],
        observation_point_indices=data["observations"]["point_indices"],
        observation_uv=data["observations"]["uv"],
    )
    summary = {
        "num_cameras": len(data["cam_names"]),
        "num_sparse_points": int(len(data["points"])),
        "num_observations": int(len(data["observations"]["cam_indices"])),
        "per_camera_observations": {
            name: int(np.sum(data["observations"]["cam_indices"] == idx)) for idx, name in enumerate(data["cam_names"])
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_sparse_scene(data: dict, dpi: int = 160) -> Path:
    points = data["points"]
    centers = data["centers"]
    limits = axis_limits(points, centers)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sample_count = min(len(points), 30000)
    sample_idx = np.random.RandomState(0).choice(len(points), sample_count, replace=False)
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
    out_path = OUT_DIR / "sparse_scene.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_camera_observations_3d(data: dict, dpi: int = 160) -> Path:
    points = data["points"]
    centers = data["centers"]
    observations = data["observations"]
    limits = axis_limits(points, centers)
    fig = plt.figure(figsize=(16, 12))
    for idx, name in enumerate(data["cam_names"]):
        ax = fig.add_subplot(3, 4, idx + 1, projection="3d")
        mask = observations["cam_indices"] == idx
        point_indices = observations["point_indices"][mask]
        observed_points = points[point_indices] if len(point_indices) else np.zeros((0, 3))
        if len(observed_points):
            ax.scatter(observed_points[:, 0], observed_points[:, 1], observed_points[:, 2], s=2, c="tab:red", alpha=0.75)
        apply_limits(ax, limits)
        ax.set_title(f"{name}: {len(observed_points)} obs", fontsize=9)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    out_path = OUT_DIR / "camera_observations_3d.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_camera_observations_2d(data: dict, dpi: int = 160) -> Path:
    observations = data["observations"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes_flat = axes.ravel()
    for idx, name in enumerate(data["cam_names"]):
        ax = axes_flat[idx]
        image_path = data["image_paths"][idx]
        image = mpimg.imread(image_path)
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        mask = observations["cam_indices"] == idx
        uv = observations["uv"][mask]
        if len(uv):
            draw_count = min(len(uv), 1200)
            draw_idx = np.random.RandomState(idx).choice(len(uv), draw_count, replace=False)
            draw_uv = uv[draw_idx]
            ax.scatter(draw_uv[:, 0], draw_uv[:, 1], s=8, c="red", marker="o", linewidths=0, alpha=0.7)
        ax.set_title(f"{name}: {len(uv)} obs", fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
    for idx in range(len(data["cam_names"]), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.tight_layout()
    out_path = OUT_DIR / "camera_observations_2d.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    points = make_sparse_cylinder_points(meta["config"])
    data = build_colmap_like_products(meta, points)
    save_products(data)
    outputs = {
        "sparse_scene": str(plot_sparse_scene(data)),
        "camera_observations_3d": str(plot_camera_observations_3d(data)),
        "camera_observations_2d": str(plot_camera_observations_2d(data)),
    }
    (OUT_DIR / "visualization_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
