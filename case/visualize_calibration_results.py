"""Create visual diagnostics for calibration cases and saved calibration results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PATTERN = (9, 6)
SQUARE_SIZE_MM = 25.0
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-9)


def make_object_points() -> np.ndarray:
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    points[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)
    points *= SQUARE_SIZE_MM
    return points


def detect_chessboard(path: Path) -> tuple[bool, np.ndarray | None, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    found, corners = cv2.findChessboardCorners(
        image,
        PATTERN,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return found, corners, image.shape[::-1]


def per_view_errors(object_points, image_points, rvecs, tvecs, K, distortion):
    errors = []
    projections = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, distortion)
        projected = projected.reshape(-1, 2)
        delta = projected - img.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        projections.append(projected)
    return errors, projections


def outlier_filter(errors: list[float], min_remaining: int = 6) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    median = float(np.median(values))
    mad_sigma = float(1.4826 * np.median(np.abs(values - median)))
    robust_threshold = max(median * 1.5, median + 2.0 * max(mad_sigma, 1e-6))
    threshold = min(0.20, robust_threshold)
    rejected = [int(i) for i, value in enumerate(values) if value > threshold]
    if len(values) - len(rejected) < min_remaining:
        rejected = []
    keep = [i for i in range(len(values)) if i not in rejected]
    return {
        "threshold_px": threshold,
        "rejected_indices_zero_based": rejected,
        "rejected_views_one_based": [i + 1 for i in rejected],
        "kept_indices_zero_based": keep,
        "applied": bool(rejected),
    }


def stereo_outlier_filter(left_errors: list[float], right_errors: list[float], min_remaining: int = 6) -> dict:
    left = np.asarray(left_errors, dtype=np.float64)
    right = np.asarray(right_errors, dtype=np.float64)
    pair = np.sqrt((left * left + right * right) * 0.5)
    pair_filter = outlier_filter(pair.tolist(), min_remaining=0)

    diff = np.abs(left - right)
    diff_median = float(np.median(diff))
    diff_mad_sigma = float(1.4826 * np.median(np.abs(diff - diff_median)))
    diff_threshold = min(0.12, max(0.05, diff_median + 2.0 * max(diff_mad_sigma, 1e-6)))
    ratio = np.maximum(left, right) / np.maximum(np.minimum(left, right), 1e-9)
    ratio_threshold = 2.0

    reasons: dict[int, list[str]] = {}
    for i, value in enumerate(pair):
        if value > pair_filter["threshold_px"]:
            reasons.setdefault(i, []).append("pair_error")
    for i, value in enumerate(diff):
        if value > diff_threshold:
            reasons.setdefault(i, []).append("left_right_abs_diff")
    for i, value in enumerate(ratio):
        if value > ratio_threshold and diff[i] > 0.03:
            reasons.setdefault(i, []).append("left_right_ratio")

    rejected = sorted(reasons)
    if len(pair) - len(rejected) < min_remaining:
        rejected = []
        reasons = {}
    keep = [i for i in range(len(pair)) if i not in rejected]
    return {
        "pair_threshold_px": pair_filter["threshold_px"],
        "left_right_diff_threshold_px": diff_threshold,
        "left_right_ratio_threshold": ratio_threshold,
        "pair_errors_px": pair.tolist(),
        "left_right_abs_diff_px": diff.tolist(),
        "left_right_ratio": ratio.tolist(),
        "rejection_reasons": {str(i + 1): reasons[i] for i in rejected},
        "rejected_indices_zero_based": rejected,
        "rejected_views_one_based": [i + 1 for i in rejected],
        "kept_indices_zero_based": keep,
        "applied": bool(rejected),
    }


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    rotation_delta, _ = cv2.Rodrigues(R_est @ R_gt.T)
    return float(np.linalg.norm(rotation_delta) * 180.0 / np.pi)


def accept_filtered_result(initial: dict, filtered: dict, has_extrinsic_gt: bool = False) -> bool:
    rms_gain = (initial["rms"] - filtered["rms"]) / max(initial["rms"], 1e-12)
    if rms_gain < 0.10:
        return False
    if has_extrinsic_gt:
        if filtered["rotation_error_deg"] > initial["rotation_error_deg"] + 0.05:
            return False
        if filtered["translation_error_mm"] > initial["translation_error_mm"] + 0.5:
            return False
    return True


def subset(items: list, indices: list[int]) -> list:
    return [items[i] for i in indices]


def calibrate_mono_points(object_points, image_points, image_size):
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
        criteria=CALIB_CRITERIA,
    )
    errors, projections = per_view_errors(object_points, image_points, rvecs, tvecs, K, distortion)
    return rms, K, distortion, rvecs, tvecs, errors, projections


def draw_overlay(image_path: Path, detected: np.ndarray, projected: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    detected = detected.reshape(-1, 2)
    for point in projected:
        cv2.drawMarker(
            image,
            tuple(np.round(point).astype(int)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
    for point in detected:
        cv2.circle(image, tuple(np.round(point).astype(int)), 4, (0, 220, 0), -1)
    cv2.imwrite(str(out_path), image)


def plot_errors(errors: list[float], title: str, out_path: Path) -> None:
    x = np.arange(1, len(errors) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.bar(x, errors, color="#3178c6")
    ax.axhline(np.mean(errors), color="#d33f49", linewidth=1.5, label=f"mean {np.mean(errors):.4f}px")
    ax.set_title(title)
    ax.set_xlabel("View")
    ax.set_ylabel("RMSE (px)")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_text_panel(lines: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.axis("off")
    y = 0.92
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        size = 13 if i == 0 else 10.5
        ax.text(0.04, y, line, fontsize=size, fontweight=weight, family="monospace", va="top")
        y -= 0.085
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def board_outer_corners() -> np.ndarray:
    width = (PATTERN[0] + 1) * SQUARE_SIZE_MM
    height = (PATTERN[1] + 1) * SQUARE_SIZE_MM
    return np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0], [0, 0, 0]], dtype=np.float64)


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_frame(ax, R: np.ndarray, t: np.ndarray, scale: float, label: str) -> None:
    origin = t.reshape(3)
    colors = ["#d33f49", "#2a9d8f", "#3178c6"]
    for axis in range(3):
        end = origin + R[:, axis] * scale
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=colors[axis], linewidth=2.0)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="black", s=18)
    ax.text(origin[0], origin[1], origin[2], label, fontsize=8)


def transformed_board_corners(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    corners = board_outer_corners()
    return (R @ corners.T + tvec.reshape(3, 1)).T


def plot_pose_distribution(
    board_rvecs,
    board_tvecs,
    out_path: Path,
    title: str,
    cameras: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    cameras = cameras or [("cam0", np.eye(3), np.zeros(3))]
    for label, R_cam, C_cam in cameras:
        draw_frame(ax, R_cam, C_cam.reshape(3), 65.0, label)
    for i, (rvec, tvec) in enumerate(zip(board_rvecs, board_tvecs), start=1):
        corners = transformed_board_corners(np.asarray(rvec), np.asarray(tvec))
        color = plt.cm.viridis((i - 1) / max(len(board_rvecs) - 1, 1))
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=1.3)
        center = corners[:4].mean(axis=0)
        if i in {1, len(board_rvecs)}:
            ax.text(center[0], center[1], center[2], f"board {i}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=-68, azim=-90)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def calibrate_mono_case():
    case_dir = ROOT / "mono_calibration"
    paths = sorted((case_dir / "images").glob("*.bmp"))
    obj_template = make_object_points()
    object_points = []
    image_points = []
    image_paths = []
    image_size = None
    for path in paths:
        found, corners, size = detect_chessboard(path)
        if found:
            object_points.append(obj_template.copy())
            image_points.append(corners)
            image_paths.append(path)
            image_size = size
    initial = calibrate_mono_points(object_points, image_points, image_size)
    filtering = outlier_filter(initial[5])
    if filtering["applied"]:
        keep = filtering["kept_indices_zero_based"]
        candidate_object_points = subset(object_points, keep)
        candidate_image_points = subset(image_points, keep)
        candidate = calibrate_mono_points(candidate_object_points, candidate_image_points, image_size)
        filtering["accepted"] = accept_filtered_result({"rms": float(initial[0])}, {"rms": float(candidate[0])})
        if filtering["accepted"]:
            object_points = candidate_object_points
            image_points = candidate_image_points
            image_paths = subset(image_paths, keep)
            rms, K, distortion, rvecs, tvecs, errors, projections = candidate
        else:
            rms, K, distortion, rvecs, tvecs, errors, projections = initial
    else:
        filtering["accepted"] = False
        rms, K, distortion, rvecs, tvecs, errors, projections = initial
    return {
        "paths": image_paths,
        "image_points": image_points,
        "projections": projections,
        "errors": errors,
        "rms": float(rms),
        "K": K,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "filtering": filtering,
        "initial_errors": initial[5],
        "initial_rms": float(initial[0]),
    }


def visualize_mono() -> None:
    case_dir = ROOT / "mono_calibration"
    out_dir = case_dir / "visualization"
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    result = calibrate_mono_case()

    for path, detected, projected in zip(result["paths"], result["image_points"], result["projections"]):
        draw_overlay(path, detected, projected, overlay_dir / f"{path.stem}_overlay.png")
    plot_errors(result["errors"], "Mono Calibration Reprojection Error", out_dir / "reprojection_errors.png")
    plot_pose_distribution(result["rvecs"], result["tvecs"], out_dir / "camera_board_distribution.png", "Mono Camera and Board Poses")
    K = result["K"]
    distortion = result["distortion"].ravel()
    lines = [
        "Mono Calibration Summary",
        f"Detected views: {len(result['paths'])}",
        f"RMS: {result['rms']:.6f} px",
        f"Mean per-view RMSE: {np.mean(result['errors']):.6f} px",
        f"Max per-view RMSE: {np.max(result['errors']):.6f} px",
        f"Filtering applied: {result['filtering']['applied']} rejected={result['filtering']['rejected_views_one_based']}",
        f"fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}",
        "distortion: " + ", ".join(f"{v:.6g}" for v in distortion[:5]),
        "Overlay legend: green = detected corner, red cross = reprojected corner",
    ]
    draw_text_panel(lines, out_dir / "summary.png")
    summary = {
        "rms_px": result["rms"],
        "mean_per_view_rmse_px": float(np.mean(result["errors"])),
        "max_per_view_rmse_px": float(np.max(result["errors"])),
        "overlay_count": len(result["paths"]),
        "filtering": result["filtering"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def calibrate_stereo_case():
    case_dir = ROOT / "stereo_calibartion"
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    left_paths = sorted((case_dir / "left").glob("*.bmp"))
    right_paths = sorted((case_dir / "right").glob("*.bmp"))
    obj_template = make_object_points()
    object_points = []
    left_points = []
    right_points = []
    used_left = []
    used_right = []
    image_size = None
    for left_path, right_path in zip(left_paths, right_paths):
        left_found, left_corners, size = detect_chessboard(left_path)
        right_found, right_corners, _ = detect_chessboard(right_path)
        if left_found and right_found:
            object_points.append(obj_template.copy())
            left_points.append(left_corners)
            right_points.append(right_corners)
            used_left.append(left_path)
            used_right.append(right_path)
            image_size = size

    left_initial = calibrate_mono_points(object_points, left_points, image_size)
    right_initial = calibrate_mono_points(object_points, right_points, image_size)
    pair_initial_errors = [
        float(np.sqrt((l * l + r * r) * 0.5)) for l, r in zip(left_initial[5], right_initial[5])
    ]
    filtering = stereo_outlier_filter(left_initial[5], right_initial[5])

    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    R_gt = np.asarray(meta["R_left_to_right"], dtype=np.float64)
    T_gt = np.asarray(meta["T_left_to_right_mm"], dtype=np.float64).reshape(3, 1)

    def calibrate_subset(obj_subset, left_subset, right_subset):
        left_rms, K_left, d_left, rv_left, tv_left, left_errors, left_projections = calibrate_mono_points(
            obj_subset, left_subset, image_size
        )
        right_rms, K_right, d_right, rv_right, tv_right, right_errors, right_projections = calibrate_mono_points(
            obj_subset, right_subset, image_size
        )
        stereo_rms, K_left, d_left, K_right, d_right, R, T, E, F = cv2.stereoCalibrate(
            obj_subset,
            left_subset,
            right_subset,
            K_left.copy(),
            d_left.copy(),
            K_right.copy(),
            d_right.copy(),
            image_size,
            flags=stereo_flags,
            criteria=CALIB_CRITERIA,
        )
        return {
            "left_rms": left_rms,
            "right_rms": right_rms,
            "stereo_rms": stereo_rms,
            "K_left": K_left,
            "K_right": K_right,
            "R": R,
            "T": T,
            "d_left": d_left,
            "d_right": d_right,
            "rv_left": rv_left,
            "tv_left": tv_left,
            "rv_right": rv_right,
            "tv_right": tv_right,
            "left_errors": left_errors,
            "right_errors": right_errors,
            "left_projections": left_projections,
            "right_projections": right_projections,
            "rotation_error_deg": rotation_error_deg(R, R_gt),
            "translation_error_mm": float(np.linalg.norm(T - T_gt)),
        }

    full = calibrate_subset(object_points, left_points, right_points)
    if filtering["applied"]:
        keep = filtering["kept_indices_zero_based"]
        candidate = calibrate_subset(subset(object_points, keep), subset(left_points, keep), subset(right_points, keep))
        filtering["accepted"] = accept_filtered_result(
            {
                "rms": float(full["stereo_rms"]),
                "rotation_error_deg": full["rotation_error_deg"],
                "translation_error_mm": full["translation_error_mm"],
            },
            {
                "rms": float(candidate["stereo_rms"]),
                "rotation_error_deg": candidate["rotation_error_deg"],
                "translation_error_mm": candidate["translation_error_mm"],
            },
            has_extrinsic_gt=True,
        )
        if filtering["accepted"]:
            chosen = candidate
            keep = filtering["kept_indices_zero_based"]
            used_left = subset(used_left, keep)
            used_right = subset(used_right, keep)
            left_points = subset(left_points, keep)
            right_points = subset(right_points, keep)
        else:
            chosen = full
    else:
        filtering["accepted"] = False
        chosen = full

    left_rms = chosen["left_rms"]
    right_rms = chosen["right_rms"]
    stereo_rms = chosen["stereo_rms"]
    K_left = chosen["K_left"]
    K_right = chosen["K_right"]
    R = chosen["R"]
    T = chosen["T"]
    rv_left = chosen["rv_left"]
    tv_left = chosen["tv_left"]
    rv_right = chosen["rv_right"]
    tv_right = chosen["tv_right"]
    left_errors = chosen["left_errors"]
    right_errors = chosen["right_errors"]
    left_projections = chosen["left_projections"]
    right_projections = chosen["right_projections"]
    return {
        "left_paths": used_left,
        "right_paths": used_right,
        "left_points": left_points,
        "right_points": right_points,
        "left_projections": left_projections,
        "right_projections": right_projections,
        "left_errors": left_errors,
        "right_errors": right_errors,
        "left_rms": float(left_rms),
        "right_rms": float(right_rms),
        "stereo_rms": float(stereo_rms),
        "K_left": K_left,
        "K_right": K_right,
        "R": R,
        "T": T,
        "left_rvecs": rv_left,
        "left_tvecs": tv_left,
        "right_rvecs": rv_right,
        "right_tvecs": tv_right,
        "filtering": filtering,
        "initial_pair_errors": pair_initial_errors,
    }


def draw_stereo_pair(left_path: Path, right_path: Path, left_points: np.ndarray, right_points: np.ndarray, out_path: Path) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    canvas = np.hstack([left, right])
    offset = left.shape[1]
    left_points = left_points.reshape(-1, 2)
    right_points = right_points.reshape(-1, 2)
    indices = np.linspace(0, len(left_points) - 1, 12, dtype=int)
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))[:, :3] * 255
    for idx, color in zip(indices, colors):
        color_bgr = tuple(int(v) for v in color[::-1])
        p_left = tuple(np.round(left_points[idx]).astype(int))
        p_right = tuple(np.round(right_points[idx] + np.array([offset, 0])).astype(int))
        cv2.circle(canvas, p_left, 5, color_bgr, -1)
        cv2.circle(canvas, p_right, 5, color_bgr, -1)
        cv2.line(canvas, p_left, p_right, color_bgr, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def plot_stereo_errors(left_errors: list[float], right_errors: list[float], out_path: Path) -> None:
    x = np.arange(1, len(left_errors) + 1)
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=140)
    ax.bar(x - width / 2, left_errors, width, label="left", color="#3178c6")
    ax.bar(x + width / 2, right_errors, width, label="right", color="#d97706")
    ax.set_title("Stereo Mono-View Reprojection Error")
    ax.set_xlabel("Pair")
    ax.set_ylabel("RMSE (px)")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def result_object_points(board: dict) -> np.ndarray:
    rows = int(board["rows"])
    cols = int(board["cols"])
    spacing = float(board["spacing"])
    points = []
    for r in range(rows):
        for c in range(cols):
            points.append((c * spacing, r * spacing, 0.0))
    return np.asarray(points, dtype=np.float32)


def result_project_errors(
    obj: np.ndarray,
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, distortion)
    projected = projected.reshape(-1, 2)
    residual = projected - img
    return projected, np.linalg.norm(residual, axis=1)


def result_solve_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, distortion: np.ndarray):
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, rvec, tvec.reshape(3)


def plot_result_errors(rows: list[dict], left_errors, right_errors, combined_errors, out_dir: Path) -> None:
    pair_ids = np.asarray([r["pair"] for r in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(pair_ids, left_errors, "o-", label="left")
    ax.plot(pair_ids, right_errors, "o-", label="right")
    ax.plot(pair_ids, combined_errors, "o-", label="combined")
    ax.set_xlabel("Calibration pair")
    ax.set_ylabel("RMS reprojection error (px)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reprojection_error_by_pair.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.boxplot([left_errors, right_errors, combined_errors], tick_labels=["left", "right", "combined"], showmeans=True)
    ax.set_ylabel("RMS reprojection error (px)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "reprojection_error_boxplot.png")
    plt.close(fig)


def draw_result_error_heatmap(points, errors, width: int, height: int, title: str, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float64)
    err = np.asarray(errors, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=err, s=8, cmap="turbo")
    ax.set_title(title)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="error (px)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_result_camera_board_distribution(board_centers, board_axes, right_center, out_dir: Path) -> None:
    centers = np.asarray(board_centers, dtype=np.float64)
    right_center = np.asarray(right_center, dtype=np.float64).reshape(3)
    axis_len = 25.0
    all_pts = np.vstack([centers, np.zeros((1, 3)), np.asarray(right_center).reshape(1, 3)])
    span = np.max(np.ptp(all_pts, axis=0))
    mid = np.mean([np.min(all_pts, axis=0), np.max(all_pts, axis=0)], axis=0)
    half = max(span * 0.55, 1.0)
    limits = [(mid[i] - half, mid[i] + half) for i in range(3)]

    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.scatter([0.0], [0.0], [0.0], c="tab:blue", s=60, label="left camera")
    ax3d.scatter([right_center[0]], [right_center[1]], [right_center[2]], c="tab:red", s=60, label="right camera")
    ax3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=np.arange(len(centers)), cmap="viridis", s=22, label="board centers")
    for center, R in zip(board_centers, board_axes):
        center = np.asarray(center)
        for axis, color in zip(range(3), ["r", "g", "b"]):
            direction = R[:, axis] * axis_len
            ax3d.plot(
                [center[0], center[0] + direction[0]],
                [center[1], center[1] + direction[1]],
                [center[2], center[2] + direction[2]],
                color=color,
                alpha=0.35,
                linewidth=0.8,
            )
    ax3d.set_title("Perspective")
    ax3d.set_xlim(limits[0])
    ax3d.set_ylim(limits[1])
    ax3d.set_zlim(limits[2])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.view_init(elev=22, azim=-62)
    ax3d.legend(loc="upper left", fontsize=8)

    def draw_projection(ax, dims: tuple[int, int], title: str, xlabel: str, ylabel: str) -> None:
        i, j = dims
        ax.scatter([0.0], [0.0], c="tab:blue", s=55, label="left camera")
        ax.scatter([right_center[i]], [right_center[j]], c="tab:red", s=55, label="right camera")
        ax.scatter(centers[:, i], centers[:, j], c=np.arange(len(centers)), cmap="viridis", s=20, label="board centers")
        for center, R in zip(board_centers, board_axes):
            center = np.asarray(center)
            for axis, color in zip(range(3), ["r", "g", "b"]):
                direction = R[:, axis] * axis_len
                ax.plot(
                    [center[i], center[i] + direction[i]],
                    [center[j], center[j] + direction[j]],
                    color=color,
                    alpha=0.35,
                    linewidth=0.8,
                )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(limits[i])
        ax.set_ylim(limits[j])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    draw_projection(fig.add_subplot(222), (0, 1), "XY view", "X", "Y")
    draw_projection(fig.add_subplot(223), (0, 2), "XZ view", "X", "Z")
    draw_projection(fig.add_subplot(224), (1, 2), "YZ view", "Y", "Z")
    fig.tight_layout()
    fig.savefig(out_dir / "camera_board_distribution.png")
    plt.close(fig)


def visualize_saved_stereo_calibration_result(calibration_json: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(calibration_json.read_text(encoding="utf-8"))
    obj = result_object_points(data["board"])
    K_l = np.asarray(data["left"]["K"], dtype=np.float64)
    K_r = np.asarray(data["right"]["K"], dtype=np.float64)
    dist_l = np.asarray(data["left"].get("distortion", []), dtype=np.float64)
    dist_r = np.asarray(data["right"].get("distortion", []), dtype=np.float64)
    R_lr = np.asarray(data["R_lr"], dtype=np.float64)
    t_lr = np.asarray(data["t_lr"], dtype=np.float64).reshape(3)
    width = int(data["left"]["image_width"])
    height = int(data["left"]["image_height"])

    left_detections = data["left_detections"]
    right_detections = data["right_detections"]
    kept_indices = data.get("kept_pair_indices", [])
    if data.get("outlier_rejection_applied") and kept_indices:
        selected_pairs = [
            (int(index) + 1, left_detections[int(index)], right_detections[int(index)])
            for index in kept_indices
        ]
    else:
        selected_pairs = [
            (pair_id, left_det, right_det)
            for pair_id, (left_det, right_det) in enumerate(zip(left_detections, right_detections), start=1)
        ]

    rows = []
    board_centers = []
    board_axes = []
    left_heat_pts = []
    left_heat_err = []
    right_heat_pts = []
    right_heat_err = []
    stereo_right_heat_pts = []
    stereo_right_heat_err = []

    for pair_id, left_det, right_det in selected_pairs:
        left_img = np.asarray(left_det["image_points"], dtype=np.float64)
        right_img = np.asarray(right_det["image_points"], dtype=np.float64)
        R_l, rvec_l, tvec_l = result_solve_pose(obj, left_img, K_l, dist_l)
        _, rvec_r, tvec_r = result_solve_pose(obj, right_img, K_r, dist_r)
        _, err_l = result_project_errors(obj, left_img, rvec_l, tvec_l, K_l, dist_l)
        _, err_r = result_project_errors(obj, right_img, rvec_r, tvec_r, K_r, dist_r)

        R_stereo = R_lr @ R_l
        t_stereo = R_lr @ tvec_l + t_lr
        rvec_stereo, _ = cv2.Rodrigues(R_stereo)
        _, err_stereo_r = result_project_errors(obj, right_img, rvec_stereo, t_stereo, K_r, dist_r)

        center_obj = np.mean(obj.astype(np.float64), axis=0)
        board_centers.append(R_l @ center_obj + tvec_l)
        board_axes.append(R_l)

        left_heat_pts.append(left_img)
        left_heat_err.append(err_l)
        right_heat_pts.append(right_img)
        right_heat_err.append(err_r)
        stereo_right_heat_pts.append(right_img)
        stereo_right_heat_err.append(err_stereo_r)

        rows.append(
            {
                "pair": pair_id,
                "left_image": left_det["image_path"],
                "right_image": right_det["image_path"],
                "left_pnp_rms_px": float(np.sqrt(np.mean(err_l**2))),
                "right_pnp_rms_px": float(np.sqrt(np.mean(err_r**2))),
                "stereo_right_rms_px": float(np.sqrt(np.mean(err_stereo_r**2))),
                "left_pnp_max_px": float(np.max(err_l)),
                "right_pnp_max_px": float(np.max(err_r)),
                "stereo_right_max_px": float(np.max(err_stereo_r)),
                "board_center_x": float(board_centers[-1][0]),
                "board_center_y": float(board_centers[-1][1]),
                "board_center_z": float(board_centers[-1][2]),
            }
        )

    csv_path = out_dir / "reprojection_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if data.get("per_pair_left_errors") and len(data["per_pair_left_errors"]) == len(rows):
        left_rms = np.asarray(data["per_pair_left_errors"], dtype=np.float64)
        right_rms = np.asarray(data["per_pair_right_errors"], dtype=np.float64)
        combined_rms = np.asarray(data["per_pair_errors"], dtype=np.float64)
    else:
        left_rms = np.asarray([r["left_pnp_rms_px"] for r in rows])
        right_rms = np.asarray([r["right_pnp_rms_px"] for r in rows])
        combined_rms = np.sqrt(0.5 * (left_rms * left_rms + right_rms * right_rms))

    plot_result_errors(rows, left_rms, right_rms, combined_rms, out_dir)
    draw_result_error_heatmap(
        np.vstack(left_heat_pts),
        np.concatenate(left_heat_err),
        width,
        height,
        "Left mono-PnP reprojection error",
        out_dir / "reprojection_error_heatmap_left.png",
    )
    draw_result_error_heatmap(
        np.vstack(right_heat_pts),
        np.concatenate(right_heat_err),
        width,
        height,
        "Right mono-PnP reprojection error",
        out_dir / "reprojection_error_heatmap_right.png",
    )
    draw_result_error_heatmap(
        np.vstack(stereo_right_heat_pts),
        np.concatenate(stereo_right_heat_err),
        width,
        height,
        "Right reprojection error from left pose and stereo extrinsic",
        out_dir / "reprojection_error_heatmap_stereo_right.png",
    )

    right_center = -R_lr.T @ t_lr
    draw_result_camera_board_distribution(board_centers, board_axes, right_center, out_dir)
    summary = {
        "pairs": len(rows),
        "total_detected_pairs": len(left_detections),
        "outlier_rejection_applied": bool(data.get("outlier_rejection_applied", False)),
        "kept_pair_indices": [int(v) for v in data.get("kept_pair_indices", [])],
        "rejected_pair_indices": [int(v) for v in data.get("rejected_pair_indices", [])],
        "rejection_reasons": [str(v) for v in data.get("rejection_reasons", [])],
        "initial_opencv_stereo_rms_px": float(data.get("initial_rms_error", 0.0)),
        "opencv_stereo_rms_px": float(data["rms_error"]),
        "left_stereo_per_view_rms_mean_px": float(np.mean(left_rms)),
        "left_stereo_per_view_rms_max_px": float(np.max(left_rms)),
        "right_stereo_per_view_rms_mean_px": float(np.mean(right_rms)),
        "right_stereo_per_view_rms_max_px": float(np.max(right_rms)),
        "combined_stereo_per_view_rms_mean_px": float(np.mean(combined_rms)),
        "combined_stereo_per_view_rms_max_px": float(np.max(combined_rms)),
        "right_camera_center_in_left_frame": right_center.tolist(),
        "baseline": float(np.linalg.norm(right_center)),
        "left_K": K_l.tolist(),
        "right_K": K_r.tolist(),
        "left_distortion": dist_l.tolist(),
        "right_distortion": dist_r.tolist(),
    }
    (out_dir / "calibration_visualization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def visualize_stereo() -> None:
    case_dir = ROOT / "stereo_calibartion"
    out_dir = case_dir / "visualization"
    left_overlay_dir = out_dir / "left_overlays"
    right_overlay_dir = out_dir / "right_overlays"
    left_overlay_dir.mkdir(parents=True, exist_ok=True)
    right_overlay_dir.mkdir(parents=True, exist_ok=True)
    result = calibrate_stereo_case()

    for path, detected, projected in zip(result["left_paths"], result["left_points"], result["left_projections"]):
        draw_overlay(path, detected, projected, left_overlay_dir / f"{path.stem}_overlay.png")
    for path, detected, projected in zip(result["right_paths"], result["right_points"], result["right_projections"]):
        draw_overlay(path, detected, projected, right_overlay_dir / f"{path.stem}_overlay.png")
    plot_stereo_errors(result["left_errors"], result["right_errors"], out_dir / "reprojection_errors.png")
    R_right_to_left = result["R"].T
    C_right_in_left = -result["R"].T @ result["T"]
    plot_pose_distribution(
        result["left_rvecs"],
        result["left_tvecs"],
        out_dir / "camera_board_distribution.png",
        "Stereo Camera and Board Poses",
        cameras=[("left", np.eye(3), np.zeros(3)), ("right", R_right_to_left, C_right_in_left.reshape(3))],
    )
    draw_stereo_pair(
        result["left_paths"][0],
        result["right_paths"][0],
        result["left_points"][0],
        result["right_points"][0],
        out_dir / "pair_01_corner_matches.png",
    )

    K_left = result["K_left"]
    K_right = result["K_right"]
    T = result["T"].ravel()
    lines = [
        "Stereo Calibration Summary",
        f"Detected pairs: {len(result['left_paths'])}",
        f"Left RMS: {result['left_rms']:.6f} px",
        f"Right RMS: {result['right_rms']:.6f} px",
        f"Stereo RMS: {result['stereo_rms']:.6f} px",
        f"Filtering applied: {result['filtering']['applied']} rejected={result['filtering']['rejected_views_one_based']}",
        f"Baseline norm: {np.linalg.norm(T):.6f} mm",
        f"T_left_to_right = [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}] mm",
        f"Left fx={K_left[0,0]:.3f}, fy={K_left[1,1]:.3f}, cx={K_left[0,2]:.3f}, cy={K_left[1,2]:.3f}",
        f"Right fx={K_right[0,0]:.3f}, fy={K_right[1,1]:.3f}, cx={K_right[0,2]:.3f}, cy={K_right[1,2]:.3f}",
        "Overlay legend: green = detected corner, red cross = reprojected corner",
    ]
    draw_text_panel(lines, out_dir / "summary.png")
    summary = {
        "left_rms_px": result["left_rms"],
        "right_rms_px": result["right_rms"],
        "stereo_rms_px": result["stereo_rms"],
        "left_mean_per_view_rmse_px": float(np.mean(result["left_errors"])),
        "right_mean_per_view_rmse_px": float(np.mean(result["right_errors"])),
        "baseline_norm_mm": float(np.linalg.norm(T)),
        "left_overlay_count": len(result["left_paths"]),
        "right_overlay_count": len(result["right_paths"]),
        "filtering": result["filtering"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=ROOT / "stereo_DIC" / "plate_center_load" / "result" / "calibration" / "stereo_calibration.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "stereo_DIC" / "plate_center_load" / "result" / "calibration",
    )
    parser.add_argument("--legacy-synthetic", action="store_true", help="also regenerate old synthetic calibration visuals")
    args = parser.parse_args()

    if args.legacy_synthetic:
        visualize_mono()
        visualize_stereo()
        print("wrote", ROOT / "mono_calibration" / "visualization")
        print("wrote", ROOT / "stereo_calibartion" / "visualization")

    summary = visualize_saved_stereo_calibration_result(args.calibration_json, args.out_dir)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out_dir)
