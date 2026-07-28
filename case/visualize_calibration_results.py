"""Create visual diagnostics for synthetic mono and stereo calibration cases."""

from __future__ import annotations

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
    visualize_mono()
    visualize_stereo()
    print("wrote", ROOT / "mono_calibration" / "visualization")
    print("wrote", ROOT / "stereo_calibartion" / "visualization")
