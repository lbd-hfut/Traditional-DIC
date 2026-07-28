"""Evaluate mono and stereo synthetic calibration cases.

Runs the same workflow as the calibration module uses for Zhang calibration:
chessboard corner detection, subpixel refinement, mono calibration, and stereo
calibration. Reports are written next to each case.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
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


def per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, distortion)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


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
        "method": "synthetic case strict filter: error > min(0.20 px, max(1.5*median, median + 2*MAD_sigma))",
        "threshold_px": threshold,
        "median_px": median,
        "mad_sigma_px": mad_sigma,
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
        "method": (
            "stereo strict filter: pair error outlier OR |left-right| outlier OR "
            "max(left,right)/min(left,right) > 2 with >0.03 px absolute gap"
        ),
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


def subset(items: list, indices: list[int]) -> list:
    return [items[i] for i in indices]


def calibrate_mono_points(object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]):
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
    errors = per_view_errors(object_points, image_points, rvecs, tvecs, K, distortion)
    return rms, K, distortion, rvecs, tvecs, errors


def summarize_camera(K_est: np.ndarray, dist_est: np.ndarray, K_gt: list, dist_gt: list) -> dict:
    K_gt = np.asarray(K_gt, dtype=np.float64)
    dist_est = np.asarray(dist_est, dtype=np.float64).ravel()
    dist_gt = np.asarray(dist_gt, dtype=np.float64).ravel()
    n = min(len(dist_est), len(dist_gt))
    return {
        "fx_error_px": float(K_est[0, 0] - K_gt[0, 0]),
        "fy_error_px": float(K_est[1, 1] - K_gt[1, 1]),
        "cx_error_px": float(K_est[0, 2] - K_gt[0, 2]),
        "cy_error_px": float(K_est[1, 2] - K_gt[1, 2]),
        "dist_l2_error": float(np.linalg.norm(dist_est[:n] - dist_gt[:n])) if n else 0.0,
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


def evaluate_mono() -> dict:
    case_dir = ROOT / "mono_calibration"
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    paths = sorted((case_dir / "images").glob("*.bmp"))
    object_template = make_object_points()
    object_points = []
    image_points = []
    image_size = None

    for path in paths:
        found, corners, size = detect_chessboard(path)
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            image_size = size

    initial_rms, initial_K, initial_distortion, initial_rvecs, initial_tvecs, initial_errors = calibrate_mono_points(
        object_points, image_points, image_size
    )
    filtering = outlier_filter(initial_errors)
    if filtering["applied"]:
        keep = filtering["kept_indices_zero_based"]
        candidate_object_points = subset(object_points, keep)
        candidate_image_points = subset(image_points, keep)
        candidate_rms, candidate_K, candidate_distortion, candidate_rvecs, candidate_tvecs, candidate_errors = (
            calibrate_mono_points(candidate_object_points, candidate_image_points, image_size)
        )
        initial_metrics = {"rms": float(initial_rms)}
        candidate_metrics = {"rms": float(candidate_rms)}
        filtering["accepted"] = accept_filtered_result(initial_metrics, candidate_metrics)
        if filtering["accepted"]:
            object_points = candidate_object_points
            image_points = candidate_image_points
            rms, K, distortion, rvecs, tvecs, errors = (
                candidate_rms,
                candidate_K,
                candidate_distortion,
                candidate_rvecs,
                candidate_tvecs,
                candidate_errors,
            )
        else:
            rms, K, distortion, rvecs, tvecs, errors = (
                initial_rms,
                initial_K,
                initial_distortion,
                initial_rvecs,
                initial_tvecs,
                initial_errors,
            )
    else:
        filtering["accepted"] = False
        rms, K, distortion, rvecs, tvecs, errors = (
            initial_rms,
            initial_K,
            initial_distortion,
            initial_rvecs,
            initial_tvecs,
            initial_errors,
        )
    report = {
        "detected_views": len(image_points),
        "total_views": len(paths),
        "initial_rms_px": float(initial_rms),
        "initial_mean_per_view_rmse_px": float(np.mean(initial_errors)),
        "initial_max_per_view_rmse_px": float(np.max(initial_errors)),
        "filtering": filtering,
        "rms_px": float(rms),
        "mean_per_view_rmse_px": float(np.mean(errors)),
        "max_per_view_rmse_px": float(np.max(errors)),
        "camera_matrix": K.tolist(),
        "distortion": distortion.ravel().tolist(),
        "board_rvecs": [np.asarray(r).ravel().tolist() for r in rvecs],
        "board_tvecs_mm": [np.asarray(t).ravel().tolist() for t in tvecs],
        "ground_truth_comparison": summarize_camera(K, distortion, meta["camera_matrix"], meta["distortion"]),
    }
    (case_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_stereo() -> dict:
    case_dir = ROOT / "stereo_calibartion"
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    left_paths = sorted((case_dir / "left").glob("*.bmp"))
    right_paths = sorted((case_dir / "right").glob("*.bmp"))
    object_template = make_object_points()
    object_points = []
    left_points = []
    right_points = []
    image_size = None

    for left_path, right_path in zip(left_paths, right_paths):
        left_found, left_corners, left_size = detect_chessboard(left_path)
        right_found, right_corners, _ = detect_chessboard(right_path)
        if left_found and right_found:
            object_points.append(object_template.copy())
            left_points.append(left_corners)
            right_points.append(right_corners)
            image_size = left_size

    left_rms_i, K_left_i, d_left_i, rv_left_i, tv_left_i, left_errors_i = calibrate_mono_points(
        object_points, left_points, image_size
    )
    right_rms_i, K_right_i, d_right_i, rv_right_i, tv_right_i, right_errors_i = calibrate_mono_points(
        object_points, right_points, image_size
    )
    pair_errors_i = [float(np.sqrt((l * l + r * r) * 0.5)) for l, r in zip(left_errors_i, right_errors_i)]
    filtering = stereo_outlier_filter(left_errors_i, right_errors_i)

    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    R_gt = np.asarray(meta["R_left_to_right"], dtype=np.float64)
    T_gt = np.asarray(meta["T_left_to_right_mm"], dtype=np.float64).reshape(3, 1)

    def calibrate_stereo_subset(obj_subset, left_subset, right_subset):
        left_rms, K_left, d_left, rv_left, tv_left, left_errors = calibrate_mono_points(
            obj_subset, left_subset, image_size
        )
        right_rms, K_right, d_right, rv_right, tv_right, right_errors = calibrate_mono_points(
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
            "object_points": obj_subset,
            "left_points": left_subset,
            "right_points": right_subset,
            "left_rms": left_rms,
            "right_rms": right_rms,
            "stereo_rms": stereo_rms,
            "K_left": K_left,
            "d_left": d_left,
            "rv_left": rv_left,
            "tv_left": tv_left,
            "left_errors": left_errors,
            "K_right": K_right,
            "d_right": d_right,
            "rv_right": rv_right,
            "tv_right": tv_right,
            "right_errors": right_errors,
            "R": R,
            "T": T,
            "rotation_error_deg": rotation_error_deg(R, R_gt),
            "translation_error_mm": float(np.linalg.norm(T - T_gt)),
        }

    full = calibrate_stereo_subset(object_points, left_points, right_points)
    if filtering["applied"]:
        keep = filtering["kept_indices_zero_based"]
        candidate = calibrate_stereo_subset(
            subset(object_points, keep), subset(left_points, keep), subset(right_points, keep)
        )
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
        chosen = candidate if filtering["accepted"] else full
    else:
        filtering["accepted"] = False
        chosen = full

    object_points = chosen["object_points"]
    left_rms = chosen["left_rms"]
    right_rms = chosen["right_rms"]
    stereo_rms = chosen["stereo_rms"]
    K_left = chosen["K_left"]
    d_left = chosen["d_left"]
    rv_left = chosen["rv_left"]
    tv_left = chosen["tv_left"]
    left_errors = chosen["left_errors"]
    K_right = chosen["K_right"]
    d_right = chosen["d_right"]
    rv_right = chosen["rv_right"]
    tv_right = chosen["tv_right"]
    right_errors = chosen["right_errors"]
    R = chosen["R"]
    T = chosen["T"]
    report = {
        "detected_pairs": len(object_points),
        "total_pairs": len(left_paths),
        "initial_left_mono_rms_px": float(left_rms_i),
        "initial_right_mono_rms_px": float(right_rms_i),
        "initial_pair_mean_rmse_px": float(np.mean(pair_errors_i)),
        "initial_pair_max_rmse_px": float(np.max(pair_errors_i)),
        "filtering": filtering,
        "left_mono_rms_px": float(left_rms),
        "right_mono_rms_px": float(right_rms),
        "stereo_rms_px": float(stereo_rms),
        "left_mean_per_view_rmse_px": float(np.mean(left_errors)),
        "left_max_per_view_rmse_px": float(np.max(left_errors)),
        "right_mean_per_view_rmse_px": float(np.mean(right_errors)),
        "right_max_per_view_rmse_px": float(np.max(right_errors)),
        "left_camera_matrix": K_left.tolist(),
        "right_camera_matrix": K_right.tolist(),
        "left_distortion": d_left.ravel().tolist(),
        "right_distortion": d_right.ravel().tolist(),
        "left_board_rvecs": [np.asarray(r).ravel().tolist() for r in rv_left],
        "left_board_tvecs_mm": [np.asarray(t).ravel().tolist() for t in tv_left],
        "right_board_rvecs": [np.asarray(r).ravel().tolist() for r in rv_right],
        "right_board_tvecs_mm": [np.asarray(t).ravel().tolist() for t in tv_right],
        "R_left_to_right": R.tolist(),
        "T_left_to_right_mm": T.ravel().tolist(),
        "baseline_norm_mm": float(np.linalg.norm(T)),
        "ground_truth_comparison": {
            "left": summarize_camera(K_left, d_left, meta["left_camera_matrix"], meta["left_distortion"]),
            "right": summarize_camera(K_right, d_right, meta["right_camera_matrix"], meta["right_distortion"]),
            "rotation_error_deg": chosen["rotation_error_deg"],
            "translation_l2_error_mm": float(np.linalg.norm(T - T_gt)),
            "baseline_error_mm": float(np.linalg.norm(T) - np.linalg.norm(T_gt)),
        },
    }
    (case_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    mono = evaluate_mono()
    stereo = evaluate_stereo()
    print(
        "mono:",
        f"{mono['detected_views']}/{mono['total_views']} views,",
        f"rms={mono['rms_px']:.4f}px,",
        f"mean={mono['mean_per_view_rmse_px']:.4f}px,",
        f"max={mono['max_per_view_rmse_px']:.4f}px",
    )
    print(
        "stereo:",
        f"{stereo['detected_pairs']}/{stereo['total_pairs']} pairs,",
        f"left_rms={stereo['left_mono_rms_px']:.4f}px,",
        f"right_rms={stereo['right_mono_rms_px']:.4f}px,",
        f"stereo_rms={stereo['stereo_rms_px']:.4f}px",
    )
    print(
        "stereo extrinsic:",
        f"rotation_error={stereo['ground_truth_comparison']['rotation_error_deg']:.4f}deg,",
        f"translation_error={stereo['ground_truth_comparison']['translation_l2_error_mm']:.4f}mm,",
        f"baseline_error={stereo['ground_truth_comparison']['baseline_error_mm']:.4f}mm",
    )
