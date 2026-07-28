"""Generate physically consistent synthetic stereo chessboard calibration pairs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
LEFT_DIR = ROOT / "left"
RIGHT_DIR = ROOT / "right"

INNER_COLS = 9
INNER_ROWS = 6
SQUARE_SIZE_MM = 25.0
SQUARE_PIXELS = 64
IMAGE_SIZE = (1280, 960)
K_LEFT = np.array([[900.0, 0.0, 640.0], [0.0, 910.0, 480.0], [0.0, 0.0, 1.0]], dtype=np.float64)
K_RIGHT = np.array([[905.0, 0.0, 636.0], [0.0, 908.0, 482.0], [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_LEFT = np.zeros(5, dtype=np.float64)
DIST_RIGHT = np.zeros(5, dtype=np.float64)
BASELINE_MM = 240.0
CONVERGENCE_ANGLE_DEG = 30.0


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


R_LEFT_TO_RIGHT = rotation_y(CONVERGENCE_ANGLE_DEG)
RIGHT_CAMERA_CENTER_IN_LEFT_MM = np.array([BASELINE_MM, 0.0, 0.0], dtype=np.float64)
T_LEFT_TO_RIGHT = -R_LEFT_TO_RIGHT @ RIGHT_CAMERA_CENTER_IN_LEFT_MM


def make_board_texture() -> np.ndarray:
    squares_x = INNER_COLS + 1
    squares_y = INNER_ROWS + 1
    texture = np.full((squares_y * SQUARE_PIXELS, squares_x * SQUARE_PIXELS), 255, np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    texture,
                    (x * SQUARE_PIXELS, y * SQUARE_PIXELS),
                    ((x + 1) * SQUARE_PIXELS, (y + 1) * SQUARE_PIXELS),
                    20,
                    thickness=-1,
                )
    return texture


def board_outer_corners_mm() -> np.ndarray:
    width = (INNER_COLS + 1) * SQUARE_SIZE_MM
    height = (INNER_ROWS + 1) * SQUARE_SIZE_MM
    return np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=np.float32)


def left_board_pose(index: int) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(-20.0 + 2.8 * index)
    pitch = np.deg2rad(-17.0 + 5.0 * np.sin(index * 0.8))
    roll = np.deg2rad(-10.0 + 4.0 * np.cos(index * 0.6))
    rvec = np.array([pitch, yaw, roll], dtype=np.float64)
    tvec = np.array(
        [
            -125.0 + 14.0 * np.sin(index * 0.5),
            -88.0 + 12.0 * np.cos(index * 0.45),
            780.0 + 20.0 * index,
        ],
        dtype=np.float64,
    )
    return rvec, tvec


def right_board_pose(left_rvec: np.ndarray, left_tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_board_to_left, _ = cv2.Rodrigues(left_rvec)
    R_board_to_right = R_LEFT_TO_RIGHT @ R_board_to_left
    t_board_to_right = R_LEFT_TO_RIGHT @ left_tvec.reshape(3, 1) + T_LEFT_TO_RIGHT.reshape(3, 1)
    right_rvec, _ = cv2.Rodrigues(R_board_to_right)
    return right_rvec.reshape(3), t_board_to_right.reshape(3)


def render(texture: np.ndarray, dst: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = texture.shape
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    image = np.full((height, width), 245, np.uint8)
    warped = cv2.warpPerspective(texture, H, (width, height), borderValue=245)
    mask = cv2.warpPerspective(np.full_like(texture, 255), H, (width, height), borderValue=0)
    image[mask > 0] = warped[mask > 0]
    return cv2.GaussianBlur(image, (3, 3), 0.35)


def render_pairs(count: int = 16) -> None:
    LEFT_DIR.mkdir(parents=True, exist_ok=True)
    RIGHT_DIR.mkdir(parents=True, exist_ok=True)
    texture = make_board_texture()
    width, height = IMAGE_SIZE
    pairs = []

    for i in range(count):
        left_rvec, left_tvec = left_board_pose(i)
        right_rvec, right_tvec = right_board_pose(left_rvec, left_tvec)
        left_quad, _ = cv2.projectPoints(board_outer_corners_mm(), left_rvec, left_tvec, K_LEFT, DIST_LEFT)
        right_quad, _ = cv2.projectPoints(board_outer_corners_mm(), right_rvec, right_tvec, K_RIGHT, DIST_RIGHT)
        left_quad = left_quad.reshape(4, 2).astype(np.float32)
        right_quad = right_quad.reshape(4, 2).astype(np.float32)

        filename = f"{i + 1:02d}.bmp"
        cv2.imwrite(str(LEFT_DIR / filename), render(texture, left_quad, width, height))
        cv2.imwrite(str(RIGHT_DIR / filename), render(texture, right_quad, width, height))
        pairs.append(
            {
                "left": f"left/{filename}",
                "right": f"right/{filename}",
                "left_rvec_board_to_camera": left_rvec.round(8).tolist(),
                "left_tvec_board_to_camera_mm": left_tvec.round(8).tolist(),
                "right_rvec_board_to_camera": right_rvec.round(8).tolist(),
                "right_tvec_board_to_camera_mm": right_tvec.round(8).tolist(),
                "left_board_quad_px": left_quad.round(3).tolist(),
                "right_board_quad_px": right_quad.round(3).tolist(),
            }
        )

    meta = {
        "board_type": "chessboard",
        "inner_cols": INNER_COLS,
        "inner_rows": INNER_ROWS,
        "square_size_mm": SQUARE_SIZE_MM,
        "image_width": width,
        "image_height": height,
        "left_camera_matrix": K_LEFT.tolist(),
        "right_camera_matrix": K_RIGHT.tolist(),
        "left_distortion": DIST_LEFT.tolist(),
        "right_distortion": DIST_RIGHT.tolist(),
        "baseline_mm": BASELINE_MM,
        "convergence_angle_deg": CONVERGENCE_ANGLE_DEG,
        "right_camera_center_in_left_mm": RIGHT_CAMERA_CENTER_IN_LEFT_MM.tolist(),
        "R_left_to_right": R_LEFT_TO_RIGHT.tolist(),
        "T_left_to_right_mm": T_LEFT_TO_RIGHT.tolist(),
        "pairs": pairs,
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    render_pairs()
