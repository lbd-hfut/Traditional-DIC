"""Generate physically consistent synthetic mono chessboard calibration images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images"

INNER_COLS = 9
INNER_ROWS = 6
SQUARE_SIZE_MM = 25.0
SQUARE_PIXELS = 64
IMAGE_SIZE = (1280, 960)  # width, height
K = np.array([[900.0, 0.0, 640.0], [0.0, 910.0, 480.0], [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)


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


def board_pose(index: int) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(-22.0 + 3.0 * index)
    pitch = np.deg2rad(-18.0 + 5.0 * np.sin(index * 0.9))
    roll = np.deg2rad(-12.0 + 4.0 * np.cos(index * 0.7))
    rvec = np.array([pitch, yaw, roll], dtype=np.float64)
    tvec = np.array(
        [
            -125.0 + 18.0 * np.sin(index * 0.6),
            -90.0 + 14.0 * np.cos(index * 0.5),
            720.0 + 22.0 * index,
        ],
        dtype=np.float64,
    )
    return rvec, tvec


def render_images(count: int = 16) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texture = make_board_texture()
    h, w = texture.shape
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    width, height = IMAGE_SIZE

    views = []
    for i in range(count):
        rvec, tvec = board_pose(i)
        dst, _ = cv2.projectPoints(board_outer_corners_mm(), rvec, tvec, K, DIST)
        dst = dst.reshape(4, 2).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        image = np.full((height, width), 245, np.uint8)
        warped = cv2.warpPerspective(texture, H, (width, height), borderValue=245)
        mask = cv2.warpPerspective(np.full_like(texture, 255), H, (width, height), borderValue=0)
        image[mask > 0] = warped[mask > 0]
        image = cv2.GaussianBlur(image, (3, 3), 0.35)

        filename = f"{i + 1:02d}.bmp"
        cv2.imwrite(str(OUT_DIR / filename), image)
        views.append(
            {
                "image": f"images/{filename}",
                "rvec_board_to_camera": rvec.round(8).tolist(),
                "tvec_board_to_camera_mm": tvec.round(8).tolist(),
                "board_quad_px": dst.round(3).tolist(),
            }
        )

    meta = {
        "board_type": "chessboard",
        "inner_cols": INNER_COLS,
        "inner_rows": INNER_ROWS,
        "square_size_mm": SQUARE_SIZE_MM,
        "image_width": width,
        "image_height": height,
        "camera_matrix": K.tolist(),
        "distortion": DIST.tolist(),
        "views": views,
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    render_images()
