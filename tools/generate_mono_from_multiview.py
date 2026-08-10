"""Convert selected cameras of a multi-view DIC case into 2D mono DIC cases.

For each selected camera, this writes a mono case directory containing:
    001.bmp  reference image
    002.bmp  current (deformed) image
    003.bmp  ROI mask (binary 0/255) covering the visible surface

Layout convention (matches config/case_paths.yaml mono_2d):
    the first sorted image is the reference, the last is the ROI mask, and
    every intermediate image is solved as a deformed frame.

ROI generation:
    The synthetic scenes render a speckled cylinder on a constant background
    (intensity 30). The surface ROI is the non-background region, filled with
    a morphological close to remove isolated dark speckle holes, restricted to
    the largest connected component, then eroded slightly so subsets stay well
    inside the surface boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def generate_roi(
    ref: np.ndarray,
    background_level: float = 30.0,
    threshold: float = 3.0,
    close_radius: int = 5,
    erode_radius: int = 3,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Return a binary (0/255) ROI mask covering the object surface."""
    mask = ref > (background_level + threshold)
    mask = mask.astype(np.uint8)

    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if keep_largest_component:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = (labels == largest).astype(np.uint8)
        if erode_radius > 0:
            kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
            mask = cv2.erode(mask, kernel_e)
    else:
        # Fallback without OpenCV: threshold + keep largest 8-connected component,
        # no morphology. OpenCV is the preferred path.
        mask = largest_component(mask) if keep_largest_component else mask

    return (mask > 0).astype(np.uint8) * 255


def largest_component(mask: np.ndarray) -> np.ndarray:
    """8-connected largest component (no-op fallback when OpenCV is missing)."""
    return mask


def build_mono_case(
    cam_dir: Path,
    out_dir: Path,
    background_level: float = 30.0,
) -> None:
    ref_path = cam_dir / "001.bmp"
    def_path = cam_dir / "002.bmp"
    if not ref_path.exists() or not def_path.exists():
        raise FileNotFoundError(f"missing reference/current image in {cam_dir}")

    ref = read_gray(ref_path)
    deformed = read_gray(def_path)
    roi = generate_roi(ref, background_level=background_level)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(ref, 0, 255).astype(np.uint8), "L").save(out_dir / "001.bmp")
    Image.fromarray(np.clip(deformed, 0, 255).astype(np.uint8), "L").save(out_dir / "002.bmp")
    Image.fromarray(roi, "L").save(out_dir / "003.bmp")
    print(f"wrote {out_dir}: ref {ref.shape[1]}x{ref.shape[0]}, roi coverage {float((roi>0).mean()):.3f}")


def save_preview(out_dir: Path, png_path: Path, background_level: float = 30.0) -> None:
    """Overlay ROI boundary on the reference image and save a preview PNG."""
    ref = read_gray(out_dir / "001.bmp")
    roi = np.asarray(Image.open(out_dir / "003.bmp").convert("L")) > 0
    if cv2 is not None:
        contour, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb = np.stack([ref] * 3, axis=-1).astype(np.uint8)
        cv2.drawContours(rgb, contour, -1, (0, 0, 255), 2)
        Image.fromarray(rgb).save(png_path)
    else:
        # fallback: mark the ROI boundary by difference with an eroded copy
        rgb = np.stack([ref] * 3, axis=-1).astype(np.uint8)
        inner = np.zeros_like(roi)
        inner[1:-1, 1:-1] = roi[1:-1, 1:-1] & roi[:-2, :-2] & roi[2:, :-2] & roi[:-2, 2:] & roi[2:, 2:]
        boundary = roi & ~inner
        rgb[boundary] = (255, 0, 0)
        Image.fromarray(rgb).save(png_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=PROJECT_ROOT / "case/multi_DIC/ComplexCylinderDIC/images")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "case/mono_DIC/Complex")
    parser.add_argument("--cameras", type=str, default="all",
                        help="comma-separated camera indices, e.g. 0,1,2, or 'all'")
    parser.add_argument("--background-level", type=float, default=30.0)
    parser.add_argument("--previews", action="store_true", help="also write ROI-overlay preview PNGs")
    args = parser.parse_args()

    src = args.src
    if args.cameras.strip().lower() == "all":
        cams = sorted(p for p in src.glob("cam_*") if p.is_dir())
    else:
        indices = [int(x) for x in args.cameras.split(",") if x.strip()]
        cams = [src / f"cam_{i}" for i in indices]

    if not cams:
        print("no cameras selected", file=sys.stderr)
        raise SystemExit(1)

    for cam_dir in cams:
        out_dir = args.out / cam_dir.name
        build_mono_case(cam_dir, out_dir, background_level=args.background_level)
        if args.previews:
            save_preview(out_dir, out_dir / "preview_roi.png", background_level=args.background_level)


if __name__ == "__main__":
    main()
