import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def add_subsetdic_to_path(path: Path) -> None:
    src = path / "src"
    if not src.exists():
        raise FileNotFoundError(f"SubsetDIC src directory not found: {src}")
    sys.path.insert(0, str(src))


def save_heatmap(path: Path, data: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
        image = ax.imshow(data, cmap="coolwarm")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        finite = np.asarray(data, dtype=np.float64)
        finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
        span = np.percentile(np.abs(finite), 99.0)
        if span <= 0.0:
            scaled = np.zeros_like(finite, dtype=np.uint8)
        else:
            scaled = np.clip((finite / (2.0 * span) + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(scaled).save(path)


def stats(diff: np.ndarray) -> dict:
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "p99_abs": float(np.percentile(abs_diff, 99.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--subsetdic", type=Path, default=Path(r"C:\02Project\Research\SubsetDIC"))
    parser.add_argument("--image", type=Path, default=Path("case/ring/001.bmp"))
    parser.add_argument("--out", type=Path, default=Path("case/ring/test_interpolation"))
    parser.add_argument("--dump-exe", type=Path, default=Path("build-ninja-tests/dump_bspline_precompute.exe"))
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--border", type=int, default=20)
    args = parser.parse_args()

    repo = args.repo.resolve()
    image_path = (repo / args.image).resolve()
    out_dir = (repo / args.out).resolve()
    dump_exe = (repo / args.dump_exe).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    add_subsetdic_to_path(args.subsetdic.resolve())
    from subsetdic.bcoef import compute_bspline_coefficients
    from subsetdic.qk import generate_QK, precompute_qk_lut, extract_gradients_from_lut

    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    max_gray = float(np.max(image))
    if max_gray > 0.0:
        image = image / max_gray
    else:
        image = np.zeros_like(image, dtype=np.float64)
    np.savetxt(out_dir / "input_image.csv", image, delimiter=",", fmt="%.17g")

    subprocess.run(
        [
            str(dump_exe),
            str(out_dir / "input_image.csv"),
            str(out_dir),
            str(args.degree),
            str(args.border),
        ],
        check=True,
        cwd=repo,
    )

    qk = generate_QK()
    py_bcoef = compute_bspline_coefficients(image, border=args.border)
    py_lut = precompute_qk_lut(py_bcoef, qk)
    py_gx, py_gy = extract_gradients_from_lut(py_lut)

    offset = args.border - args.degree // 2
    py_gx_image = py_gx[offset : offset + image.shape[0], offset : offset + image.shape[1]]
    py_gy_image = py_gy[offset : offset + image.shape[0], offset : offset + image.shape[1]]

    cpp_bcoef = np.loadtxt(out_dir / "cpp_bcoef.csv", delimiter=",")
    cpp_gx = np.loadtxt(out_dir / "cpp_gradient_x.csv", delimiter=",")
    cpp_gy = np.loadtxt(out_dir / "cpp_gradient_y.csv", delimiter=",")

    np.save(out_dir / "python_bcoef.npy", py_bcoef)
    np.save(out_dir / "python_gradient_x_image.npy", py_gx_image)
    np.save(out_dir / "python_gradient_y_image.npy", py_gy_image)
    np.save(out_dir / "cpp_bcoef.npy", cpp_bcoef)
    np.save(out_dir / "cpp_gradient_x.npy", cpp_gx)
    np.save(out_dir / "cpp_gradient_y.npy", cpp_gy)

    bcoef_diff = cpp_bcoef - py_bcoef
    gx_diff = cpp_gx - py_gx_image
    gy_diff = cpp_gy - py_gy_image

    np.save(out_dir / "diff_bcoef.npy", bcoef_diff)
    np.save(out_dir / "diff_gradient_x.npy", gx_diff)
    np.save(out_dir / "diff_gradient_y.npy", gy_diff)

    save_heatmap(out_dir / "diff_bcoef.png", bcoef_diff, "C++ - Python B-spline coefficients")
    save_heatmap(out_dir / "diff_gradient_x.png", gx_diff, "C++ - Python gradient X")
    save_heatmap(out_dir / "diff_gradient_y.png", gy_diff, "C++ - Python gradient Y")

    summary = {
        "image": str(image_path),
        "shape": list(image.shape),
        "normalization": "full_image_max_gray",
        "max_gray": max_gray,
        "degree": args.degree,
        "border": args.border,
        "gradient_python_crop_offset": offset,
        "bcoef": stats(bcoef_diff),
        "gradient_x": stats(gx_diff),
        "gradient_y": stats(gy_diff),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
