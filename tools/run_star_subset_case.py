"""Run one Subset-DIC optimizer case for the mono_DIC/star inputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    star_root = PROJECT_ROOT / "case" / "mono_DIC" / "star"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--shape", choices=("first_order", "second_order"), required=True)
    parser.add_argument("--objective", choices=("ssd", "znssd"), required=True)
    parser.add_argument("--optimizer", choices=("icgn", "forward_gauss_newton"), required=True)
    parser.add_argument("--reference", type=Path, default=star_root / "001.bmp")
    parser.add_argument("--deformed", type=Path, default=star_root / "002.bmp")
    parser.add_argument("--roi", type=Path, default=star_root / "003.bmp")
    parser.add_argument("--result-root", type=Path, default=star_root / "result" / "subset")
    return parser.parse_args()


def configuration(args: argparse.Namespace) -> dict:
    return {
        "subset": {"radius": 15, "truncate_roi_subsets": True, "min_valid_sample_ratio": 0.5, "min_valid_samples": 12},
        "shape_function": {"order": args.shape},
        "optimization": {"method": args.optimizer, "max_iterations": 30, "convergence_threshold": 1.0e-3},
        "correlation": {"criterion": args.objective},
        "interpolation": {"method": "bspline", "degree": 5},
        "initialization": {
            "method": "integer_search",
            "integer_search": {
                "subset_radius": 10, "search_radius": 30, "sift_enabled": True,
                "pyramid_enabled": False, "pyramid_scale": 4, "pyramid_refinement_radius": 4,
            },
            "subpixel_refinement": {
                "enabled": True, "shape_function": args.shape, "optimizer": args.optimizer,
                "objective": args.objective, "subset_radius": 15, "max_iterations": 30,
                "convergence_threshold": 1.0e-3,
            },
        },
        "seed_selection": {
            "method": "roi_kmeans", "seed_count": 16, "threads": 1, "quality_metric": "znssd",
            "max_znssd": 2.0, "min_zncc": 0.7, "max_ssd": 0.05, "min_displacement_norm": 0.0,
            "min_texture_std": 0.02, "kmeans_iterations": 20, "kmeans_sample_limit": 20000,
        },
        "reliability_propagation": {"spacing": 3, "max_znssd": 2.0},
    }


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.result_root / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration(args), sort_keys=False), encoding="utf-8")
    subprocess.run(
        [
            str(PROJECT_ROOT / "build" / "subset_dic_diagnostic.exe"),
            str(args.reference), str(args.deformed), str(args.roi),
            str(output_dir / "displacements.csv"), str(config_path),
        ],
        check=True,
    )
