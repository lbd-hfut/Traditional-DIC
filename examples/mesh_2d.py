"""Thin command-line adapter for the normalized 2D Mesh-DIC facade."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from traditional_dic.case import resolve_case  # noqa: E402
from traditional_dic.config_resolver import resolve_config  # noqa: E402
from traditional_dic.workflows import run_mesh_2d  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "config" / "case_paths.yaml")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "mesh_2d.yaml")
    parser.add_argument("--case", default="mono_2d", help="case key in the paths config (e.g. mono_2d, mono_2d_01)")
    parser.add_argument("--element", choices=["T3", "Q4", "Q8", "all"], default="all")
    parser.add_argument("--initialization", choices=["fedic_fft"], default=None, help="Override the mesh nodal-initialization route from the YAML configuration.")
    parser.add_argument("--optimization", choices=["fedic_element_icgn", "fedic_element_fgn"], default=None, help="Override the Mesh-DIC global optimization route from the YAML configuration.")
    parser.add_argument("--objective", choices=["ssd", "znssd"], default=None, help="Override the Mesh-DIC photometric objective from the YAML configuration.")
    parser.add_argument("--bspline-degree", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--search-radius", type=int, default=None)
    parser.add_argument("--regularization-alpha", type=float)
    parser.add_argument("--dense-samples-per-axis", type=int, default=25)
    parser.add_argument("--init-quality-control", action="store_true")
    parser.add_argument("--init-min-zncc", type=float)
    parser.add_argument("--init-max-znssd", type=float)
    parser.add_argument("--init-fedic-qfactor", action="store_true")
    parser.add_argument("--init-fedic-qfactor-std-factor", type=float)
    parser.add_argument("--init-neighbor-mad-factor", type=float)
    parser.add_argument("--init-max-neighbor-deviation", type=float)
    parser.add_argument("--init-interpolation-neighbors", type=int)
    args = parser.parse_args()

    resolved_case = resolve_case(
        "mesh_2d",
        paths_config=args.paths_config,
        case_key=args.case,
        repository_root=PROJECT_ROOT,
    )
    overrides = {
        key: value
        for key, value in {
            "initialization.method": args.initialization,
            "optimization.method": args.optimization,
            "optimization.objective": args.objective,
            "optimization.regularization_alpha": args.regularization_alpha,
            "interpolation.degree": args.bspline_degree,
            "optimization.max_iterations": args.max_iterations,
            "optimization.convergence_threshold": args.tolerance,
            "initialization.fedic_fft.search_radius": args.search_radius,
        }.items()
        if value is not None
    }
    if args.init_quality_control:
        overrides["initialization.quality_control.enabled"] = True
        for key, value in {
            "initialization.quality_control.min_zncc": args.init_min_zncc,
            "initialization.quality_control.max_znssd": args.init_max_znssd,
            "initialization.quality_control.fedic_qfactor_std_factor": args.init_fedic_qfactor_std_factor,
            "initialization.quality_control.neighbor_mad_factor": args.init_neighbor_mad_factor,
            "initialization.quality_control.max_neighbor_deviation": args.init_max_neighbor_deviation,
            "initialization.quality_control.interpolation_neighbors": args.init_interpolation_neighbors,
        }.items():
            if value is not None:
                overrides[key] = value
        if args.init_fedic_qfactor:
            overrides["initialization.quality_control.fedic_qfactor_enabled"] = True
    resolved_config = resolve_config(
        "mesh_2d",
        config_path=args.config,
        overrides=overrides,
        repository_root=PROJECT_ROOT,
    )
    element_types = ["T3", "Q4", "Q8"] if args.element == "all" else [args.element]
    result = run_mesh_2d(
        resolved_case,
        resolved_config,
        repository_root=PROJECT_ROOT,
        element_types=element_types,
        dense_samples_per_axis=args.dense_samples_per_axis,
    )
    for path in result.artifacts.get("results", []):
        print(f"Wrote Mesh-DIC results to {path}")


if __name__ == "__main__":
    main()
