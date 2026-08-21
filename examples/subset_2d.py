"""Thin command-line adapter for the normalized 2D Subset-DIC facade."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"

import sys

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from traditional_dic.case import resolve_case  # noqa: E402
from traditional_dic.config_resolver import resolve_config  # noqa: E402
from traditional_dic.workflows import run_subset_2d  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "config" / "case_paths.yaml")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "subset_2d.yaml")
    parser.add_argument("--case", default="mono_2d", help="case key in the paths config (e.g. mono_2d, mono_2d_01)")
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--spacing", type=int, default=None)
    parser.add_argument("--search-radius", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    resolved_case = resolve_case(
        "subset_2d",
        paths_config=args.paths_config,
        case_key=args.case,
        repository_root=PROJECT_ROOT,
    )
    overrides = {
        key: value
        for key, value in {
            "subset.radius": args.radius,
            "reliability_propagation.spacing": args.spacing,
            "initialization.integer_search.search_radius": args.search_radius,
            "seed_selection.seed_count": args.seed_count,
            "optimization.max_iterations": args.max_iterations,
        }.items()
        if value is not None
    }
    resolved_config = resolve_config(
        "subset_2d",
        config_path=args.config,
        overrides=overrides,
        repository_root=PROJECT_ROOT,
    )
    result = run_subset_2d(resolved_case, resolved_config, repository_root=PROJECT_ROOT)
    for path in result.artifacts.get("displacements", []):
        print(f"Wrote Subset-DIC results to {path}")


if __name__ == "__main__":
    main()
