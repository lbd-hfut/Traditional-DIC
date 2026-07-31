"""Run pairwise 2D-DIC fields for the CylinderDIC multiview case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from python.traditional_dic.multiview import compute_pairwise_2d_dic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=PROJECT_ROOT / "case" / "multi_DIC" / "CylinderDIC")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--subset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mesh-types", nargs="+", default=["T3", "Q4", "Q8"], choices=["T3", "Q4", "Q8"])
    args = parser.parse_args()

    result = compute_pairwise_2d_dic(
        args.case_root,
        options={
            "max_pairs": args.max_pairs,
            "run_subset": args.subset,
            "run_mesh": args.mesh,
            "mesh_types": tuple(args.mesh_types),
        },
    )
    print(result)


if __name__ == "__main__":
    main()
