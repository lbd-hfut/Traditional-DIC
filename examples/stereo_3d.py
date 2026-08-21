"""Thin command-line adapter for the Stereo Subset-DIC facade."""

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
from traditional_dic.workflows import run_stereo_3d  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-config", type=Path, default=PROJECT_ROOT / "config" / "stereo_3d.yaml")
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "config" / "case_paths.yaml")
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--compute-fields", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    resolved_case = resolve_case(
        "stereo_3d",
        paths_config=args.paths_config,
        repository_root=PROJECT_ROOT,
    )
    resolved_config = resolve_config(
        "stereo_3d",
        config_path=args.stereo_config,
        calibration_config_path=args.calibration_config,
        repository_root=PROJECT_ROOT,
    )
    result = run_stereo_3d(
        resolved_case,
        resolved_config,
        repository_root=PROJECT_ROOT,
        calibrate=not args.skip_calibration,
        compute_fields=args.compute_fields,
    )
    print(f"Stereo-DIC workflow completed under {result.output_root}")


if __name__ == "__main__":
    main()
