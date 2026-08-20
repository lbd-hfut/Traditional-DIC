"""Thin command-line adapter for the Multiview Subset-DIC facade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from traditional_dic.case import resolve_case  # noqa: E402
from traditional_dic.config_resolver import resolve_config  # noqa: E402
from traditional_dic.workflows import run_multiview_3d  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default=PROJECT_ROOT / "config" / "case_paths.yaml", type=Path, help="Case input/output YAML config.")
    parser.add_argument("--config", default=None, type=Path, help="Multiview DIC YAML config.")
    parser.add_argument("--solver", choices=("subset",), default="subset", help="Multiview pairwise correspondence solver (Subset-DIC only).")
    parser.add_argument("--resume", action="store_true", help="Reuse completed pairwise 2D fields.")
    args = parser.parse_args()

    resolved_case = resolve_case(
        "multiview_3d",
        paths_config=args.paths_config,
        repository_root=PROJECT_ROOT,
    )
    resolved_config = resolve_config(
        "multiview_3d",
        config_path=args.config,
        repository_root=PROJECT_ROOT,
    )
    result = run_multiview_3d(
        resolved_case,
        resolved_config,
        repository_root=PROJECT_ROOT,
        resume=args.resume,
    )
    print(json.dumps(dict(result.artifacts), indent=2))


if __name__ == "__main__":
    main()
