"""Run the canonical F0B Stereo Subset workflow in an isolated staging tree.

This is an explicit maintenance operation, never a pytest fixture. It delegates
calibration and three-field generation to the existing example workflow and
delegates reconstruction to the existing authoritative Python/C++ API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from traditional_dic.config import load_config
from traditional_dic.stereo import load_camera_pair, reconstruct_from_field_files


def load_stereo_example(repository: Path):
    path = repository / "examples/stereo_3d.py"
    spec = importlib.util.spec_from_file_location("f0b_stereo_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stereo workflow from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--paths-config", type=Path, required=True)
    parser.add_argument("--stereo-config", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--subset-config", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    staging = args.staging_root.resolve()
    if repository == staging or repository in staging.parents:
        parser.error("staging-root must be outside the repository")
    if staging.exists():
        if not args.force:
            parser.error(f"staging-root exists: {staging}; pass --force to replace it")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    example = load_stereo_example(repository)
    case_cfg = dict(load_config(args.paths_config).get("stereo_3d", {}) or {})
    stereo_cfg = load_config(args.stereo_config)
    calibration_cfg = load_config(args.calibration_config)
    subset_cfg = load_config(args.subset_config)
    if str(stereo_cfg.get("solver", {}).get("method", "subset")).lower() != "subset":
        raise RuntimeError("F0B requires stereo solver.method=subset")
    paths = example.build_paths(case_cfg)
    runtime: dict[str, Any] = {
        "working_directory": str(repository),
        "staging_directory": str(staging),
        "architectural_contract": "stereo_3d=subset_only",
        "operations": {},
    }

    def measure(name: str, function: Callable[[], Any]) -> Any:
        started_wall = time.time()
        started = time.perf_counter()
        record: dict[str, Any] = {"started_unix": started_wall, "status": "running"}
        runtime["operations"][name] = record
        write_json(staging / "runtime.json", runtime)
        try:
            value = function()
        except Exception as exc:
            record.update(
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_json(staging / "runtime.json", runtime)
            raise
        record.update(status="succeeded", runtime_seconds=time.perf_counter() - started)
        write_json(staging / "runtime.json", runtime)
        return value

    calibration_a = staging / "calibration-a"
    calibration_b = staging / "calibration-b"
    camera_a = measure(
        "calibration_run_a",
        lambda: example.run_calibration(paths, calibration_cfg, calibration_a),
    )
    measure(
        "calibration_run_b",
        lambda: example.run_calibration(paths, calibration_cfg, calibration_b),
    )

    field_dir = staging / "fields"
    field_visualization = staging / "field-visualization"
    measure(
        "three_subset_fields",
        lambda: example.compute_subset_fields(paths, subset_cfg, field_dir, field_visualization),
    )
    runtime["field_repeatability"] = {
        "executed": False,
        "reason": "F0A already protects the Subset engine; F0B locks canonical Stereo field routing from one expensive three-field run.",
    }
    write_json(staging / "runtime.json", runtime)

    left_camera, right_camera, world_scale = load_camera_pair(camera_a)
    recon_cfg = dict(stereo_cfg.get("reconstruction", {}) or {})

    def reconstruct(output: Path):
        return reconstruct_from_field_files(
            field_dir,
            left_camera,
            right_camera,
            out_dir=output / "reconstruct",
            deformation_out_dir=output / "deformation",
            write_shape_maps=False,
            write_deformation_maps=False,
            write_surface_strain=False,
            min_correlation=float(recon_cfg.get("min_correlation", 0.0)),
            quality_metric=str(recon_cfg.get("quality_metric", "correlation")),
            max_znssd=float(recon_cfg.get("max_znssd", 2.0)),
            max_reprojection_error_px=float(recon_cfg.get("max_reprojection_error_px", 5.0)),
            world_scale=world_scale,
            remove_rigid_body_motion=bool(recon_cfg.get("remove_rigid_body_motion", False)),
        )

    measure("reconstruction_run_a", lambda: reconstruct(staging / "reconstruction-a"))
    measure("reconstruction_run_b", lambda: reconstruct(staging / "reconstruction-b"))
    runtime["resolved_inputs"] = {key: str(value) for key, value in paths.items()}
    runtime["world_scale"] = world_scale
    write_json(staging / "runtime.json", runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
