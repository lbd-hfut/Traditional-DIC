"""Generate the canonical Multiview Subset-only F0C staging artifacts.

This explicit maintenance runner delegates every scientific operation to the
existing calibration and multiview APIs. It never runs the production CLI and
never writes beneath the source case.
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

import numpy as np

from traditional_dic import calibration
from traditional_dic.config import load_config
from traditional_dic.multiview import (
    compute_pairwise_2d_dic,
    compute_pairwise_3d_dic,
    generate_pair_masks_from_calibration,
    recover_multiview_calibration_scale,
    save_pair_selection_report,
    select_camera_pairs,
    stitch_pairwise_3d_surfaces,
)


def load_example(repository: Path):
    path = repository / "examples/multiview_3d.py"
    spec = importlib.util.spec_from_file_location("f0c_multiview_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load multiview example: {path}")
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
    parser.add_argument("--multiview-config", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    staging = args.staging_root.resolve()
    if repository == staging or repository in staging.parents:
        parser.error("staging-root must be outside repository")
    if staging.exists():
        if not args.force:
            parser.error(f"staging-root exists: {staging}; pass --force to replace it")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    example = load_example(repository)
    paths_cfg = load_config(args.paths_config)
    case_cfg = dict(paths_cfg.get("multiview_3d", {}) or {})
    case_root = (repository / str(case_cfg["case_root"])).resolve()
    cfg = load_config(args.multiview_config)
    image_root = (case_root / str(dict(case_cfg.get("images", {}) or {})["root"])).resolve()
    image_paths = example._collect_reference_images(
        staging,
        {"pairwise_2d_dic": {"image_dir": str(image_root), "reference_frame": "001.bmp"}},
        str(image_root),
        "001.bmp",
    )
    reference_frame, deformed_frame, roi_frame = example._camera_frame_names(
        staging, str(image_root), manual_roi=False
    )
    if reference_frame != "001.bmp" or deformed_frame != "002.bmp" or roi_frame is not None:
        raise RuntimeError(
            f"unexpected CylinderDIC frame contract: {reference_frame}, {deformed_frame}, {roi_frame}"
        )

    runtime: dict[str, Any] = {
        "architectural_contract": "multiview_3d=subset_only",
        "case_root": str(case_root),
        "staging_root": str(staging),
        "resolved_image_paths": [str(path) for path in image_paths],
        "operations": {},
        "pairwise_field_repeatability": {
            "executed": False,
            "reason": "Canonical pairwise Subset generation is expensive; F0A/F0B protect the Subset engine and F0C locks all pair routing and fields from one run.",
        },
    }

    def measure(name: str, function: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        record = {"status": "running"}
        runtime["operations"][name] = record
        write_json(staging / "runtime.json", runtime)
        try:
            value = function()
        except Exception as exc:
            record.update(status="failed", runtime_seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
            write_json(staging / "runtime.json", runtime)
            raise
        record.update(status="succeeded", runtime_seconds=time.perf_counter() - started)
        write_json(staging / "runtime.json", runtime)
        return value

    def run_calibration(tag: str):
        calibration_data = calibration.calibrate_multiview_colmap_like(image_paths, config=runtime_cfg)
        calibration_dir = staging / tag / "calibration"
        example._save_calibration_products(calibration_data, calibration_dir, image_paths)
        return calibration_data

    def run_scale(tag: str, calibration_data: dict[str, Any]):
        calibration_dir = staging / tag / "calibration"
        scale_cfg = dict(runtime_cfg)
        scale_cfg["scale"] = dict(runtime_cfg["scale"])
        scale_cfg["scale"]["calibration_dir"] = str(calibration_dir)
        scale_cfg["scale"]["chessboard_dir"] = str(case_root / str(dict(case_cfg.get("calibration", {}) or {})["chessboard_dir"]))
        return recover_multiview_calibration_scale(staging, calibration_data, config=scale_cfg)

    def run_pair_selection(tag: str, calibration_data: dict[str, Any]):
        selection = select_camera_pairs(calibration_data, runtime_cfg.get("camera_pair_selection"))
        save_pair_selection_report(selection, staging / tag / "calibration" / "pair_selection_report.json")
        return selection

    def run_masks(tag: str, calibration_data: dict[str, Any], selection: Any):
        mask_dir = staging / tag / "mask"
        mask_cfg = dict(runtime_cfg)
        mask_cfg["output"] = {"mask_dir": str(mask_dir), "roi_dir": str(mask_dir)}
        return generate_pair_masks_from_calibration(
            staging,
            calibration_data,
            config=mask_cfg,
            pair_selection=selection,
            output_dir=mask_dir,
        )

    runtime_cfg = dict(cfg)
    runtime_cfg["output"] = {
        "mask_dir": str(staging / "mask-a"),
        "roi_dir": str(staging / "mask-a"),
        "calibration_dir": str(staging / "calibration-a"),
        "disp_dir": str(staging / "fields-a"),
        "reconstruct_dir": str(staging / "pairwise-a"),
    }
    runtime_cfg["pairwise_2d_dic"] = dict(cfg.get("pairwise_2d_dic", {}) or {})
    runtime_cfg["pairwise_2d_dic"].update(
        {
            "run_subset": True,
            "run_mesh": False,
            "image_dir": str(image_root),
            "reference_frame": reference_frame,
            "deformed_frame": deformed_frame,
            "mask_dir": str(staging / "mask-a"),
            "roi_dir": str(staging / "mask-a"),
            "output_dir": str(staging / "fields-a"),
            "calibration_dir": str(staging / "calibration-a"),
            "subset_config": str(repository / "config/subset_2d.yaml"),
        }
    )
    runtime_cfg["scale"] = dict(cfg.get("scale", {}) or {})
    runtime_cfg["scale"].update(
        {
            "calibration_dir": str(staging / "calibration-a"),
            "chessboard_dir": str(case_root / str(dict(case_cfg.get("calibration", {}) or {})["chessboard_dir"])),
        }
    )
    runtime_cfg["pairwise_3d_dic"] = dict(cfg.get("pairwise_3d_dic", {}) or {})
    runtime_cfg["pairwise_3d_dic"].update(
        {
            "solver": "subset",
            "field_dir": str(staging / "fields-a"),
            "output_dir": str(staging / "pairwise-a"),
            "calibration_dir": str(staging / "calibration-a"),
            "write_shape_maps": False,
            "write_deformation_maps": False,
            "write_surface_strain": False,
        }
    )
    runtime_cfg["surface_stitch"] = dict(cfg.get("surface_stitch", {}) or {})
    runtime_cfg["surface_stitch"].update(
        {
            "solver": "subset",
            "pairwise_3d_dir": str(staging / "pairwise-a"),
            "output_dir": str(staging / "stitched-a"),
            "calibration_dir": str(staging / "calibration-a"),
        }
    )

    calibration_a = measure("calibration_run_a", lambda: run_calibration("calibration-a"))
    calibration_b = measure("calibration_run_b", lambda: run_calibration("calibration-b"))

    runtime_cfg["scale"]["calibration_dir"] = str(staging / "calibration-a")
    scale_a = measure("scale_run_a", lambda: run_scale("calibration-a", calibration_a))
    runtime_cfg["scale"]["calibration_dir"] = str(staging / "calibration-b")
    scale_b = measure("scale_run_b", lambda: run_scale("calibration-b", calibration_b))

    runtime_cfg["scale"]["calibration_dir"] = str(staging / "calibration-a")
    selection_a = measure("pair_selection_run_a", lambda: run_pair_selection("calibration-a", calibration_a))
    selection_b = measure("pair_selection_run_b", lambda: run_pair_selection("calibration-b", calibration_b))
    masks_a = measure("mask_generation_run_a", lambda: run_masks("mask-a", calibration_a, selection_a))
    masks_b = measure("mask_generation_run_b", lambda: run_masks("mask-b", calibration_b, selection_b))

    pairwise_2d_options = dict(runtime_cfg["pairwise_2d_dic"])
    pairwise_2d_options.update({"run_subset": True, "run_mesh": False, "output_dir": str(staging / "fields-a"), "mask_dir": str(staging / "mask-a"), "roi_dir": str(staging / "mask-a"), "overwrite": True})
    pairwise_2d = measure(
        "pairwise_subset_fields",
        lambda: compute_pairwise_2d_dic(
            staging,
            calibration_a,
            config=runtime_cfg,
            pair_selection=selection_a,
            subset_config=repository / "config/subset_2d.yaml",
            options=pairwise_2d_options,
        ),
    )

    pairwise_3d_options = dict(runtime_cfg["pairwise_3d_dic"])
    pairwise_3d_options.update({"solver": "subset", "field_dir": str(staging / "fields-a"), "output_dir": str(staging / "pairwise-a"), "calibration_dir": str(staging / "calibration-a"), "write_shape_maps": False, "write_deformation_maps": False, "write_surface_strain": False})
    pairwise_3d = measure(
        "pairwise_reconstruction_run_a",
        lambda: compute_pairwise_3d_dic(staging, calibration_a, config=runtime_cfg, pair_selection=selection_a, options=pairwise_3d_options),
    )
    pairwise_3d_b_options = dict(pairwise_3d_options)
    pairwise_3d_b_options["output_dir"] = str(staging / "pairwise-b")
    pairwise_3d_b = measure(
        "pairwise_reconstruction_run_b",
        lambda: compute_pairwise_3d_dic(staging, calibration_a, config=runtime_cfg, pair_selection=selection_a, options=pairwise_3d_b_options),
    )

    stitch_options = dict(runtime_cfg["surface_stitch"])
    stitch_options.update({"solver": "subset", "pairwise_3d_dir": str(staging / "pairwise-a"), "output_dir": str(staging / "stitched-a"), "calibration_dir": str(staging / "calibration-a")})
    stitched_a = measure(
        "stitch_run_a",
        lambda: stitch_pairwise_3d_surfaces(staging, config=runtime_cfg, pair_selection=selection_a, options=stitch_options),
    )
    stitch_options_b = dict(stitch_options)
    stitch_options_b.update({"pairwise_3d_dir": str(staging / "pairwise-b"), "output_dir": str(staging / "stitched-b")})
    stitched_b = measure(
        "stitch_run_b",
        lambda: stitch_pairwise_3d_surfaces(staging, config=runtime_cfg, pair_selection=selection_a, options=stitch_options_b),
    )

    runtime["resolved_contract"] = {
        "reference_frame": reference_frame,
        "deformed_frame": deformed_frame,
        "camera_count": len(image_paths),
        "pairwise_solver": "subset",
        "mesh_executed": False,
        "pair_count": len(selection_a.pairs),
        "pairwise_2d": {"run_subset": pairwise_2d.run_subset, "run_mesh": pairwise_2d.run_mesh},
        "pairwise_3d_solver": pairwise_3d.solver,
        "stitch_solver": stitched_a.solver,
        "scale_a": {"sfm_to_world_scale": scale_a.sfm_to_world_scale, "world_to_sfm_scale": scale_a.world_to_sfm_scale},
        "scale_b": {"sfm_to_world_scale": scale_b.sfm_to_world_scale, "world_to_sfm_scale": scale_b.world_to_sfm_scale},
        "selected_pairs_a": [list(pair) for pair in selection_a.pair_names],
        "selected_pairs_b": [list(pair) for pair in selection_b.pair_names],
        "mask_files_a": masks_a.mask_files,
        "mask_files_b": masks_b.mask_files,
        "pairwise_reconstruction_a": {"total_points": pairwise_3d.total_points, "valid_points": pairwise_3d.valid_points},
        "pairwise_reconstruction_b": {"total_points": pairwise_3d_b.total_points, "valid_points": pairwise_3d_b.valid_points},
        "stitch_a": {"point_count": stitched_a.point_count, "face_count": stitched_a.face_count},
        "stitch_b": {"point_count": stitched_b.point_count, "face_count": stitched_b.face_count},
    }
    write_json(staging / "runtime.json", runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
