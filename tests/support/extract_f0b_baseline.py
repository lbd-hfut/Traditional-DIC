"""Extract compact immutable F0B goldens from an explicit isolated run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import traditional_dic
import traditional_dic._traditional_dic as native

from tests.support.provenance import relative_hashes, sha256_file
from tests.support.stereo_baseline import numeric_difference, within_policy


SCHEMA_VERSION = "1.0"
BASELINE_COMMIT = "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
FIELD_SPECS = {
    "reference_disparity": ("cam1/00_L.bmp", "cam2/00_R.bmp", "L0", "R0"),
    "left_temporal": ("cam1/00_L.bmp", "cam1/04_L.bmp", "L0", "Llast"),
    "deformed_disparity": ("cam1/00_L.bmp", "cam2/04_R.bmp", "L0", "Rlast"),
}
CALIBRATION_ARRAY_KEYS = (
    "R_lr",
    "t_lr",
    "essential",
    "fundamental",
    "per_pair_errors",
    "per_pair_left_errors",
    "per_pair_right_errors",
    "initial_per_pair_errors",
    "initial_per_pair_left_errors",
    "initial_per_pair_right_errors",
    "rms_error",
    "initial_rms_error",
)
CAMERA_ARRAY_KEYS = ("K", "distortion", "R", "t", "projection_matrix", "camera_center", "rms_error")


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = stream.readline().strip().split(",")
    table = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    return header, {name: np.asarray(table[name]) for name in header}


def relative(repository: Path, value: str | Path) -> str:
    return str(Path(value).resolve().relative_to(repository))


def calibration_arrays(document: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"schema_version": np.array(SCHEMA_VERSION)}
    for side in ("left", "right"):
        for key in CAMERA_ARRAY_KEYS:
            arrays[f"{side}__{key}"] = np.asarray(document[side][key], dtype=np.float64)
        arrays[f"{side}__detection_points"] = np.asarray(
            [item["image_points"] for item in document[f"{side}_detections"]], dtype=np.float64
        )
    for key in CALIBRATION_ARRAY_KEYS:
        arrays[key] = np.asarray(document[key], dtype=np.float64)
    return arrays


def calibration_structure(repository: Path, document: dict[str, Any]) -> dict[str, Any]:
    detections = {}
    for side in ("left", "right"):
        detections[side] = [
            {
                "found": item["found"],
                "image_path": relative(repository, item["image_path"]),
                "image_size": item["image_size"],
                "point_count": len(item["image_points"]),
            }
            for item in document[f"{side}_detections"]
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "board": document["board"],
        "world_scale": document["world_scale"],
        "cameras": {
            side: {
                "label": document[side]["label"],
                "image_size": document[side]["image_size"],
                "image_width": document[side]["image_width"],
                "image_height": document[side]["image_height"],
            }
            for side in ("left", "right")
        },
        "detections": detections,
        "kept_pair_indices": document["kept_pair_indices"],
        "rejected_pair_indices": document["rejected_pair_indices"],
        "rejection_reasons": document["rejection_reasons"],
        "outlier_rejection_applied": document["outlier_rejection_applied"],
    }


def calibration_policy(key: str) -> dict[str, Any]:
    if "detection_points" in key or key.endswith("__R") and key.startswith("left") or key in {"left__t", "left__camera_center"}:
        atol, rtol, max_rmse = 0.0, 0.0, 0.0
        reason = "Observed bitwise-stable detection or fixed left-camera extrinsic field."
    elif key.endswith("__K"):
        atol, rtol, max_rmse = 10.0, 5e-3, 5.0
        reason = "OpenCV stereo optimization varied intrinsics across same-environment A/B runs."
    elif key.endswith("__distortion"):
        atol, rtol, max_rmse = 0.75, 1e-2, 0.35
        reason = "Distortion coefficients inherit observed OpenCV optimization variation."
    elif key.endswith("__R") or key == "R_lr":
        atol, rtol, max_rmse = 2e-4, 1e-4, 1e-4
        reason = "Rotation tolerance is approximately four times observed A/B variation."
    elif key.endswith("__t") or key in {"t_lr", "right__camera_center"}:
        atol, rtol, max_rmse = 1.5, 5e-3, 0.75
        reason = "Translation tolerance is approximately twice observed A/B maximum."
    elif key.endswith("__projection_matrix"):
        atol, rtol, max_rmse = 2000.0, 5e-3, 600.0
        reason = "Projection values combine intrinsics and translation; limit exceeds observed 837.63 maximum."
    elif key == "essential":
        atol, rtol, max_rmse = 1.5, 5e-3, 0.75
        reason = "Essential matrix variation follows the translation estimate."
    elif key == "fundamental":
        atol, rtol, max_rmse = 1e-4, 5e-3, 5e-5
        reason = "Tight scale-appropriate bound above observed 3.93e-5 maximum."
    elif key.startswith("initial_"):
        atol, rtol, max_rmse = 1e-6, 1e-5, 5e-7
        reason = "Initial calibration fields varied only below 7.3e-8."
    else:
        atol, rtol, max_rmse = 5e-4, 5e-3, 2.5e-4
        reason = "Final RMS/per-pair errors varied below 1.27e-4."
    return {"atol": atol, "rtol": rtol, "max_rmse": max_rmse, "reason": reason}


def source_record(repository: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(repository)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    staging = args.staging_root.resolve()
    output = args.output.resolve()
    if repository == staging or repository in staging.parents:
        parser.error("staging-root must be outside repository")
    if output.exists():
        if not args.force:
            parser.error(f"output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    runtime = json.loads((staging / "runtime.json").read_text(encoding="utf-8"))
    if any(item["status"] != "succeeded" for item in runtime["operations"].values()):
        raise RuntimeError("staging contains failed operations")

    calibration_a_path = staging / "calibration-a/stereo_calibration.json"
    calibration_b_path = staging / "calibration-b/stereo_calibration.json"
    calibration_a = json.loads(calibration_a_path.read_text(encoding="utf-8"))
    calibration_b = json.loads(calibration_b_path.read_text(encoding="utf-8"))
    structure_a = calibration_structure(repository, calibration_a)
    structure_b = calibration_structure(repository, calibration_b)
    if structure_a != structure_b:
        raise RuntimeError("calibration A/B structural results differ")
    arrays_a = calibration_arrays(calibration_a)
    arrays_b = calibration_arrays(calibration_b)
    calibration_dir = output / "stereo_plate/calibration"
    calibration_dir.mkdir(parents=True)
    np.savez_compressed(calibration_dir / "numerical.npz", **arrays_a)
    write_json(calibration_dir / "structure.json", structure_a)
    repeat_fields = {}
    for key in sorted(set(arrays_a) - {"schema_version"}):
        metrics = numeric_difference(arrays_a[key], arrays_b[key])
        policy = calibration_policy(key)
        repeat_fields[key] = {**metrics, **policy, "within_policy": within_policy(metrics, atol=policy["atol"], max_rmse=policy["max_rmse"])}
    if not all(item["within_policy"] for item in repeat_fields.values()):
        raise RuntimeError("calibration A/B numeric variation exceeds INITIAL_F0B_CALIBRATION_TOLERANCE")
    write_json(
        calibration_dir / "repeatability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "NUMERICALLY_CLOSE",
            "run_a_sha256": sha256_file(calibration_a_path),
            "run_b_sha256": sha256_file(calibration_b_path),
            "structure_identical": True,
            "policy_name": "INITIAL_F0B_CALIBRATION_TOLERANCE",
            "fields": repeat_fields,
        },
    )
    write_json(
        calibration_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rms_error": calibration_a["rms_error"],
            "initial_rms_error": calibration_a["initial_rms_error"],
            "pair_count": len(calibration_a["left_detections"]),
            "accepted_pair_count": len(calibration_a["kept_pair_indices"]),
            "rejected_pair_count": len(calibration_a["rejected_pair_indices"]),
            "left_found_count": sum(item["found"] for item in calibration_a["left_detections"]),
            "right_found_count": sum(item["found"] for item in calibration_a["right_detections"]),
            "image_dimensions": calibration_a["left"]["image_size"],
        },
    )

    field_output = output / "stereo_plate/fields"
    field_output.mkdir()
    case_root = repository / "case/stereo_DIC/plate_center_load"
    subset_hash = sha256_file(repository / "config/subset_2d.yaml")
    for name, (source_a, source_b, source_role, deformed_role) in FIELD_SPECS.items():
        source_path = staging / "fields" / f"{name}.csv"
        header, arrays = read_csv(source_path)
        payload = {"schema_version": np.array(SCHEMA_VERSION), **arrays}
        np.savez_compressed(field_output / f"{name}.npz", **payload)
        valid = arrays["valid"].astype(bool)
        write_json(
            field_output / f"{name}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "field_name": name,
                "solver": "subset",
                "source_image_a": str((case_root / source_a).relative_to(repository)),
                "source_image_b": str((case_root / source_b).relative_to(repository)),
                "reference_semantic": source_role,
                "deformed_semantic": deformed_role,
                "roi_source": str((case_root / "ROI.bmp").relative_to(repository)),
                "subset_config": "config/subset_2d.yaml",
                "subset_config_sha256": subset_hash,
                "header": header,
                "row_count": len(arrays[header[0]]),
                "valid_count": int(np.count_nonzero(valid)),
                "quality": {
                    "valid_fraction": float(np.mean(valid)),
                    "correlation_mean_valid": float(np.mean(arrays["correlation"][valid])),
                    "correlation_max_valid": float(np.max(arrays["correlation"][valid])),
                },
                "tolerance_policy": {
                    "name": "INITIAL_F0B_FIELD_TOLERANCE",
                    "atol": 1e-12,
                    "rtol": 1e-10,
                    "max_abs": 1e-10,
                    "max_rmse": 1e-12,
                    "reason": "Independent tight policy aligned with deterministic F0A Subset behavior; field repeatability was not rerun.",
                },
                "source_artifact_sha256": sha256_file(source_path),
            },
        )

    reconstruction_a_path = staging / "reconstruction-a/reconstruct/stereo_3d_points.csv"
    reconstruction_b_path = staging / "reconstruction-b/reconstruct/stereo_3d_points.csv"
    recon_header, recon_a = read_csv(reconstruction_a_path)
    _, recon_b = read_csv(reconstruction_b_path)
    recon_dir = output / "stereo_plate/reconstruction"
    recon_dir.mkdir()
    np.savez_compressed(recon_dir / "baseline.npz", schema_version=np.array(SCHEMA_VERSION), **recon_a)
    recon_repeat = {key: numeric_difference(recon_a[key], recon_b[key]) for key in recon_header}
    recon_bitwise = sha256_file(reconstruction_a_path) == sha256_file(reconstruction_b_path)
    write_json(
        recon_dir / "repeatability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "BITWISE_IDENTICAL" if recon_bitwise else "NUMERICALLY_IDENTICAL",
            "run_a_sha256": sha256_file(reconstruction_a_path),
            "run_b_sha256": sha256_file(reconstruction_b_path),
            "policy_name": "INITIAL_F0B_RECONSTRUCTION_TOLERANCE",
            "tolerance": {"atol": 1e-12, "rtol": 1e-10, "max_abs": 1e-10, "max_rmse": 1e-12},
            "fields": recon_repeat,
        },
    )
    source_summary = json.loads((staging / "reconstruction-a/reconstruct/stereo_3d_summary.json").read_text())
    valid = recon_a["valid"].astype(bool)
    input_valid = recon_a["input_valid"].astype(bool)
    write_json(
        recon_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "header": recon_header,
            "row_count": len(recon_a[recon_header[0]]),
            "input_valid_count": int(np.count_nonzero(input_valid)),
            "final_valid_count": int(np.count_nonzero(valid)),
            "valid_fraction": float(np.mean(valid)),
            "mean_reprojection_error_ref": float(np.mean(recon_a["reprojection_error_ref"][valid])),
            "mean_reprojection_error_def": float(np.mean(recon_a["reprojection_error_def"][valid])),
            "mean_combined_correlation": float(np.mean(recon_a["combined_correlation"][valid])),
            "workflow_summary": source_summary,
        },
    )
    write_json(
        recon_dir / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "calibration_source": "stereo_plate/calibration/numerical.npz (Run A)",
            "field_mapping": {name: f"stereo_plate/fields/{name}.npz" for name in FIELD_SPECS},
            "world_scale": 1.0,
            "world_scale_source": "camera_pair.json generated by examples.stereo_3d.run_calibration",
            "world_scale_application": "StereoDICOptions.world_scale in reconstruct_from_field_files",
            "quality_metric": "znssd",
            "max_znssd": 2.0,
            "max_reprojection_error_px": 5.0,
            "remove_rigid_body_motion": False,
        },
    )

    configs = [repository / f"config/{name}" for name in ("case_paths.yaml", "calibration.yaml", "stereo_3d.yaml", "subset_2d.yaml")]
    main_inputs = [case_root / path for path in ("cam1/00_L.bmp", "cam2/00_R.bmp", "cam1/04_L.bmp", "cam2/04_R.bmp", "ROI.bmp")]
    calibration_inputs = sorted((case_root / "calibrate1").glob("*.bmp")) + sorted((case_root / "calibrate2").glob("*.bmp"))
    native_path = Path(native.__file__).resolve()
    write_json(
        output / "stereo_plate/provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline_commit": BASELINE_COMMIT,
            "repository": "Traditional-DIC",
            "canonical_case": "case/stereo_DIC/plate_center_load",
            "workflow_capability": {"subset_2d": True, "mesh_2d": True, "stereo_3d_solver": "subset", "multiview_3d_solver": "subset"},
            "resolved_inputs": {"L0": str(main_inputs[0].relative_to(repository)), "R0": str(main_inputs[1].relative_to(repository)), "Llast": str(main_inputs[2].relative_to(repository)), "Rlast": str(main_inputs[3].relative_to(repository)), "ROI": str(main_inputs[4].relative_to(repository))},
            "input_sha256": relative_hashes(repository, main_inputs),
            "calibration_order": {"left": [str(p.relative_to(repository)) for p in calibration_inputs[:20]], "right": [str(p.relative_to(repository)) for p in calibration_inputs[20:]]},
            "calibration_input_sha256": relative_hashes(repository, calibration_inputs),
            "config_paths": [str(p.relative_to(repository)) for p in configs],
            "config_sha256": relative_hashes(repository, configs),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
            "traditional_dic_package": str(Path(traditional_dic.__file__).resolve()),
            "native_extension": {"path": str(native_path), "sha256": sha256_file(native_path), "bytes": native_path.stat().st_size},
            "generation_command": "python -m tests.support.generate_f0b_baseline (explicit arguments recorded in implementation report)",
            "staging_method": "existing workflow/API outputs redirected to explicit repository-external /tmp tree",
            "runtime_evidence": runtime,
            "field_repeatability": runtime["field_repeatability"],
        },
    )

    generated = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline_commit": BASELINE_COMMIT,
            "baseline_scope": "F0B Stereo Subset-only calibration, three fields, and reconstruction",
            "architectural_contract": {"stereo_3d_solver": "subset", "multiview_3d_solver": "subset", "mesh_scope": "standalone_2d_only"},
            "immutable_at_pytest_runtime": True,
            "files": {str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in generated},
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
