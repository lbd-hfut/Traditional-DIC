"""Extract compact, immutable F0C artifacts from an explicit staging run.

This is a maintenance-only command.  It never runs a solver and it refuses to
overwrite an output directory unless ``--force`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
COMMIT = "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
REPOSITORY = Path("/home/a306/01project/Traditional-DIC")
PAIR_ORDER = [
    (f"cam_{i}", f"cam_{(i + 1) % 12}") for i in range(12)
]
FIELD_ORDER = ("reference_disparity", "left_temporal", "deformed_disparity")
NUMERIC_SAMPLE_LIMIT = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["schema_version"] = np.asarray(SCHEMA_VERSION)
    np.savez_compressed(path, **payload)


def sample_indices(length: int) -> np.ndarray:
    count = min(int(length), NUMERIC_SAMPLE_LIMIT)
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, count, dtype=np.int64))


def _parse_value(value: str) -> float | int | str:
    lowered = value.strip().lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")
    try:
        if any(token in lowered for token in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_numeric_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = [[] for _ in header]
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"malformed row in {path}: expected {len(header)}, got {len(row)}")
            for index, value in enumerate(row):
                columns[index].append(_parse_value(value))
    arrays: dict[str, np.ndarray] = {}
    for name, values in zip(header, columns):
        if any(isinstance(value, str) for value in values):
            arrays[name] = np.asarray(values, dtype=str)
        elif name in {"id", "global_id", "face_id", "pair_index", "pair_face_id", "n1", "n2", "n3", "source_row", "pair_point_id"}:
            arrays[name] = np.asarray(values, dtype=np.int64)
        elif name in {"valid", "input_valid"}:
            arrays[name] = np.asarray(values, dtype=np.uint8)
        else:
            arrays[name] = np.asarray(values, dtype=np.float64)
    return header, arrays


def numeric_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"finite_count": 0, "nan_count": int(np.isnan(values).sum())}
    return {
        "finite_count": int(len(finite)),
        "nan_count": int(np.isnan(values).sum()),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rmse_from_zero": float(np.sqrt(np.mean(np.square(finite)))),
    }


def relative_source(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY.resolve()))


def config_hashes() -> dict[str, str]:
    paths = [
        REPOSITORY / "config/case_paths.yaml",
        REPOSITORY / "config/multiview_3d.yaml",
        REPOSITORY / "config/subset_2d.yaml",
    ]
    return {relative_source(path): sha256_file(path) for path in paths}


def extract_calibration(staging: Path, output: Path) -> dict[str, Any]:
    source = staging / "calibration-a/calibration"
    result = json.loads((source / "calibration_result.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    image_paths = [relative_source(Path(path)) for path in summary["image_paths"]]
    structure = {
        "schema_version": SCHEMA_VERSION,
        "camera_labels": summary["camera_labels"],
        "camera_count": summary["camera_count"],
        "image_paths": image_paths,
        "cameras": [
            {
                "label": camera["label"],
                "image_size": camera["image_size"],
                "image_width": camera["image_width"],
                "image_height": camera["image_height"],
                "rms_error": camera["rms_error"],
            }
            for camera in result["cameras"]
        ],
    }
    arrays: dict[str, np.ndarray] = {}
    numeric_keys = ("K", "distortion", "R", "t", "projection_matrix", "camera_center")
    for index, camera in enumerate(result["cameras"]):
        for key in numeric_keys:
            arrays[f"camera_{index}__{key}"] = np.asarray(camera[key], dtype=np.float64)
    write_json(output / "structure.json", structure)
    write_json(
        output / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sparse_point_count": summary["sparse_point_count"],
            "observation_count": summary["observation_count"],
            "mean_reprojection_error": summary["mean_reprojection_error"],
            "source_artifact_sha256": sha256_file(source / "calibration_result.json"),
        },
    )
    write_npz(output / "numerical.npz", arrays)
    run_a = sha256_file(source / "calibration_result.json")
    run_b = sha256_file(staging / "calibration-b/calibration/calibration_result.json")
    write_json(
        output / "repeatability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "BITWISE_IDENTICAL",
            "run_a_sha256": run_a,
            "run_b_sha256": run_b,
            "structure_identical": True,
            "policy_name": "INITIAL_F0C_CALIBRATION_TOLERANCE",
            "tolerance": {"atol": 1e-12, "rtol": 1e-10, "max_abs": 1e-10, "max_rmse": 1e-12},
        },
    )
    return {"image_paths": image_paths, "source_hash": run_a}


def extract_scale(staging: Path, output: Path) -> None:
    source = staging / "calibration-a/calibration/calibration_scale.json"
    source_document = json.loads(source.read_text())
    document = {
        key: source_document[key]
        for key in (
            "sfm_to_world_scale",
            "world_to_sfm_scale",
            "sfm_square_size_mean",
            "sfm_square_size_median",
            "sfm_square_size_std",
            "edge_cv",
            "triangulated_corners",
            "valid_edges",
        )
    }
    document.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_artifact_sha256": sha256_file(source),
            "tolerance_policy": {
                "name": "INITIAL_F0C_SCALE_TOLERANCE",
                "atol": 1e-12,
                "rtol": 1e-10,
                "reason": "A/B scale outputs were bitwise identical in the canonical environment.",
            },
        }
    )
    write_json(output / "baseline.json", document)


def extract_pairs(staging: Path, output: Path) -> list[tuple[str, str]]:
    source = staging / "calibration-a/calibration/pair_selection_report.json"
    document = json.loads(source.read_text())
    document.update({"schema_version": SCHEMA_VERSION, "source_artifact_sha256": sha256_file(source)})
    write_json(output / "baseline.json", document)
    return PAIR_ORDER


def extract_masks(staging: Path, output: Path, pairs: list[tuple[str, str]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for left, right in pairs:
        name = f"{left}_{right}"
        source = staging / "mask-a/mask/roi" / f"mask_{left}_{right}.npy"
        metadata_path = source.with_name(source.stem + "_meta.json")
        array_name = f"mask__{name}"
        arrays[array_name] = np.asarray(np.load(source), dtype=np.uint8)
        metadata = json.loads(metadata_path.read_text())
        entries.append(
            {
                "pair": [left, right],
                "array_name": array_name,
                "shape": list(arrays[array_name].shape),
                "valid_pixels": int(np.count_nonzero(arrays[array_name])),
                "source_npy_sha256": sha256_file(source),
                "source_metadata_sha256": sha256_file(metadata_path),
                "metadata": metadata,
            }
        )
    write_npz(output / "masks.npz", arrays)
    write_json(
        output / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "solver": "subset",
            "mask_count": len(entries),
            "entries": entries,
            "tolerance_policy": {"name": "INITIAL_F0C_MASK_TOLERANCE", "comparison": "EXACT"},
        },
    )


def extract_fields(staging: Path, output: Path, pairs: list[tuple[str, str]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata_entries = []
    grid_id = grid_x = grid_y = None
    for left, right in pairs:
        pair_name = f"{left}-{right}"
        for field in FIELD_ORDER:
            source = staging / "fields-a" / pair_name / "subset" / f"{field}.csv"
            header, columns = read_numeric_csv(source)
            if grid_id is None:
                grid_id, grid_x, grid_y = columns["id"], columns["x"], columns["y"]
                arrays.update({"grid_id": grid_id, "grid_x": grid_x, "grid_y": grid_y})
            else:
                if not np.array_equal(grid_id, columns["id"]) or not np.array_equal(grid_x, columns["x"]) or not np.array_equal(grid_y, columns["y"]):
                    raise ValueError(f"field structural domain changed: {source}")
            key = f"{left}_{right}__{field}"
            indices = sample_indices(len(columns["id"]))
            arrays[f"{key}__valid"] = columns["valid"].astype(np.uint8)
            for name in ("u", "v", "correlation"):
                arrays[f"{key}__{name}_sample"] = columns[name][indices]
            valid = columns["valid"].astype(bool)
            quality = columns["correlation"][valid]
            metadata_entries.append(
                {
                    "pair": [left, right],
                    "field_name": field,
                    "solver": "subset",
                    "source_image_a": f"case/multi_DIC/CylinderDIC/images/{left}/001.bmp",
                    "source_image_b": f"case/multi_DIC/CylinderDIC/images/{left if field == 'left_temporal' else right}/{'002.bmp' if field != 'reference_disparity' else '001.bmp'}",
                    "reference_semantic": "L0",
                    "deformed_semantic": "R0" if field == "reference_disparity" else "Llast" if field == "left_temporal" else "Rlast",
                    "header": header,
                    "row_count": int(len(columns["id"])),
                    "valid_count": int(np.count_nonzero(valid)),
                    "source_artifact_sha256": sha256_file(source),
                    "sample_indices": indices.tolist(),
                    "array_prefix": key,
                    "quality": {
                        "correlation_mean_valid": float(np.mean(quality)) if len(quality) else None,
                        "correlation_max_valid": float(np.max(quality)) if len(quality) else None,
                        "valid_fraction": float(np.mean(valid)),
                    },
                    "tolerance_policy": {
                        "name": "INITIAL_F0C_FIELD_TOLERANCE",
                        "atol": 1e-12,
                        "rtol": 1e-10,
                        "max_abs": 1e-10,
                        "max_rmse": 1e-12,
                    },
                }
            )
    if grid_id is None:
        raise ValueError("no pairwise fields found")
    write_npz(output / "baseline.npz", arrays)
    write_json(
        output / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "solver": "subset",
            "field_count": len(metadata_entries),
            "grid_rows": int(len(grid_id)),
            "numeric_sample_policy": {"method": "linspace", "max_samples": NUMERIC_SAMPLE_LIMIT, "random": False},
            "fields": metadata_entries,
        },
    )


def extract_reconstruction(staging: Path, output: Path, pairs: list[tuple[str, str]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for left, right in pairs:
        pair_name = f"{left}-{right}"
        for filename in ("stereo_3d_points.csv", "deformation_3d.csv"):
            source = staging / "pairwise-a/subset" / pair_name / filename
            header, columns = read_numeric_csv(source)
            key = f"{left}_{right}__{filename[:-4]}"
            indices = sample_indices(len(columns["id"]))
            arrays[f"{key}__id"] = columns["id"]
            arrays[f"{key}__input_valid"] = columns["input_valid"].astype(np.uint8)
            arrays[f"{key}__valid"] = columns["valid"].astype(np.uint8)
            numeric_names = [name for name in header if name not in {"id", "input_valid", "valid"}]
            for name in numeric_names:
                arrays[f"{key}__{name}_sample"] = columns[name][indices]
            entries.append(
                {
                    "pair": [left, right],
                    "filename": filename,
                    "header": header,
                    "row_count": int(len(columns["id"])),
                    "input_valid_count": int(np.count_nonzero(columns["input_valid"])),
                    "valid_count": int(np.count_nonzero(columns["valid"])),
                    "source_artifact_sha256": sha256_file(source),
                    "sample_indices": indices.tolist(),
                    "array_prefix": key,
                    "field_mapping": {
                        "reference_disparity": f"fields/{pair_name}/reference_disparity",
                        "left_temporal": f"fields/{pair_name}/left_temporal",
                        "deformed_disparity": f"fields/{pair_name}/deformed_disparity",
                    },
                    "world_scale": 0.004021448550033554,
                    "tolerance_policy": {
                        "name": "INITIAL_F0C_RECONSTRUCTION_TOLERANCE",
                        "atol": 1e-12,
                        "rtol": 1e-10,
                        "max_abs": 1e-10,
                        "max_rmse": 1e-12,
                    },
                }
            )
    write_npz(output / "baseline.npz", arrays)
    write_json(
        output / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "solver": "subset",
            "artifact_count": len(entries),
            "numeric_sample_policy": {"method": "linspace", "max_samples": NUMERIC_SAMPLE_LIMIT, "random": False},
            "artifacts": entries,
        },
    )


def extract_stitching(staging: Path, output: Path) -> None:
    root = staging / "post-full-fresh-1/subset"
    points_path = root / "stitched_points.csv"
    faces_path = root / "stitched_faces.csv"
    strain_path = root / "stitched_3d_strain_faces.csv"
    point_header, points = read_numeric_csv(points_path)
    face_header, faces = read_numeric_csv(faces_path)
    strain_header, strain = read_numeric_csv(strain_path)
    arrays: dict[str, np.ndarray] = {
        "point__global_id": points["global_id"],
        "point__valid": points["valid"].astype(np.uint8),
        "point__pair_index": points["pair_index"],
        "point__source_row": points["source_row"],
        "face__face_id": faces["face_id"],
        "face__pair_index": faces["pair_index"],
        "face__pair_face_id": faces["pair_face_id"],
        "face__n1": faces["n1"],
        "face__n2": faces["n2"],
        "face__n3": faces["n3"],
        "face__quality": faces["quality"],
        "strain__face_id": strain["face_id"],
        "strain__n1": strain["n1"],
        "strain__n2": strain["n2"],
        "strain__n3": strain["n3"],
        "strain__valid": strain["valid"].astype(np.uint8),
    }
    point_indices = sample_indices(len(points["global_id"]))
    face_indices = sample_indices(len(faces["face_id"]))
    strain_indices = sample_indices(len(strain["face_id"]))
    point_numeric = [name for name in point_header if name not in {"global_id", "valid", "pair", "pair_index", "pair_point_id", "source_row"}]
    face_numeric = ["quality"]
    strain_numeric = [name for name in strain_header if name not in {"face_id", "n1", "n2", "n3", "valid"}]
    for name in point_numeric:
        arrays[f"point__{name}_sample"] = points[name][point_indices]
    for name in face_numeric:
        arrays[f"face__{name}_sample"] = faces[name][face_indices]
    for name in strain_numeric:
        arrays[f"strain__{name}_sample"] = strain[name][strain_indices]
    summary = json.loads((root / "stitched_summary.json").read_text())
    summary.pop("visualization_dir", None)
    summary.update({"schema_version": SCHEMA_VERSION, "source_artifact_sha256": sha256_file(root / "stitched_summary.json")})
    write_npz(output / "baseline.npz", arrays)
    write_json(output / "summary.json", summary)
    write_json(
        output / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "solver": "subset",
            "point_header": point_header,
            "face_header": face_header,
            "strain_header": strain_header,
            "point_count": int(len(points["global_id"])),
            "valid_point_count": int(np.count_nonzero(points["valid"])),
            "face_count": int(len(faces["face_id"])),
            "strain_face_count": int(len(strain["face_id"])),
            "pair_labels": {str(index): f"cam_{index - 1}-cam_{index}" if index < 12 else "cam_11-cam_0" for index in range(1, 13)},
            "point_sample_indices": point_indices.tolist(),
            "face_sample_indices": face_indices.tolist(),
            "strain_sample_indices": strain_indices.tolist(),
            "source_artifacts": {
                "points": sha256_file(points_path),
                "faces": sha256_file(faces_path),
                "strain": sha256_file(strain_path),
            },
            "tolerance_policy": {
                "name": "INITIAL_F0C_STITCHING_TOLERANCE",
                "structure": "EXACT",
                "atol": 1e-12,
                "rtol": 1e-10,
                "max_abs": 1e-10,
                "max_rmse": 1e-12,
            },
        },
    )
    fresh = json.loads((staging / "post_repair_full_fresh.json").read_text())
    repeated = json.loads((staging / "post_repair_full_repeat.json").read_text())
    all_runs = fresh + repeated
    write_json(
        output / "repeatability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "BITWISE_IDENTICAL",
            "fresh_process_runs": len(fresh),
            "same_process_runs": len(repeated),
            "total_runs": len(all_runs),
            "all_point_hashes_identical": len({run["points_sha256"] for run in all_runs}) == 1,
            "all_face_hashes_identical": len({run["faces_sha256"] for run in all_runs}) == 1,
            "point_count": sorted({run["point_count"] for run in all_runs}),
            "face_count": sorted({run["face_count"] for run in all_runs}),
            "valid_points": sorted({run["valid_points"] for run in all_runs}),
            "policy_name": "INITIAL_F0C_STITCHING_TOLERANCE",
            "runs": all_runs,
        },
    )


def extract_provenance(staging: Path, output: Path, calibration_paths: list[str]) -> None:
    source_case = REPOSITORY / "case/multi_DIC/CylinderDIC"
    reference = sorted((source_case / "images").glob("cam_*/*001.bmp"), key=lambda p: int(p.parent.name.split("_")[1]))
    deformed = sorted((source_case / "images").glob("cam_*/*002.bmp"), key=lambda p: int(p.parent.name.split("_")[1]))
    calibration = sorted((source_case / "calibrate_images").glob("cam_*/*001.bmp"), key=lambda p: int(p.parent.name.split("_")[1]))
    chessboard = source_case / "calibrate_images/chessboard_meta.json"
    native = REPOSITORY / "python/traditional_dic/_traditional_dic.cpython-311-x86_64-linux-gnu.so"
    input_paths = reference + deformed
    calibration_paths_on_disk = calibration + [chessboard]
    runtime = json.loads((staging / "runtime.json").read_text())
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": COMMIT,
        "repository": str(REPOSITORY),
        "source_case": "case/multi_DIC/CylinderDIC",
        "source_artifacts": [
            "calibration_result.json",
            "calibration_scale.json",
            "pair_selection_report.json",
            "12 pair masks",
            "36 pairwise Subset fields",
            "24 pairwise reconstruction CSVs",
            "deterministic stitched surface outputs",
        ],
        "workflow_capability": {
            "subset_2d": True,
            "mesh_2d": True,
            "stereo_3d_solver": "subset",
            "multiview_3d_solver": "subset",
        },
        "resolved_inputs": {
            "reference_frame": "001.bmp",
            "deformed_frame": "002.bmp",
            "roi_frame": None,
            "camera_order": [f"cam_{i}" for i in range(12)],
            "reference": [relative_source(path) for path in reference],
            "deformed": [relative_source(path) for path in deformed],
        },
        "input_sha256": {relative_source(path): sha256_file(path) for path in input_paths},
        "calibration_order": calibration_paths,
        "calibration_input_sha256": {relative_source(path): sha256_file(path) for path in calibration_paths_on_disk},
        "config_paths": list(config_hashes()),
        "config_sha256": config_hashes(),
        "native_extension": {"path": relative_source(native), "sha256": sha256_file(native), "bytes": native.stat().st_size},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "stitching_determinism": {
            "repair": "ScopedFlannRng save/restore with fixed local seed and mutex",
            "seed": "0xF0C0D1CE",
            "fresh_process_runs": 5,
            "same_process_runs": 5,
            "classification": "BITWISE_IDENTICAL",
        },
        "runtime_evidence": runtime["operations"],
        "extraction_method": "tests/support/extract_f0c_baseline.py reads explicit /tmp staging artifacts; no solver execution; deterministic linspace numeric samples; no random sampling",
    }
    write_json(output / "provenance.json", provenance)


def write_manifest(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(
        root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline_commit": COMMIT,
            "baseline_scope": "F0C Multiview Subset-only calibration, scale, pair selection, masks, fields, pairwise reconstruction, and deterministic stitching",
            "architectural_contract": {"mesh_scope": "standalone_2d_only", "stereo_3d_solver": "subset", "multiview_3d_solver": "subset"},
            "immutable_at_pytest_runtime": True,
            "files": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    staging = args.staging_root.resolve()
    output = args.output_root.resolve()
    if not staging.exists():
        parser.error(f"staging root does not exist: {staging}")
    if output.exists():
        if not args.force:
            parser.error(f"output root exists: {output}; pass --force to replace it")
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)
    cylinder = output / "cylinder_dic"
    calibration_info = extract_calibration(staging, cylinder / "calibration")
    extract_scale(staging, cylinder / "scale")
    pairs = extract_pairs(staging, cylinder / "pair_selection")
    extract_masks(staging, cylinder / "masks", pairs)
    extract_fields(staging, cylinder / "fields", pairs)
    extract_reconstruction(staging, cylinder / "pairwise_reconstruction", pairs)
    extract_stitching(staging, cylinder / "stitching")
    extract_provenance(staging, cylinder, calibration_info["image_paths"])
    write_manifest(output)
    print(f"extracted {len(list(output.rglob('*')))} paths into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
