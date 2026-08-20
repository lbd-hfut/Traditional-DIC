from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline
from tests.support.provenance import verify_relative_hashes
from tests.support.regression_compare import assert_exact


pytestmark = pytest.mark.regression
COMMIT = "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
PAIRS = [(f"cam_{i}", f"cam_{(i + 1) % 12}") for i in range(12)]
FIELDS = ("reference_disparity", "left_temporal", "deformed_disparity")


def _root(repository_root: Path) -> Path:
    return repository_root / "tests/data/f0c"


def test_f0c_manifest_integrity(repository_root: Path) -> None:
    root = _root(repository_root)
    manifest = load_json_baseline(
        root / "manifest.json",
        required_keys=("baseline_commit", "baseline_scope", "architectural_contract", "immutable_at_pytest_runtime", "files"),
    )
    assert manifest["baseline_commit"] == COMMIT
    assert manifest["architectural_contract"] == {
        "mesh_scope": "standalone_2d_only",
        "stereo_3d_solver": "subset",
        "multiview_3d_solver": "subset",
    }
    assert manifest["immutable_at_pytest_runtime"] is True
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == actual_files
    verify_relative_hashes(root, {name: value["sha256"] for name, value in manifest["files"].items()})


def test_f0c_provenance_and_capability_contract(repository_root: Path) -> None:
    root = _root(repository_root)
    provenance = load_json_baseline(
        root / "cylinder_dic/provenance.json",
        required_keys=(
            "baseline_commit", "repository", "source_case", "source_artifacts",
            "workflow_capability", "resolved_inputs", "input_sha256",
            "calibration_order", "calibration_input_sha256", "config_sha256",
            "native_extension", "environment", "extraction_method",
        ),
    )
    assert provenance["baseline_commit"] == COMMIT
    assert provenance["source_case"] == "case/multi_DIC/CylinderDIC"
    assert provenance["workflow_capability"] == {
        "subset_2d": True,
        "mesh_2d": True,
        "stereo_3d_solver": "subset",
        "multiview_3d_solver": "subset",
    }
    assert provenance["resolved_inputs"]["reference_frame"] == "001.bmp"
    assert provenance["resolved_inputs"]["deformed_frame"] == "002.bmp"
    assert provenance["resolved_inputs"]["roi_frame"] is None
    assert len(provenance["resolved_inputs"]["reference"]) == 12
    assert len(provenance["resolved_inputs"]["deformed"]) == 12
    verify_relative_hashes(repository_root, provenance["input_sha256"])
    verify_relative_hashes(repository_root, provenance["calibration_input_sha256"])
    verify_relative_hashes(repository_root, provenance["config_sha256"])
    native = repository_root / provenance["native_extension"]["path"]
    assert native.stat().st_size == provenance["native_extension"]["bytes"]
    assert provenance["stitching_determinism"]["classification"] == "BITWISE_IDENTICAL"
    assert provenance["stitching_determinism"]["fresh_process_runs"] == 5
    assert provenance["stitching_determinism"]["same_process_runs"] == 5


def test_f0c_calibration_and_scale_baselines(repository_root: Path) -> None:
    root = _root(repository_root) / "cylinder_dic"
    structure = load_json_baseline(root / "calibration/structure.json", required_keys=("camera_labels", "camera_count", "image_paths", "cameras"))
    arrays = load_npz_baseline(root / "calibration/numerical.npz")
    repeatability = load_json_baseline(root / "calibration/repeatability.json", required_keys=("classification", "run_a_sha256", "run_b_sha256", "policy_name"))
    scale = load_json_baseline(root / "scale/baseline.json", required_keys=("sfm_to_world_scale", "world_to_sfm_scale", "tolerance_policy"))
    assert structure["camera_count"] == 12
    assert structure["camera_labels"] == [f"cam_{i}" for i in range(12)]
    assert len(structure["image_paths"]) == 12
    assert arrays["camera_0__K"].shape == (3, 3)
    assert arrays["camera_11__R"].shape == (3, 3)
    assert repeatability["classification"] == "BITWISE_IDENTICAL"
    assert repeatability["run_a_sha256"] == repeatability["run_b_sha256"]
    assert scale["sfm_to_world_scale"] == 248.66661541440243
    assert scale["world_to_sfm_scale"] == 0.004021448550033554
    assert scale["tolerance_policy"]["name"] == "INITIAL_F0C_SCALE_TOLERANCE"


def test_f0c_pair_selection_and_masks(repository_root: Path) -> None:
    root = _root(repository_root) / "cylinder_dic"
    selection = load_json_baseline(root / "pair_selection/baseline.json", required_keys=("pair_names", "is_circular", "mode"))
    mask_metadata = load_json_baseline(root / "masks/metadata.json", required_keys=("mask_count", "entries", "tolerance_policy"))
    masks = load_npz_baseline(root / "masks/masks.npz")
    assert selection["mode"] == "auto_natural"
    assert selection["is_circular"] is True
    assert selection["pair_names"] == [list(pair) for pair in PAIRS]
    assert mask_metadata["mask_count"] == 12
    for left, right in PAIRS:
        array = masks[f"mask__{left}_{right}"]
        assert array.shape == (1080, 1440)
        assert set(np.unique(array)) <= {0, 1}
    assert mask_metadata["tolerance_policy"]["comparison"] == "EXACT"


def test_f0c_pairwise_subset_fields(repository_root: Path) -> None:
    root = _root(repository_root) / "cylinder_dic/fields"
    metadata = load_json_baseline(root / "metadata.json", required_keys=("solver", "field_count", "grid_rows", "fields", "numeric_sample_policy"))
    arrays = load_npz_baseline(root / "baseline.npz", required_arrays=("grid_id", "grid_x", "grid_y"))
    assert metadata["solver"] == "subset"
    assert metadata["field_count"] == 36
    assert metadata["grid_rows"] == 97200
    assert arrays["grid_id"].shape == (97200,)
    assert_exact(arrays["grid_id"], np.arange(1, 97201), field="f0c.fields.grid_id")
    seen = {(tuple(entry["pair"]), entry["field_name"]) for entry in metadata["fields"]}
    assert seen == {(pair, field) for pair in PAIRS for field in FIELDS}
    for entry in metadata["fields"]:
        prefix = entry["array_prefix"]
        assert entry["solver"] == "subset"
        assert entry["reference_semantic"] == "L0"
        assert entry["deformed_semantic"] in {"R0", "Llast", "Rlast"}
        assert entry["row_count"] == 97200
        assert arrays[f"{prefix}__valid"].shape == (97200,)
        assert set(np.unique(arrays[f"{prefix}__valid"])) <= {0, 1}
        for name in ("u", "v", "correlation"):
            assert np.isfinite(arrays[f"{prefix}__{name}_sample"]).all()
        assert entry["tolerance_policy"]["name"] == "INITIAL_F0C_FIELD_TOLERANCE"


def test_f0c_pairwise_reconstruction(repository_root: Path) -> None:
    root = _root(repository_root) / "cylinder_dic/pairwise_reconstruction"
    metadata = load_json_baseline(root / "metadata.json", required_keys=("solver", "artifact_count", "artifacts", "numeric_sample_policy"))
    arrays = load_npz_baseline(root / "baseline.npz")
    assert metadata["solver"] == "subset"
    assert metadata["artifact_count"] == 24
    assert sum(entry["row_count"] for entry in metadata["artifacts"] if entry["filename"] == "stereo_3d_points.csv") == 180586
    assert sum(entry["valid_count"] for entry in metadata["artifacts"] if entry["filename"] == "stereo_3d_points.csv") == 180586
    for entry in metadata["artifacts"]:
        prefix = entry["array_prefix"]
        assert arrays[f"{prefix}__id"].shape == (entry["row_count"],)
        assert arrays[f"{prefix}__input_valid"].shape == (entry["row_count"],)
        assert arrays[f"{prefix}__valid"].shape == (entry["row_count"],)
        assert_exact(arrays[f"{prefix}__input_valid"], arrays[f"{prefix}__valid"], field=f"{prefix}.validity")
        assert entry["tolerance_policy"]["name"] == "INITIAL_F0C_RECONSTRUCTION_TOLERANCE"


def test_f0c_deterministic_stitching_baseline(repository_root: Path) -> None:
    root = _root(repository_root) / "cylinder_dic/stitching"
    metadata = load_json_baseline(root / "metadata.json", required_keys=("point_count", "valid_point_count", "face_count", "point_header", "face_header", "tolerance_policy"))
    summary = load_json_baseline(root / "summary.json", required_keys=("point_count", "face_count", "raw_face_count", "cleaned_removed_points", "cleaned_removed_faces"))
    repeatability = load_json_baseline(root / "repeatability.json", required_keys=("classification", "total_runs", "fresh_process_runs", "same_process_runs", "all_point_hashes_identical", "all_face_hashes_identical"))
    arrays = load_npz_baseline(root / "baseline.npz")
    assert metadata["point_count"] == 180586
    assert metadata["valid_point_count"] == 178848
    assert metadata["face_count"] == 350873
    assert summary["raw_face_count"] == 355396
    assert summary["face_count"] == 350873
    assert arrays["point__global_id"].shape == (180586,)
    assert arrays["point__valid"].shape == (180586,)
    assert arrays["face__n1"].shape == (350873,)
    assert arrays["face__n2"].shape == arrays["face__n3"].shape == (350873,)
    assert int(np.min(arrays["face__n1"])) >= 1
    assert int(np.max(arrays["face__n3"])) <= 180586
    assert repeatability["classification"] == "BITWISE_IDENTICAL"
    assert repeatability["total_runs"] == 10
    assert repeatability["fresh_process_runs"] == 5
    assert repeatability["same_process_runs"] == 5
    assert repeatability["all_point_hashes_identical"] is True
    assert repeatability["all_face_hashes_identical"] is True
    assert metadata["tolerance_policy"]["structure"] == "EXACT"
