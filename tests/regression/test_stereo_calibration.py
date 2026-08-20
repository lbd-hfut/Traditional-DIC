from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline
from tests.support.provenance import verify_relative_hashes


pytestmark = pytest.mark.regression


def test_f0b_manifest_integrity(repository_root: Path) -> None:
    root = repository_root / "tests/data/f0b"
    manifest = load_json_baseline(
        root / "manifest.json",
        required_keys=("baseline_commit", "baseline_scope", "architectural_contract", "immutable_at_pytest_runtime", "files"),
    )
    assert manifest["baseline_commit"] == "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
    assert manifest["architectural_contract"] == {
        "mesh_scope": "standalone_2d_only",
        "multiview_3d_solver": "subset",
        "stereo_3d_solver": "subset",
    }
    assert manifest["immutable_at_pytest_runtime"] is True
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == actual_files
    verify_relative_hashes(root, {name: value["sha256"] for name, value in manifest["files"].items()})


def test_stereo_calibration_structure_and_numerical_baseline(repository_root: Path) -> None:
    root = repository_root / "tests/data/f0b/stereo_plate/calibration"
    structure = load_json_baseline(
        root / "structure.json",
        required_keys=("board", "world_scale", "cameras", "detections", "kept_pair_indices", "rejected_pair_indices", "rejection_reasons", "outlier_rejection_applied"),
    )
    summary = load_json_baseline(root / "summary.json", required_keys=("rms_error", "initial_rms_error", "pair_count", "accepted_pair_count", "rejected_pair_count", "left_found_count", "right_found_count", "image_dimensions"))
    required = (
        "left__K", "left__distortion", "left__R", "left__t", "left__projection_matrix", "left__camera_center",
        "right__K", "right__distortion", "right__R", "right__t", "right__projection_matrix", "right__camera_center",
        "R_lr", "t_lr", "essential", "fundamental", "per_pair_errors", "left__detection_points", "right__detection_points",
    )
    arrays = load_npz_baseline(root / "numerical.npz", required_arrays=required)
    assert structure["board"] == {"type": "chessboard", "rows": 8, "cols": 11, "spacing": 5.0}
    assert structure["world_scale"] == 1.0
    assert structure["cameras"]["left"]["image_size"] == [1440, 1080]
    assert structure["cameras"]["right"]["image_size"] == [1440, 1080]
    assert structure["kept_pair_indices"] == [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
    assert structure["rejected_pair_indices"] == [0, 3, 11]
    assert structure["outlier_rejection_applied"] is True
    assert summary["pair_count"] == 20
    assert summary["accepted_pair_count"] == 17
    assert summary["rejected_pair_count"] == 3
    assert summary["left_found_count"] == summary["right_found_count"] == 20
    assert arrays["left__K"].shape == arrays["right__K"].shape == (3, 3)
    assert arrays["R_lr"].shape == (3, 3)
    assert arrays["t_lr"].shape == (3,)
    assert arrays["left__detection_points"].shape == arrays["right__detection_points"].shape == (20, 88, 2)
    assert np.isfinite(arrays["left__K"]).all()


def test_calibration_repeatability_evidence(repository_root: Path) -> None:
    repeatability = load_json_baseline(
        repository_root / "tests/data/f0b/stereo_plate/calibration/repeatability.json",
        required_keys=("classification", "run_a_sha256", "run_b_sha256", "structure_identical", "policy_name", "fields"),
    )
    assert repeatability["classification"] == "NUMERICALLY_CLOSE"
    assert repeatability["run_a_sha256"] != repeatability["run_b_sha256"]
    assert repeatability["structure_identical"] is True
    assert repeatability["policy_name"] == "INITIAL_F0B_CALIBRATION_TOLERANCE"
    assert repeatability["fields"]
    for field, evidence in repeatability["fields"].items():
        assert evidence["within_policy"] is True, field
        assert evidence["max_abs_diff"] <= evidence["atol"], field
        assert evidence["rmse"] <= evidence["max_rmse"], field
        assert evidence["reason"]


def test_f0b_provenance_hashes_and_capability_contract(repository_root: Path) -> None:
    provenance = load_json_baseline(
        repository_root / "tests/data/f0b/stereo_plate/provenance.json",
        required_keys=("baseline_commit", "repository", "canonical_case", "workflow_capability", "resolved_inputs", "input_sha256", "calibration_order", "calibration_input_sha256", "config_paths", "config_sha256", "native_extension", "runtime_evidence"),
    )
    assert provenance["workflow_capability"] == {
        "subset_2d": True,
        "mesh_2d": True,
        "stereo_3d_solver": "subset",
        "multiview_3d_solver": "subset",
    }
    assert provenance["resolved_inputs"] == {
        "L0": "case/stereo_DIC/plate_center_load/cam1/00_L.bmp",
        "R0": "case/stereo_DIC/plate_center_load/cam2/00_R.bmp",
        "Llast": "case/stereo_DIC/plate_center_load/cam1/04_L.bmp",
        "Rlast": "case/stereo_DIC/plate_center_load/cam2/04_R.bmp",
        "ROI": "case/stereo_DIC/plate_center_load/ROI.bmp",
    }
    assert len(provenance["calibration_order"]["left"]) == 20
    assert len(provenance["calibration_order"]["right"]) == 20
    verify_relative_hashes(repository_root, provenance["input_sha256"])
    verify_relative_hashes(repository_root, provenance["calibration_input_sha256"])
    verify_relative_hashes(repository_root, provenance["config_sha256"])
    native = Path(provenance["native_extension"]["path"])
    assert native.stat().st_size == provenance["native_extension"]["bytes"]
