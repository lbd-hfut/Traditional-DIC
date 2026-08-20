from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline
from tests.support.regression_compare import assert_exact


pytestmark = pytest.mark.regression
HEADER = [
    "id", "x_l0", "y_l0", "x_r0", "y_r0", "x_l1", "y_l1", "x_r1", "y_r1",
    "X0", "Y0", "Z0", "X1", "Y1", "Z1", "Ux", "Uy", "Uz", "Umag",
    "reprojection_error_ref", "reprojection_error_def", "combined_correlation", "input_valid", "valid",
]


def test_stereo_reconstruction_structure_displacement_and_quality(repository_root: Path) -> None:
    root = repository_root / "tests/data/f0b/stereo_plate/reconstruction"
    arrays = load_npz_baseline(root / "baseline.npz", required_arrays=HEADER)
    summary = load_json_baseline(
        root / "summary.json",
        required_keys=("header", "row_count", "input_valid_count", "final_valid_count", "valid_fraction", "mean_reprojection_error_ref", "mean_reprojection_error_def", "mean_combined_correlation", "workflow_summary"),
    )
    assert summary["header"] == HEADER
    assert summary["row_count"] == 7654
    assert summary["input_valid_count"] == 7654
    assert summary["final_valid_count"] == 7654
    assert summary["valid_fraction"] == 1.0
    assert summary["workflow_summary"]["total_points"] == 7654
    assert summary["workflow_summary"]["valid_points"] == 7654
    assert summary["workflow_summary"]["world_scale"] == 1.0
    assert_exact(arrays["input_valid"], np.ones(7654, dtype=int), field="reconstruction.input_valid")
    assert_exact(arrays["valid"], np.ones(7654, dtype=int), field="reconstruction.valid")
    for field in ("X0", "Y0", "Z0", "X1", "Y1", "Z1", "Ux", "Uy", "Uz", "Umag", "reprojection_error_ref", "reprojection_error_def", "combined_correlation"):
        assert np.isfinite(arrays[field]).all(), field
    computed_magnitude = np.sqrt(arrays["Ux"] ** 2 + arrays["Uy"] ** 2 + arrays["Uz"] ** 2)
    np.testing.assert_allclose(arrays["Umag"], computed_magnitude, atol=1e-12, rtol=1e-10)


def test_stereo_reconstruction_repeatability(repository_root: Path) -> None:
    repeatability = load_json_baseline(
        repository_root / "tests/data/f0b/stereo_plate/reconstruction/repeatability.json",
        required_keys=("classification", "run_a_sha256", "run_b_sha256", "policy_name", "tolerance", "fields"),
    )
    assert repeatability["classification"] == "BITWISE_IDENTICAL"
    assert repeatability["run_a_sha256"] == repeatability["run_b_sha256"]
    assert repeatability["policy_name"] == "INITIAL_F0B_RECONSTRUCTION_TOLERANCE"
    for field, metrics in repeatability["fields"].items():
        assert metrics["max_abs_diff"] == 0.0, field
        assert metrics["rmse"] == 0.0, field


def test_stereo_world_scale_and_field_mapping_contract(repository_root: Path) -> None:
    provenance = load_json_baseline(
        repository_root / "tests/data/f0b/stereo_plate/reconstruction/provenance.json",
        required_keys=("calibration_source", "field_mapping", "world_scale", "world_scale_source", "world_scale_application", "quality_metric", "max_znssd", "max_reprojection_error_px", "remove_rigid_body_motion"),
    )
    assert provenance["world_scale"] == 1.0
    assert provenance["world_scale_application"] == "StereoDICOptions.world_scale in reconstruct_from_field_files"
    assert provenance["field_mapping"] == {
        "reference_disparity": "stereo_plate/fields/reference_disparity.npz",
        "left_temporal": "stereo_plate/fields/left_temporal.npz",
        "deformed_disparity": "stereo_plate/fields/deformed_disparity.npz",
    }
    assert provenance["quality_metric"] == "znssd"
    assert provenance["max_znssd"] == 2.0
    assert provenance["max_reprojection_error_px"] == 5.0
    assert provenance["remove_rigid_body_motion"] is False
