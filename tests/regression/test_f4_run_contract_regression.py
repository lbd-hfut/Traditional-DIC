import json
from pathlib import Path

import pytest

from traditional_dic.run_contract import FINAL_STATUSES, SCHEMA_VERSION


pytestmark = pytest.mark.regression


EXPECTED = {
    "subset_ring": {
        "workflow_kind": "subset_2d",
        "primary_result_type": "2d_displacement",
        "metric_keys": ["total_points", "valid_points", "valid_fraction"],
    },
    "mesh_case01_q4": {
        "workflow_kind": "mesh_2d",
        "primary_result_type": "2d_mesh_displacement",
        "metric_keys": ["element_types", "node_count", "element_count"],
    },
    "stereo_plate": {
        "workflow_kind": "stereo_3d",
        "primary_result_type": "stereo_3d_displacement",
        "metric_keys": ["input_valid", "final_valid", "valid_fraction"],
    },
    "multiview_cylinder": {
        "workflow_kind": "multiview_3d",
        "primary_result_type": "stitched_3d_surface",
        "metric_keys": ["camera_count", "pair_count", "field_count", "stitched_point_count"],
    },
}


def test_f4_metadata_snapshots_define_common_contract(repository_root: Path) -> None:
    snapshot_root = repository_root / "tests/data/f4"
    capability = {
        "subset_2d": True,
        "mesh_2d": True,
        "stereo_3d_solver": "subset",
        "multiview_3d_solver": "subset",
    }
    assert set(FINAL_STATUSES) == {
        "SUCCESS",
        "SUCCESS_WITH_WARNINGS",
        "PARTIAL_SUCCESS",
        "VALIDATION_FAILED",
        "EXECUTION_FAILED",
    }
    for name, expected in EXPECTED.items():
        payload = json.loads((snapshot_root / name / "contract.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["workflow_kind"] == expected["workflow_kind"]
        assert payload["primary_result_type"] == expected["primary_result_type"]
        assert payload["capability_contract"] == capability
        assert payload["metric_keys"] == expected["metric_keys"]
        assert payload["dynamic_fields_excluded"] == ["run_id", "timestamps", "duration_seconds"]
