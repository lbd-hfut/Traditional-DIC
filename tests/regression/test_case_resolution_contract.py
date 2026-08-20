import json
from pathlib import Path

import pytest

from traditional_dic.case import resolve_case


pytestmark = pytest.mark.regression


def _snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ring_subset_snapshot(repository_root: Path) -> None:
    case = resolve_case("subset_2d", paths_config=repository_root / "config/case_paths.yaml", case_key="mono_2d", repository_root=repository_root)
    expected = _snapshot(repository_root / "tests/data/f1/ring_subset.json")
    actual = case.to_dict(repository_root=repository_root)
    assert {key: actual[key] for key in ("workflow_kind", "case_root", "frames", "roi", "metadata")} == expected


def test_ring_mesh_snapshot(repository_root: Path) -> None:
    case = resolve_case("mesh_2d", paths_config=repository_root / "config/case_paths.yaml", case_key="mono_2d_01", repository_root=repository_root)
    expected = _snapshot(repository_root / "tests/data/f1/ring_mesh.json")
    actual = case.to_dict(repository_root=repository_root)
    assert {key: actual[key] for key in ("workflow_kind", "case_root", "frames", "roi", "metadata")} == expected


def test_stereo_snapshot(repository_root: Path) -> None:
    case = resolve_case("stereo_3d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    expected = _snapshot(repository_root / "tests/data/f1/stereo_plate.json")
    actual = case.to_dict(repository_root=repository_root)
    frames = {item["role"]: item["path"] for item in actual["frames"]}
    calibration = actual["calibration_inputs"]
    assert {"workflow_kind": actual["workflow_kind"], "case_root": actual["case_root"], "frames": frames, "roi": actual["roi"], "stereo_roles": actual["metadata"]["stereo_roles"]} == {"workflow_kind": expected["workflow_kind"], "case_root": expected["case_root"], "frames": expected["frames"], "roi": expected["roi"], "stereo_roles": expected["stereo_roles"]}
    assert len(calibration) == expected["calibration"]["count"]
    assert calibration[0]["paths"] == expected["calibration"]["first"]
    assert calibration[-1]["paths"] == expected["calibration"]["last"]


def test_multiview_snapshot(repository_root: Path) -> None:
    case = resolve_case("multiview_3d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    expected = _snapshot(repository_root / "tests/data/f1/multiview_cylinder.json")
    actual = case.to_dict(repository_root=repository_root)
    assert actual["workflow_kind"] == expected["workflow_kind"]
    assert actual["case_root"] == expected["case_root"]
    assert [camera["camera_id"] for camera in actual["cameras"]] == expected["camera_ids"]
    assert actual["metadata"]["reference_frame"] == expected["reference_frame"]
    assert actual["metadata"]["deformed_frame"] == expected["deformed_frame"]
    assert actual["roi"] == expected["roi"]
    assert len(actual["calibration_inputs"]) == expected["calibration"]["count"]
    assert actual["calibration_inputs"][0]["paths"][0] == expected["calibration"]["first"]
    assert actual["calibration_metadata"] == [expected["calibration"]["metadata"]]
