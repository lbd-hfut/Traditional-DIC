from pathlib import Path

import pytest

from traditional_dic.case import CaseResolutionError, CaseSpec, resolve_case, resolve_mono_case, resolve_multiview_case, resolve_stereo_case


pytestmark = pytest.mark.unit


def _touch_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_mono_resolution_is_shared_and_explicit(repository_root: Path) -> None:
    subset = resolve_case("subset_2d", paths_config=repository_root / "config/case_paths.yaml", case_key="mono_2d", repository_root=repository_root)
    mesh = resolve_case("mesh_2d", paths_config=repository_root / "config/case_paths.yaml", case_key="mono_2d_01", repository_root=repository_root)
    assert subset.frame("reference").path.name == "001.bmp"
    assert [frame.path.name for frame in subset.frames if frame.role == "deformed"] == ["002.bmp"]
    assert subset.roi.mode == "explicit_image"
    assert mesh.frame("reference").path.name == "test1_000.bmp"
    assert mesh.frame("roi").path.name == "test1_010.bmp"


def test_case_spec_is_an_unresolved_serializable_contract(repository_root: Path) -> None:
    spec = CaseSpec("subset_2d", paths_config=repository_root / "config/case_paths.yaml", case_key="mono_2d")
    resolved = resolve_case(spec, repository_root=repository_root)
    assert resolved.workflow_kind == "subset_2d"
    assert spec.to_dict(repository_root=repository_root)["paths_config"] == "config/case_paths.yaml"


def test_mono_rejects_insufficient_images(tmp_path: Path) -> None:
    _touch_images(tmp_path, ["001.bmp", "002.bmp"])
    with pytest.raises(CaseResolutionError, match="INSUFFICIENT_IMAGES"):
        resolve_mono_case(tmp_path)


def test_stereo_roles_and_calibration_order(repository_root: Path) -> None:
    case = resolve_case("stereo_3d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    assert {frame.role: frame.path.name for frame in case.frames} == {
        "left_reference": "00_L.bmp", "right_reference": "00_R.bmp",
        "left_deformed": "04_L.bmp", "right_deformed": "04_R.bmp",
    }
    assert len(case.calibration_inputs) == 20
    assert case.calibration_inputs[0].paths[0].name == "01.bmp"
    assert case.calibration_inputs[-1].paths[1].name == "20.bmp"


def test_multiview_camera_order_and_frames(repository_root: Path) -> None:
    case = resolve_case("multiview_3d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    assert [camera.camera_id for camera in case.cameras] == [f"cam_{i}" for i in range(12)]
    assert all(camera.reference.path.name == "001.bmp" for camera in case.cameras)
    assert all(camera.deformed.path.name == "002.bmp" for camera in case.cameras)
    assert len(case.calibration_inputs) == 12
    assert case.roi.mode == "auto"


def test_multiview_last_image_roi_mode(tmp_path: Path) -> None:
    root = tmp_path / "case"
    for camera in ("cam_0", "cam_1"):
        _touch_images(root / "images" / camera, ["001.bmp", "002.bmp", "003.bmp"])
        _touch_images(root / "calibrate_images" / camera, ["001.bmp"])
    case = resolve_multiview_case(root, config={"images": {"root": "images"}, "roi": {"mode": "last_image"}, "calibration": {"chessboard_dir": "calibrate_images"}})
    assert case.roi.mode == "last_image"
    assert all(camera.deformed.path.name == "002.bmp" for camera in case.cameras)
    assert all(camera.roi and camera.roi.path.name == "003.bmp" for camera in case.cameras)


def test_multiview_explicit_and_none_roi_modes(tmp_path: Path) -> None:
    root = tmp_path / "case"
    for camera in ("cam_0", "cam_1"):
        _touch_images(root / "images" / camera, ["001.bmp", "002.bmp"])
        _touch_images(root / "calibrate_images" / camera, ["001.bmp"])
    config = {"images": {"root": "images"}, "calibration": {"chessboard_dir": "calibrate_images"}}
    none_case = resolve_multiview_case(root, config={**config, "roi": {"mode": "none"}})
    assert none_case.roi.mode == "none"
    roi = root / "roi.bmp"
    roi.touch()
    explicit_case = resolve_multiview_case(root, config={**config, "roi": {"mode": "explicit_image", "path": "roi.bmp"}})
    assert explicit_case.roi.path == roi
