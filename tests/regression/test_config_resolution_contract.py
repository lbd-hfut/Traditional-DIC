import json
from pathlib import Path

import pytest

from traditional_dic.config_resolver import resolve_config


pytestmark = pytest.mark.regression


def _snapshot(root: Path, name: str) -> dict:
    return json.loads((root / "tests/data/f2" / name).read_text(encoding="utf-8"))


def _effective(config) -> dict:
    values = config.values
    if config.workflow_kind == "subset_2d":
        return {
            "subset_radius": values["subset"]["radius"],
            "shape_order": values["shape_function"]["order"],
            "optimization_method": values["optimization"]["method"],
            "max_iterations": values["optimization"]["max_iterations"],
            "convergence_threshold": values["optimization"]["convergence_threshold"],
            "criterion": values["correlation"]["criterion"],
            "search_radius": values["initialization"]["integer_search"]["search_radius"],
            "seed_subset_radius": values["initialization"]["integer_search"]["subset_radius"],
            "seed_count": values["seed_selection"]["seed_count"],
            "propagation_spacing": values["reliability_propagation"]["spacing"],
            "strain_radius": values["strain"]["radius"],
        }
    if config.workflow_kind == "mesh_2d":
        return {
            "selected_element_type": values["mesh"]["element_type"],
            "mesh_element_type": values["mesh"]["element_type"],
            "generation_element_type": values["mesh_generation"]["element_type"],
            "optimization_method": values["optimization"]["method"],
            "objective": values["optimization"]["objective"],
            "max_iterations": values["optimization"]["max_iterations"],
            "convergence_threshold": values["optimization"]["convergence_threshold"],
            "fft_window_size": values["initialization"]["fedic_fft"]["window_size"],
            "fft_search_radius": values["initialization"]["fedic_fft"]["search_radius"],
            "strain_radius": values["strain"]["radius"],
        }
    if config.workflow_kind == "stereo_3d":
        return {
            "quality_metric": values["reconstruction"]["quality_metric"],
            "max_reprojection_error_px": values["reconstruction"]["max_reprojection_error_px"],
            "subset_radius": values["subset_config"]["subset"]["radius"],
            "subset_method": values["subset_config"]["optimization"]["method"],
            "board_type": values["calibration_config"]["board"]["type"],
            "board_rows": values["calibration_config"]["board"]["rows"],
            "board_cols": values["calibration_config"]["board"]["cols"],
            "board_spacing": values["calibration_config"]["board"]["spacing"],
        }
    return {
        "run_subset": values["pairwise_2d_dic"]["run_subset"],
        "run_mesh": values["pairwise_2d_dic"]["run_mesh"],
        "pairwise_2d_solver": values["pairwise_2d_solver"],
        "pairwise_3d_solver": values["pairwise_3d_dic"]["solver"],
        "stitch_solver": values["surface_stitch"]["solver"],
        "subset_radius": values["subset_config"]["subset"]["radius"],
        "calibration_backend": values["self_calibration"]["backend"],
        "scale_board_rows": values["scale"]["board_rows"],
        "scale_board_cols": values["scale"]["board_cols"],
        "scale_square_size": values["scale"]["square_size"],
    }


def _assert_snapshot(config, expected: dict) -> None:
    assert config.workflow_kind == expected["workflow_kind"]
    assert list(config.source_files) == expected["source_files"]
    assert dict(config.capabilities) == expected["capabilities"]
    assert _effective(config) == expected["effective"]
    for path, source in expected.get("provenance", {}).items():
        assert config.provenance[path]["source"] == source


def test_subset_snapshot(repository_root: Path) -> None:
    config = resolve_config("subset_2d", repository_root=repository_root)
    _assert_snapshot(config, _snapshot(repository_root, "subset_ring.json"))
    assert {key: config.provenance[key]["source"] for key in ("subset.radius", "optimization.max_iterations")} == _snapshot(repository_root, "subset_ring.json")["provenance"]


@pytest.mark.parametrize("element", ["t3", "q4", "q8"])
def test_mesh_snapshots(repository_root: Path, element: str) -> None:
    config = resolve_config("mesh_2d", repository_root=repository_root, overrides={"mesh.element_type": element, "mesh_generation.element_type": element})
    _assert_snapshot(config, _snapshot(repository_root, f"mesh_case01_{element}.json"))


def test_stereo_snapshot(repository_root: Path) -> None:
    _assert_snapshot(resolve_config("stereo_3d", repository_root=repository_root), _snapshot(repository_root, "stereo_plate.json"))


def test_multiview_snapshot(repository_root: Path) -> None:
    config = resolve_config("multiview_3d", repository_root=repository_root)
    expected = _snapshot(repository_root, "multiview_cylinder.json")
    _assert_snapshot(config, expected)
    assert [warning.code for warning in config.warnings] == expected["warnings"]
