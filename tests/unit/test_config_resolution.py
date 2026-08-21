from pathlib import Path

import pytest

from traditional_dic.config_resolver import ConfigResolutionError, resolve_config


pytestmark = pytest.mark.unit


def test_subset_canonical_resolution_and_provenance(repository_root: Path) -> None:
    config = resolve_config("subset_2d", repository_root=repository_root)
    assert config.values["subset"]["radius"] == 41
    assert config.values["optimization"]["max_iterations"] == 50
    assert config.values["initialization"]["integer_search"]["search_radius"] == 30
    assert config.provenance["subset.radius"]["source"] == "config/subset_2d.yaml"


def test_mesh_resolution_preserves_yaml_solver_settings(repository_root: Path) -> None:
    config = resolve_config("mesh_2d", repository_root=repository_root)
    assert config.values["solver"] == "mesh"
    assert config.values["optimization"]["method"] == "fedic_element_fgn"
    assert config.values["mesh_generation"]["element_type"] == "Q4"


def test_mesh_element_override_is_explicit(repository_root: Path) -> None:
    config = resolve_config("mesh_2d", repository_root=repository_root, overrides={"mesh.element_type": "t3", "mesh_generation.element_type": "t3"})
    assert config.values["mesh"]["element_type"] == "T3"
    assert config.values["mesh_generation"]["element_type"] == "T3"


def test_nested_stereo_and_multiview_configs_are_resolved(repository_root: Path) -> None:
    stereo = resolve_config("stereo_3d", repository_root=repository_root)
    multiview = resolve_config("multiview_3d", repository_root=repository_root)
    assert stereo.values["correspondence_solver"] == "subset"
    assert stereo.values["subset_config"]["subset"]["radius"] == 41
    assert stereo.values["calibration_config"]["board"]["rows"] == 8
    assert multiview.values["pairwise_2d_solver"] == "subset"
    assert multiview.values["pairwise_2d_dic"]["run_mesh"] is False
    assert multiview.values["subset_config"]["subset"]["radius"] == 41


def test_override_precedence_and_override_provenance(repository_root: Path) -> None:
    config = resolve_config("subset_2d", repository_root=repository_root, overrides={"subset.radius": 43})
    assert config.values["subset"]["radius"] == 43
    assert config.provenance["subset.radius"]["source"] == "override:subset.radius"


def test_partial_mapping_uses_workflow_yaml_as_default_layer(repository_root: Path) -> None:
    config = resolve_config("subset_2d", config={"subset": {"radius": 43}}, repository_root=repository_root)
    assert config.values["subset"]["radius"] == 43
    assert config.provenance["subset.radius"]["source"] == "mapping"
    assert config.values["optimization"]["max_iterations"] == 50
    assert config.values["initialization"]["integer_search"]["search_radius"] == 30


def test_unrelated_multiview_override_does_not_reject_legacy_yaml_field(repository_root: Path) -> None:
    config = resolve_config("multiview_3d", repository_root=repository_root, overrides={"pairwise_2d_dic.pair_roi_erode_pixels": 2})
    assert config.values["pairwise_2d_dic"]["run_mesh"] is False
    assert [warning.code for warning in config.warnings] == ["LEGACY_MESH_KEY_IGNORED"]


@pytest.mark.parametrize(
    ("workflow", "config", "match"),
    [
        ("subset_2d", {"subset": {"radius": "41"}}, "INVALID_CONFIG_TYPE"),
        ("subset_2d", {"subset": {"radius": 0}}, "INVALID_CONFIG_VALUE"),
        ("subset_2d", {"subset_raduis": 41}, "UNKNOWN_CONFIG_KEY"),
        ("subset_2d", {"initialization": {"integer_search": {"search_raduis": 30}}}, "UNKNOWN_CONFIG_KEY"),
        ("stereo_3d", {"solver": {"method": "mesh"}}, "UNSUPPORTED_SOLVER_FOR_WORKFLOW"),
        ("multiview_3d", {"pairwise_2d_dic": {"solver": "mesh"}}, "UNSUPPORTED_SOLVER_FOR_WORKFLOW"),
        ("multiview_3d", {"pairwise_2d_dic": {"run_mesh": True}}, "UNSUPPORTED_SOLVER_FOR_WORKFLOW"),
        ("multiview_3d", {"pairwise_2d_dic": {"run_subset": False}}, "UNSUPPORTED_SOLVER_FOR_WORKFLOW"),
    ],
)
def test_invalid_config_fails_closed(workflow: str, config: dict, match: str, repository_root: Path) -> None:
    with pytest.raises(ConfigResolutionError, match=match):
        resolve_config(workflow, config=config, repository_root=repository_root)


def test_direct_python_subset_default_remains_distinct(repository_root: Path) -> None:
    import inspect

    from traditional_dic.subset import subset

    assert inspect.signature(subset).parameters["radius"].default == 15
    assert resolve_config("subset_2d", repository_root=repository_root).values["subset"]["radius"] == 41


def test_config_serialization_is_cwd_independent(repository_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = resolve_config("stereo_3d", repository_root=repository_root)
    first = config.to_json(repository_root=repository_root)
    monkeypatch.chdir(tmp_path)
    second = resolve_config("stereo_3d", repository_root=repository_root).to_json(repository_root=repository_root)
    assert first == second
