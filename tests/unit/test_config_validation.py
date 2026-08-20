from pathlib import Path

import pytest

from traditional_dic.config_resolver import inspect_config, resolve_config


pytestmark = pytest.mark.unit


def test_calibration_config_is_typed_and_validated(repository_root: Path) -> None:
    config = resolve_config("calibration", repository_root=repository_root)
    assert config.values["board"]["rows"] == 8
    assert config.values["board"]["cols"] == 11
    assert config.values["board"]["spacing"] == 5.0


def test_inspect_config_returns_structured_failure(repository_root: Path) -> None:
    result = inspect_config("subset_2d", config={"subset": {"radius": -1}}, repository_root=repository_root)
    assert result["valid"] is False
    assert result["errors"][0]["code"] == "INVALID_CONFIG_VALUE"
    assert result["errors"][0]["path"] == "subset.radius"


def test_stereo_and_multiview_capabilities_are_subset_only(repository_root: Path) -> None:
    stereo = resolve_config("stereo_3d", repository_root=repository_root)
    multiview = resolve_config("multiview_3d", repository_root=repository_root)
    assert stereo.capabilities == {"workflow": "stereo_3d", "correspondence_solver": "subset"}
    assert multiview.capabilities["pairwise_2d_solver"] == "subset"
    assert multiview.capabilities["pairwise_3d_solver"] == "subset"
