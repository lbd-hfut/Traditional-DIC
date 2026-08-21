from __future__ import annotations

import json

import pytest

from traditional_dic import cli


pytestmark = pytest.mark.regression


def test_cli_workflow_and_capability_contract_is_frozen(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["capabilities", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"
    assert payload["supported_workflows"] == ["subset-2d", "mesh-2d", "stereo-3d", "multiview-3d"]
    assert set(payload["capability_contract"]) == {"subset_2d", "mesh_2d", "stereo_3d", "multiview_3d"}
    assert payload["capability_contract"]["stereo_3d"] == {"supported": True, "solver": "subset"}
    assert payload["capability_contract"]["multiview_3d"] == {"supported": True, "solver": "subset"}


def test_cli_module_entrypoint_is_available() -> None:
    assert callable(cli.main)
    assert cli.WORKFLOW_NAMES == {
        "subset-2d": "subset_2d",
        "mesh-2d": "mesh_2d",
        "stereo-3d": "stereo_3d",
        "multiview-3d": "multiview_3d",
    }
