import inspect
from pathlib import Path

import pytest

from traditional_dic.case import resolve_case
from traditional_dic.config_resolver import resolve_config
from traditional_dic.workflows import run_mesh_2d, run_multiview_3d, run_stereo_3d, run_subset_2d


pytestmark = pytest.mark.regression


def test_canonical_workflow_facade_contracts(repository_root: Path) -> None:
    cases = {
        kind: resolve_case(kind, paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
        for kind in ("subset_2d", "mesh_2d", "stereo_3d", "multiview_3d")
    }
    configs = {kind: resolve_config(kind, repository_root=repository_root) for kind in cases}
    assert cases["subset_2d"].workflow_kind == configs["subset_2d"].workflow_kind == "subset_2d"
    assert cases["mesh_2d"].workflow_kind == configs["mesh_2d"].workflow_kind == "mesh_2d"
    assert cases["subset_2d"].output_roots["subset"]["result_root"] == "result/subset"
    assert cases["mesh_2d"].output_roots["mesh"]["result_root"] == "result/mesh"
    assert cases["stereo_3d"].output_roots["subset"]["visualization_root"] == "visualization/subset"
    assert cases["multiview_3d"].output_roots["subset"]["result_root"] == "result/subset"
    assert configs["stereo_3d"].capabilities["correspondence_solver"] == "subset"
    assert configs["multiview_3d"].capabilities["pairwise_2d_solver"] == "subset"
    assert "solver" not in inspect.signature(run_stereo_3d).parameters
    assert "solver" not in inspect.signature(run_multiview_3d).parameters
    assert all(callable(fn) for fn in (run_subset_2d, run_mesh_2d, run_stereo_3d, run_multiview_3d))
