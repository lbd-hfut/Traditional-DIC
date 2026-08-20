import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traditional_dic.case import resolve_mono_case
from traditional_dic.config_resolver import ResolvedConfig, resolve_config
from traditional_dic.run_contract import load_run
from traditional_dic.workflows import (
    WorkflowContext,
    run_mesh_2d,
    run_multiview_3d,
    run_stereo_3d,
    run_subset_2d,
)
import traditional_dic.workflows.mesh_2d as mesh_workflow
import traditional_dic.workflows.subset_2d as subset_workflow


pytestmark = pytest.mark.unit


def _case_fixture(tmp_path: Path, workflow_kind: str = "subset_2d"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, value in (("001.bmp", 20), ("002.bmp", 40), ("003.bmp", 255)):
        Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(tmp_path / name)
    case = resolve_mono_case(tmp_path, workflow_kind=workflow_kind, repository_root=tmp_path)
    return case


def test_facade_exports_and_fixed_3d_solver_contract() -> None:
    assert callable(run_subset_2d)
    assert callable(run_mesh_2d)
    assert callable(run_stereo_3d)
    assert callable(run_multiview_3d)
    assert "solver" not in inspect.signature(run_stereo_3d).parameters
    assert "solver" not in inspect.signature(run_multiview_3d).parameters


def test_context_requires_matching_resolved_contracts(repository_root: Path) -> None:
    case = resolve_mono_case(repository_root / "case/mono_DIC/ring", workflow_kind="subset_2d", repository_root=repository_root)
    config = resolve_config("subset_2d", repository_root=repository_root)
    context = WorkflowContext(repository_root, case, config)
    assert context.resolved_case is case
    with pytest.raises(TypeError):
        WorkflowContext(repository_root, case, {})  # type: ignore[arg-type]


def test_subset_facade_writes_only_to_explicit_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case_fixture(tmp_path / "case")
    config = resolve_config("subset_2d", repository_root=Path.cwd())
    output_root = tmp_path / "run-output"
    calls: list[tuple[Path, Path]] = []

    def fake_subset(reference, deformed, *, config, roi):
        calls.append((Path(str(reference.shape)), Path(str(deformed.shape))))
        return {
            "x": np.array([2.0]), "y": np.array([2.0]),
            "u": np.array([1.0]), "v": np.array([0.5]),
            "du_dx": np.array([0.0]), "du_dy": np.array([0.0]),
            "dv_dx": np.array([0.0]), "dv_dy": np.array([0.0]),
            "correlation": np.array([0.01]), "valid": np.array([True]),
        }

    def fake_plot(reference, result, path, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "white").save(path)

    monkeypatch.setattr(subset_workflow.tdic, "subset", fake_subset)
    monkeypatch.setattr(subset_workflow, "plot_2d_field_overlay", fake_plot)
    monkeypatch.setattr(subset_workflow, "save_least_squares_strain_csv", lambda path, *args, **kwargs: Path(path).write_text("strain\n", encoding="utf-8"))
    result = run_subset_2d(case, config, repository_root=Path.cwd(), output_root=output_root)
    assert result.output_root == output_root.resolve()
    assert result.manifest_path == output_root / "manifest.json"
    contract = load_run(output_root)
    assert contract["manifest"]["case"]["workflow_kind"] == "subset_2d"
    assert contract["manifest"]["config"]["workflow_kind"] == "subset_2d"
    assert calls
    assert (output_root / "002_znssd_icgn_1st" / "displacements.csv").is_file()
    assert not (case.case_root / "result").exists()


def test_mesh_facade_uses_explicit_root_without_solver_execution_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case_fixture(tmp_path / "case", workflow_kind="mesh_2d")
    config = resolve_config("mesh_2d", repository_root=Path.cwd())
    output_root = tmp_path / "mesh-output"
    monkeypatch.setattr(mesh_workflow, "generate_meshes_from_roi", lambda roi, generation: {"T3": {"nodes": np.zeros((3, 2)), "elements": np.zeros((1, 3), dtype=int)}, "summary": {}})
    def fake_run_element(*args, **kwargs):
        # The F4 registry treats the per-element output directory as required.
        Path(args[3], args[5]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mesh_workflow, "run_element", fake_run_element)
    monkeypatch.setattr(mesh_workflow, "save_overview", lambda *args, **kwargs: None)
    result = run_mesh_2d(case, config, repository_root=Path.cwd(), output_root=output_root, element_types=["T3"])
    assert result.output_root == output_root.resolve()
    assert result.status_path == output_root / "status.json"
    assert load_run(output_root)["status"]["execution_status"] == "SUCCESS"
    assert output_root.exists()
