from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import traditional_dic.cli as cli
from traditional_dic.run_contract import RunWorkspace, finalize_run_contract
from traditional_dic.workflows.common import WorkflowRunResult


pytestmark = pytest.mark.unit


def test_capabilities_json_is_single_machine_document(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["capabilities", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["supported_workflows"] == ["subset-2d", "mesh-2d", "stereo-3d", "multiview-3d"]
    assert payload["capability_contract"]["stereo_3d"]["solver"] == "subset"
    assert payload["capability_contract"]["multiview_3d"]["solver"] == "subset"


def test_inspect_and_validate_do_not_execute_solver(repository_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect", "--workflow", "subset-2d", "--case", "case/mono_DIC/ring", "--format", "json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["workflow_kind"] == "subset_2d"
    assert inspected["case"]["case_root"] == "case/mono_DIC/ring"
    assert cli.main(["validate", "--workflow", "stereo-3d", "--case", "case/stereo_DIC/plate_center_load", "--format", "json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True


def test_invalid_stereo_solver_is_validation_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "validate",
            "--workflow",
            "stereo-3d",
            "--case",
            "case/stereo_DIC/plate_center_load",
            "--set",
            "solver=mesh",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == cli.EXIT_VALIDATION
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert any(error["code"] == "UNSUPPORTED_SOLVER_FOR_WORKFLOW" for error in payload["errors"])
    assert json.loads(captured.err)["error"]["code"] == "VALIDATION_FAILED"


def test_invalid_case_emits_structured_json_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["inspect", "--workflow", "subset-2d", "--case", "does-not-exist", "--format", "json"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_VALIDATION
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] in {"MISSING_CASE_CONFIG", "MISSING_CASE_KEY", "MISSING_DIRECTORY", "MISSING_CASE_ROOT"}


def test_unknown_workflow_is_structured_json_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["inspect", "--workflow", "stereo-mesh", "--case", "case/stereo_DIC/plate_center_load", "--format", "json"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_VALIDATION
    assert json.loads(captured.err)["error"]["code"] == "INVALID_ARGUMENT"


def test_run_dispatches_without_subprocess(monkeypatch: pytest.MonkeyPatch, repository_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    calls: list[tuple[object, object, dict[str, object]]] = []

    def fake_runner(case: object, config: object, **kwargs: object) -> WorkflowRunResult:
        calls.append((case, config, kwargs))
        return WorkflowRunResult(
            "subset_2d",
            tmp_path,
            run_id="cli-test",
            manifest_path=tmp_path / "manifest.json",
            status_path=tmp_path / "status.json",
            metrics_path=tmp_path / "metrics.json",
            result_path=tmp_path / "result.json",
            contract={"status": {"execution_status": "SUCCESS", "quality_status": "QUALITY_OK"}},
        )

    monkeypatch.setitem(cli.WORKFLOW_DISPATCH, "subset_2d", fake_runner)
    # Avoid an expensive solve while checking the explicit dispatch contract.
    fake_case = SimpleNamespace()
    fake_config = SimpleNamespace()
    monkeypatch.setattr(cli, "_resolve_inputs", lambda args: ("subset_2d", fake_case, fake_config))
    code = cli.main(
        [
            "run",
            "--workflow",
            "subset-2d",
            "--case",
            "case/mono_DIC/ring",
            "--output",
            str(tmp_path),
            "--run-id",
            "cli-test",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "cli-test"
    assert calls and calls[0][2]["output_root"] == tmp_path


def test_status_and_summarize_read_f4_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = RunWorkspace.create(tmp_path, "subset_2d", run_id="status-test")
    artifact = tmp_path / "displacements.csv"
    artifact.write_text("x,y,u,v\n0,0,0,0\n", encoding="utf-8")
    finalize_run_contract(workspace, WorkflowRunResult("subset_2d", tmp_path, {"displacements": str(artifact)}))
    assert cli.main(["status", str(tmp_path), "--format", "json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["run_id"] == "status-test"
    assert cli.main(["summarize", str(tmp_path), "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["execution_status"] == "SUCCESS"
    assert summary["artifacts"]


def test_run_status_exit_mapping() -> None:
    assert cli._status_exit("SUCCESS") == 0
    assert cli._status_exit("SUCCESS_WITH_WARNINGS") == 0
    assert cli._status_exit("PARTIAL_SUCCESS") == cli.EXIT_EXECUTION
    assert cli._status_exit("VALIDATION_FAILED") == cli.EXIT_VALIDATION
    assert cli._status_exit("EXECUTION_FAILED") == cli.EXIT_EXECUTION
