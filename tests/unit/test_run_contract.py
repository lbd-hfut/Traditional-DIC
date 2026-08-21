import json
from pathlib import Path

import pytest

from traditional_dic.run_contract import (
    FINAL_STATUSES,
    RunContractError,
    RunWorkspace,
    finalize_run_contract,
    load_run,
    validate_run_contract,
    write_execution_failure,
)
from traditional_dic.workflows.common import WorkflowRunResult


pytestmark = pytest.mark.unit


def _success_workspace(tmp_path: Path, *, workflow: str = "subset_2d"):
    workspace = RunWorkspace.create(
        tmp_path / "run",
        workflow,
        repository_root=Path.cwd(),
        run_id="fixed-run",
        started_at="2026-01-01T00:00:00Z",
    )
    artifact = workspace.root / "result" / "displacements.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("x,y,valid\n1,2,1\n", encoding="utf-8")
    return workspace, WorkflowRunResult(workflow, workspace.root, {"displacements": [artifact]})


def test_run_contract_writes_and_loads_four_metadata_files(tmp_path: Path) -> None:
    workspace, result = _success_workspace(tmp_path)
    finalized = finalize_run_contract(
        workspace,
        result,
        finished_at="2026-01-01T00:00:01Z",
    )
    assert set(finalized) == {"manifest", "status", "metrics", "result"}
    assert validate_run_contract(workspace.root)["status"]["execution_status"] == "SUCCESS"
    loaded = load_run(workspace.root)
    assert loaded["manifest"]["run_id"] == loaded["result"]["run_id"] == "fixed-run"
    descriptor = loaded["manifest"]["artifacts"][0]
    assert descriptor["path_scope"] == "workspace"
    assert not descriptor["path"].startswith("/")


def test_status_vocabulary_and_warning_semantics(tmp_path: Path) -> None:
    assert FINAL_STATUSES == {
        "SUCCESS",
        "SUCCESS_WITH_WARNINGS",
        "PARTIAL_SUCCESS",
        "VALIDATION_FAILED",
        "EXECUTION_FAILED",
    }
    workspace, result = _success_workspace(tmp_path)
    finalize_run_contract(
        workspace,
        result,
        warnings=[{"code": "TEST_WARNING", "message": "quality concern"}],
        finished_at="2026-01-01T00:00:01Z",
    )
    assert json.loads(workspace.status_path.read_text())["execution_status"] == "SUCCESS_WITH_WARNINGS"
    assert json.loads(workspace.status_path.read_text())["quality_status"] == "QUALITY_WARNINGS"


def test_partial_and_validation_statuses_are_explicit(tmp_path: Path) -> None:
    for index, status in enumerate(("PARTIAL_SUCCESS", "VALIDATION_FAILED")):
        workspace, result = _success_workspace(tmp_path / str(index))
        finalize_run_contract(workspace, result, execution_status=status, finished_at="2026-01-01T00:00:01Z")
        assert json.loads(workspace.status_path.read_text())["execution_status"] == status


def test_execution_failure_preserves_failure_contract(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(
        tmp_path / "failed",
        "stereo_3d",
        repository_root=Path.cwd(),
        run_id="failed-run",
        started_at="2026-01-01T00:00:00Z",
    )
    write_execution_failure(workspace, RuntimeError("solver failed"), stage="reconstruction")
    status = json.loads(workspace.status_path.read_text())
    assert status["execution_status"] == "EXECUTION_FAILED"
    assert status["failed_stage"] == "reconstruction"
    assert status["errors"][0]["exception_type"] == "RuntimeError"


def test_missing_required_artifact_cannot_finalize_success(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(
        tmp_path / "missing",
        "subset_2d",
        repository_root=Path.cwd(),
        run_id="missing-run",
        started_at="2026-01-01T00:00:00Z",
    )
    result = WorkflowRunResult(
        "subset_2d",
        workspace.root,
        {"displacements": [workspace.root / "result" / "missing.csv"]},
    )
    finalize_run_contract(workspace, result, finished_at="2026-01-01T00:00:01Z")
    status = json.loads(workspace.status_path.read_text())
    assert status["execution_status"] == "EXECUTION_FAILED"
    assert status["failed_stage"] == "contract_finalization"
    assert status["errors"][0]["code"] == "REQUIRED_ARTIFACT_MISSING"


def test_schema_and_run_id_validation_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RunContractError):
        RunWorkspace.create(tmp_path / "bad", "subset_2d", run_id="bad run")
    workspace, result = _success_workspace(tmp_path)
    finalize_run_contract(workspace, result, finished_at="2026-01-01T00:00:01Z")
    status = json.loads(workspace.status_path.read_text())
    status["schema_version"] = "unsupported"
    workspace.status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RunContractError, match="unsupported run contract schema"):
        validate_run_contract(workspace.root)


def test_result_references_are_checked_against_manifest(tmp_path: Path) -> None:
    workspace, result = _success_workspace(tmp_path)
    finalize_run_contract(workspace, result, finished_at="2026-01-01T00:00:01Z")
    manifest = json.loads(workspace.manifest_path.read_text())
    result_payload = json.loads(workspace.result_path.read_text())
    result_payload["primary_artifacts"].append("missing-artifact")
    workspace.result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    with pytest.raises(RunContractError, match="unknown artifacts"):
        validate_run_contract(workspace.root)
