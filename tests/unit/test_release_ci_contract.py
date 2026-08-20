from __future__ import annotations

from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit


def _workflow(name: str) -> tuple[str, dict]:
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github" / "workflows" / name).read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def test_release_workflow_is_manual_read_only_and_never_publishes() -> None:
    text, workflow = _workflow("release-qualification.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" in text
    for forbidden in ("twine upload", "gh release create", "pypi", "contents: write", "id-token: write"):
        assert forbidden not in text.lower()
    assert "actions/upload-artifact@v4" in text
    assert "actions/download-artifact@v4" not in text


def test_release_workflow_uses_the_exact_build_artifact_and_supported_runtime() -> None:
    text, workflow = _workflow("release-qualification.yml")
    assert "traditional-dic-release-qualification-linux-cp311" in text
    assert "environment-file: environment.yml" in text
    assert "python-version: \"3.11\"" in text
    assert "verify_release.py" in text
    assert set(workflow["jobs"]) == {"release-qualification"}
    assert "direct-wheel-before.sha256" in text
    assert "direct-wheel-after.sha256" in text


def test_source_workflow_has_read_only_linux_python_311_regression_contract() -> None:
    text, workflow = _workflow("ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request" in text and "branches: [main]" in text
    assert "environment-file: environment.yml" in text
    assert "PYTHONPATH=python python -m pytest tests -q" in text


def test_release_qualifier_has_no_full_stereo_or_multiview_invocation() -> None:
    source = (Path(__file__).resolve().parents[2] / "qualification" / "package" / "verify_release.py").read_text(encoding="utf-8")
    assert "traditional_dic_run', {'workflow':'stereo_3d" not in source
    assert "traditional_dic_run', {'workflow':'multiview_3d" not in source
    assert "multiview_3d" in source  # validation-only restriction gate
    assert "workflow':'mesh_2d" in source
