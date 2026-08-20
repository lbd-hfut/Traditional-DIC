from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.regression


def test_skill_is_cli_first_and_capability_safe(repository_root: Path) -> None:
    text = (repository_root / "SKILL.md").read_text(encoding="utf-8")
    for command in ("capabilities", "inspect", "validate", "run", "status", "summarize"):
        assert f"traditional-dic {command}" in text
    for workflow in ("subset-2d", "mesh-2d", "stereo-3d", "multiview-3d"):
        assert workflow in text
    assert "Subset correspondence only" in text
    assert "Subset pairwise correspondence only" in text
    assert "Never request Mesh-DIC for" in text
    assert "never request `solver=both`" in text
    assert "Do not use `python examples/subset_2d.py`" in text
    assert "do not automatically install or upgrade" in text.lower()
    for tool in (
        "traditional_dic_capabilities",
        "traditional_dic_inspect",
        "traditional_dic_validate",
        "traditional_dic_run",
        "traditional_dic_status",
        "traditional_dic_summarize",
    ):
        assert tool in text
    assert "MCP" in text


def test_skill_describes_normalized_run_contract_and_statuses(repository_root: Path) -> None:
    text = (repository_root / "SKILL.md").read_text(encoding="utf-8")
    for filename in ("manifest.json", "status.json", "metrics.json", "result.json"):
        assert filename in text
    for status in ("SUCCESS", "SUCCESS_WITH_WARNINGS", "PARTIAL_SUCCESS", "VALIDATION_FAILED", "EXECUTION_FAILED"):
        assert status in text
    for code in ("0  SUCCESS", "2  usage/argument error", "3  validation failure", "4  PARTIAL_SUCCESS", "5  run-contract"):
        assert code in text
