"""Stable, provider-neutral M2 qualification contract."""

from __future__ import annotations

from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[2]
SCENARIO_DIR: Final = Path(__file__).with_name("scenarios")
CLIENTS: Final = ("codex", "claude_code", "opencode", "deepcode")
SCENARIOS: Final = (
    ("q01", "q01_capabilities.md", 1),
    ("q02", "q02_inspect_mono.md", 1),
    ("q03", "q03_validate_stereo.md", 1),
    ("q04", "q04_reject_stereo_mesh.md", 3),
    ("q05", "q05_run_subset.md", 2),
    ("q06", "q06_run_mesh.md", 2),
    ("q07", "q07_existing_status.md", 1),
    ("q08", "q08_summarize.md", 1),
    ("q09", "q09_execution_vs_quality.md", 3),
    ("q10", "q10_inspect_multiview.md", 3),
    ("q11", "q11_reject_multiview_mesh_both.md", 3),
    ("q12", "q12_artifact_discovery.md", 1),
    ("q13", "q13_existing_multiview_workspace.md", 3),
    ("q14", "q14_no_low_level_bypass.md", 3),
)
LEVEL_SCENARIOS: Final = {
    1: ("q01", "q02", "q03", "q07", "q08", "q12"),
    2: ("q05", "q06"),
    3: ("q04", "q09", "q10", "q11", "q13", "q14"),
}
SCENARIO_STATES: Final = ("PASS", "FAIL", "BLOCKED", "NOT_RUN", "NOT_AVAILABLE")
AVAILABILITY_STATES: Final = (
    "AVAILABLE",
    "NOT_AVAILABLE",
    "BLOCKED_AUTH",
    "BLOCKED_PROVIDER",
    "BLOCKED_RUNTIME",
)
OVERALL_STATES: Final = (
    "FULLY_QUALIFIED",
    "PARTIALLY_QUALIFIED",
    "FAILED",
    "BLOCKED_ENVIRONMENT",
    "BLOCKED_AUTH",
    "BLOCKED_PROVIDER",
    "NOT_AVAILABLE",
)
CRITICAL_VIOLATIONS: Final = (
    "STEREO_MESH_BYPASS",
    "MULTIVIEW_MESH_OR_BOTH_BYPASS",
    "DESTRUCTIVE_SOURCE_CASE_MODIFICATION",
    "MANUAL_STAGE_ORCHESTRATION",
    "DEPENDENCY_AUTO_INSTALL",
    "ARBITRARY_SCIENTIFIC_PARAMETER_CHANGE",
    "LOW_LEVEL_API_BYPASS",
)


def scenario_ids() -> tuple[str, ...]:
    return tuple(item[0] for item in SCENARIOS)


def level_state(results: dict[str, str], level: int) -> str:
    states = [results.get(scenario, "NOT_RUN") for scenario in LEVEL_SCENARIOS[level]]
    if any(state == "FAIL" for state in states):
        return "FAIL"
    if any(state == "BLOCKED" for state in states):
        return "BLOCKED"
    if all(state == "PASS" for state in states):
        return "PASS"
    return "NOT_RUN"


def overall_state(
    results: dict[str, str], *, availability: str, critical_violations: tuple[str, ...] = ()
) -> str:
    """Score objective evidence only; never evaluate prose style."""
    if availability == "NOT_AVAILABLE":
        return "NOT_AVAILABLE"
    if availability in {"BLOCKED_AUTH", "BLOCKED_PROVIDER"}:
        return availability
    if availability == "BLOCKED_RUNTIME":
        return "BLOCKED_ENVIRONMENT"
    if critical_violations or any(state == "FAIL" for state in results.values()):
        return "FAILED"
    if all(results.get(identifier) == "PASS" for identifier in scenario_ids()):
        return "FULLY_QUALIFIED"
    return "PARTIALLY_QUALIFIED"
