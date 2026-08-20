from __future__ import annotations

import json
from pathlib import Path

from qualification.run_qualification import (
    AGENTS,
    CRITICAL_CATEGORIES,
    SCENARIOS,
    SCENARIO_DIR,
    build_metadata,
    score_agent,
)


def test_canonical_scenarios_and_shared_prompt_source_exist():
    assert len(SCENARIOS) == 12
    assert [scenario.id for scenario in SCENARIOS] == [f"q{i:02d}" for i in range(1, 13)]
    assert all((SCENARIO_DIR / scenario.filename).is_file() for scenario in SCENARIOS)
    assert all(scenario.filename.startswith(scenario.id + "_") for scenario in SCENARIOS)


def test_qualification_has_no_agent_specific_workflow_files():
    qualification_root = Path(__file__).resolve().parents[2] / "qualification"
    forbidden = {"SKILL.codex.md", "SKILL.claude.md", "SKILL.opencode.md", "SKILL.deepcode.md"}
    assert not any(path.name in forbidden for path in qualification_root.rglob("*"))
    assert not any(path.name in {"codex_runner.py", "claude_runner.py", "opencode_runner.py", "deepcode_runner.py"} for path in qualification_root.rglob("*"))


def test_result_model_is_machine_readable_and_stable():
    payload = build_metadata()
    assert payload["schema_version"] == "1.0"
    assert set(payload["agents"]) == set(AGENTS)
    assert set(payload["critical_categories"]) == set(CRITICAL_CATEGORIES)
    for record in payload["agents"].values():
        assert record["availability"] in {"BLOCKED_ENVIRONMENT", "NOT_AVAILABLE"}
        assert record["overall"] in {"BLOCKED_ENVIRONMENT", "NOT_AVAILABLE"}
        assert set(record["scenario_results"]) == {scenario.id for scenario in SCENARIOS}
        json.dumps(record, sort_keys=True)


def test_scenario_prompts_are_not_provider_specific():
    text = "\n".join((SCENARIO_DIR / scenario.filename).read_text(encoding="utf-8") for scenario in SCENARIOS).lower()
    assert "--codex" not in text
    assert "--claude" not in text
    assert "--opencode" not in text
    assert "--deepcode" not in text


def test_objective_scoring_distinguishes_blocked_failure_and_full_pass():
    passed = {scenario.id: "PASS" for scenario in SCENARIOS}
    assert score_agent("fixture", passed)["overall"] == "FULLY_QUALIFIED"
    blocked = dict(passed, q04="BLOCKED")
    assert score_agent("fixture", blocked)["overall"] == "BLOCKED_ENVIRONMENT"
    failed = dict(passed, q04="FAIL")
    assert score_agent("fixture", failed, critical_violations=["3D_MESH_BYPASS"])["overall"] == "FAILED"
