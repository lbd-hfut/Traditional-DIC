from __future__ import annotations

import json

from qualification.mcp.contract import (
    AVAILABILITY_STATES,
    CLIENTS,
    CRITICAL_VIOLATIONS,
    LEVEL_SCENARIOS,
    OVERALL_STATES,
    SCENARIOS,
    SCENARIO_DIR,
    overall_state,
    scenario_ids,
)


def test_m2_has_fourteen_shared_provider_neutral_scenarios():
    assert scenario_ids() == tuple(f"q{index:02d}" for index in range(1, 15))
    assert all((SCENARIO_DIR / filename).is_file() for _, filename, _ in SCENARIOS)
    prompts = "\n".join((SCENARIO_DIR / filename).read_text(encoding="utf-8") for _, filename, _ in SCENARIOS).lower()
    for provider in ("codex", "claude", "opencode", "deepcode"):
        assert provider not in prompts


def test_m2_levels_and_critical_violations_are_stable():
    assert set().union(*map(set, LEVEL_SCENARIOS.values())) == set(scenario_ids())
    assert "STEREO_MESH_BYPASS" in CRITICAL_VIOLATIONS
    assert "MULTIVIEW_MESH_OR_BOTH_BYPASS" in CRITICAL_VIOLATIONS
    assert "LOW_LEVEL_API_BYPASS" in CRITICAL_VIOLATIONS


def test_m2_result_schema_is_machine_readable_and_scoring_is_objective():
    schema_path = SCENARIO_DIR.parent / "schema" / "qualification_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["client"]["enum"] == list(CLIENTS)
    assert set(schema["properties"]["availability"]["enum"]) == set(AVAILABILITY_STATES)
    assert set(schema["properties"]["overall"]["enum"]) == set(OVERALL_STATES)
    passing = {identifier: "PASS" for identifier in scenario_ids()}
    assert overall_state(passing, availability="AVAILABLE") == "FULLY_QUALIFIED"
    assert overall_state(passing, availability="AVAILABLE", critical_violations=("LOW_LEVEL_API_BYPASS",)) == "FAILED"
