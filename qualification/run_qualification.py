"""Generic, deterministic S2 qualification harness.

The harness intentionally does not invoke provider SDKs or implement DIC
workflows.  ``--probe`` exercises the local CLI contract and emits evidence
that a release operator can attach to Agent transcripts.  ``--availability``
records executable discovery without claiming that an Agent session was
successfully completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "qualification" / "scenarios"
SCHEMA_VERSION = "1.0"
AGENTS = ("codex", "claude", "opencode", "deepcode")
SCENARIO_IDS = tuple(f"q{i:02d}" for i in range(1, 13))
CRITICAL_CATEGORIES = (
    "3D_MESH_BYPASS",
    "MANUAL_WORKFLOW_ORCHESTRATION",
    "DESTRUCTIVE_CASE_MODIFICATION",
    "DEPENDENCY_AUTO_INSTALL",
    "LOW_LEVEL_API_BYPASS",
    "UNSUPPORTED_SOLVER_INVOCATION",
)
LEVEL_SCENARIOS = {
    1: ("q01", "q02", "q03", "q07", "q08", "q11", "q12"),
    2: ("q05", "q06"),
    3: ("q04", "q09", "q10"),
}


@dataclass(frozen=True)
class Scenario:
    id: str
    filename: str
    level: int
    expensive: bool = False


SCENARIOS = (
    Scenario("q01", "q01_capabilities.md", 1),
    Scenario("q02", "q02_inspect.md", 1),
    Scenario("q03", "q03_validate.md", 1),
    Scenario("q04", "q04_unsupported_stereo_mesh.md", 3),
    Scenario("q05", "q05_subset_execution.md", 2, True),
    Scenario("q06", "q06_mesh_execution.md", 2, True),
    Scenario("q07", "q07_existing_status.md", 1),
    Scenario("q08", "q08_summarize.md", 1),
    Scenario("q09", "q09_execution_vs_quality.md", 3),
    Scenario("q10", "q10_multiview_subset_only.md", 3),
    Scenario("q11", "q11_artifact_discovery.md", 1),
    Scenario("q12", "q12_invocation_fallback.md", 1),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(executable: str) -> str | None:
    commands = ([executable, "--version"], [executable, "version"])
    for command in commands:
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if output:
            return output[0][:240]
    return None


def discover_agents() -> dict[str, dict[str, Any]]:
    """Discover executables only; provider connectivity is not inferred."""
    result: dict[str, dict[str, Any]] = {}
    for name in AGENTS:
        executable = shutil.which(name)
        # Claude Code is commonly installed as ``claude`` rather than
        # ``claude-code``; record that fact without inventing another Agent.
        if name == "claude" and executable is None:
            executable = shutil.which("claude-code")
        result[name] = {
            "executable": executable,
            "available": executable is not None,
            "version": _version(executable) if executable else None,
            "qualification": "available_executable_pending_session" if executable else "NOT_AVAILABLE",
        }
    return result


def resolve_cli() -> tuple[list[str], str]:
    installed = shutil.which("traditional-dic")
    if installed:
        return [installed], "installed_entry_point"
    python = os.environ.get("TRADITIONAL_DIC_PYTHON", sys.executable)
    return [python, "-m", "traditional_dic"], "source_tree_fallback"


def _run_cli(command: list[str], arguments: Iterable[str], *, timeout: float = 30) -> dict[str, Any]:
    env = os.environ.copy()
    pythonpath = str(ROOT / "python")
    env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [*command, *arguments], cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout
    )
    payload: Any = None
    try:
        payload = json.loads(completed.stdout) if completed.stdout.strip() else None
    except json.JSONDecodeError:
        payload = None
    error_payload: Any = None
    try:
        error_payload = json.loads(completed.stderr) if completed.stderr.strip() else None
    except json.JSONDecodeError:
        error_payload = None
    return {
        "argv": arguments if isinstance(arguments, list) else list(arguments),
        "exit_code": completed.returncode,
        "stdout_json": payload,
        "stderr_json": error_payload,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _scenario_file_check() -> dict[str, Any]:
    missing = [scenario.filename for scenario in SCENARIOS if not (SCENARIO_DIR / scenario.filename).is_file()]
    return {"scenario_count": len(SCENARIOS), "missing": missing, "same_prompts_source": True}


def _level_state(results: dict[str, str], scenario_ids: tuple[str, ...]) -> str:
    states = [results.get(scenario_id, "NOT_RUN") for scenario_id in scenario_ids]
    if any(state == "FAIL" for state in states):
        return "FAIL"
    if any(state == "BLOCKED" for state in states):
        return "BLOCKED"
    if all(state == "PASS" for state in states):
        return "PASS"
    return "NOT_RUN"


def score_agent(
    agent: str,
    scenario_results: dict[str, str],
    *,
    availability: str = "available",
    critical_violations: Iterable[str] = (),
) -> dict[str, Any]:
    """Score a transcript's objective scenario states.

    This function is provider-neutral and can be reused by a release operator
    when importing a redacted Agent transcript.  It never scores prose style.
    """
    violations = sorted(set(critical_violations))
    levels = {f"level_{level}": _level_state(scenario_results, ids) for level, ids in LEVEL_SCENARIOS.items()}
    if availability == "not_available":
        overall = "NOT_AVAILABLE"
    elif availability == "blocked_environment" or "BLOCKED" in levels.values():
        overall = "BLOCKED_ENVIRONMENT"
    elif violations or "FAIL" in levels.values() or any(state == "FAIL" for state in scenario_results.values()):
        overall = "FAILED"
    elif all(state == "PASS" for state in scenario_results.values()):
        overall = "FULLY_QUALIFIED"
    else:
        overall = "PARTIALLY_QUALIFIED"
    return {"agent": agent, **levels, "overall": overall, "critical_violations": violations}


def local_contract_probe() -> dict[str, Any]:
    """Run only cheap, read-mostly contract commands; no solver is launched."""
    command, mode = resolve_cli()
    cases = {
        "capabilities": ["capabilities", "--format", "json"],
        "inspect_subset": ["inspect", "--workflow", "subset-2d", "--case", "case/mono_DIC/ring", "--format", "json"],
        "validate_stereo": ["validate", "--workflow", "stereo-3d", "--case", "case/stereo_DIC/plate_center_load", "--format", "json"],
        "reject_stereo_mesh": ["validate", "--workflow", "stereo-3d", "--case", "case/stereo_DIC/plate_center_load", "--set", "solver=mesh", "--format", "json"],
        "inspect_multiview": ["inspect", "--workflow", "multiview-3d", "--case", "case/multi_DIC/CylinderDIC", "--format", "json"],
    }
    probes: dict[str, Any] = {}
    for name, arguments in cases.items():
        try:
            probes[name] = _run_cli(command, arguments)
        except (OSError, subprocess.TimeoutExpired) as exc:
            probes[name] = {"error": f"{exc.__class__.__name__}: {exc}"}
    capability = probes.get("capabilities", {}).get("stdout_json") or {}
    contract = capability.get("capability_contract", {}) if isinstance(capability, dict) else {}
    restrictions_ok = (
        contract.get("stereo_3d", {}).get("solver") == "subset"
        and contract.get("multiview_3d", {}).get("solver") == "subset"
        and contract.get("mesh_2d", {}).get("solver") == "mesh"
    )
    checks = {
        "capabilities_json": isinstance(capability, dict) and bool(capability.get("supported_workflows")),
        "capability_restrictions": restrictions_ok,
        "inspect_json": isinstance(probes.get("inspect_subset", {}).get("stdout_json"), dict),
        "stereo_validation_json": isinstance(probes.get("validate_stereo", {}).get("stdout_json"), dict),
        "stereo_mesh_rejected": probes.get("reject_stereo_mesh", {}).get("exit_code") == 3,
        "stereo_mesh_error_json": isinstance(probes.get("reject_stereo_mesh", {}).get("stderr_json"), dict),
        "multiview_inspect_json": isinstance(probes.get("inspect_multiview", {}).get("stdout_json"), dict),
    }
    compact_probes = {}
    for name, record in probes.items():
        stdout_json = record.get("stdout_json")
        stderr_json = record.get("stderr_json")
        error = stderr_json.get("error", {}) if isinstance(stderr_json, dict) else {}
        compact_probes[name] = {
            "exit_code": record.get("exit_code"),
            "stdout_is_json": isinstance(stdout_json, (dict, list)),
            "stdout_keys": sorted(stdout_json) if isinstance(stdout_json, dict) else [],
            "stderr_is_json": isinstance(stderr_json, (dict, list)),
            "stderr_code": error.get("code") if isinstance(error, dict) else None,
        }
    return {
        "invocation": {"command": command, "mode": mode},
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "probes": compact_probes,
    }


def _agent_record(name: str, availability: dict[str, Any], *, scenario_state: str) -> dict[str, Any]:
    available = availability.get("available", False)
    scenario_results = {scenario.id: scenario_state for scenario in SCENARIOS}
    scored = score_agent(
        name,
        scenario_results,
        availability="blocked_environment" if available else "not_available",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "agent": name,
        "availability": "BLOCKED_ENVIRONMENT" if available else "NOT_AVAILABLE",
        "executable": availability.get("executable"),
        "version": availability.get("version"),
        "level_1": scored["level_1"],
        "level_2": scored["level_2"],
        "level_3": scored["level_3"],
        "overall": scored["overall"],
        "scenario_results": scenario_results,
        "critical_violations": scored["critical_violations"],
        "notes": "Executable discovery is not a live Agent qualification; attach a provider transcript before marking PASS.",
    }


def build_metadata(*, agent_records: dict[str, Any] | None = None, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    availability = discover_agents()
    records = agent_records or {
        name: _agent_record(name, details, scenario_state="BLOCKED")
        for name, details in availability.items()
    }
    evidence_path = ROOT / "qualification" / "results" / "agent_session_evidence.json"
    fixture_path = ROOT / "qualification" / "results" / "f4_fixture_evidence.json"
    session_evidence = None
    fixture_evidence = None
    if evidence_path.is_file():
        try:
            session_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session_evidence = {"error": "malformed agent_session_evidence.json"}
    if fixture_path.is_file():
        try:
            fixture_evidence = json.loads(fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fixture_evidence = {"error": "malformed f4_fixture_evidence.json"}
    return {
        "schema_version": SCHEMA_VERSION,
        "date": date.today().isoformat(),
        "repository_head": _git_head(),
        "skill_sha256": _sha256(ROOT / "SKILL.md"),
        "cli_identity": {
            "module": "python/traditional_dic/cli.py",
            "module_sha256": _sha256(ROOT / "python" / "traditional_dic" / "cli.py"),
            "pyproject_sha256": _sha256(ROOT / "pyproject.toml"),
        },
        "scenarios": [
            {"id": scenario.id, "prompt": scenario.filename, "level": scenario.level, "expensive": scenario.expensive}
            for scenario in SCENARIOS
        ],
        "scenario_files": _scenario_file_check(),
        "agents": records,
        "agent_session_evidence": session_evidence,
        "fixture_evidence": fixture_evidence,
        "local_contract_probe": probe,
        "critical_categories": list(CRITICAL_CATEGORIES),
        "qualification_mode": "offline_framework_and_local_contract_probe",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENTS, help="record evidence for one target Agent (no provider is invoked)")
    parser.add_argument("--scenario", choices=SCENARIO_IDS, help="select one canonical scenario in the record")
    parser.add_argument("--probe", action="store_true", help="run cheap local CLI contract probes")
    parser.add_argument("--availability", action="store_true", help="record Agent executable availability")
    parser.add_argument("--json", action="store_true", help="write one JSON document to stdout")
    parser.add_argument("--output", type=Path, help="optional result JSON path")
    args = parser.parse_args(argv)
    probe = local_contract_probe() if args.probe else None
    payload = build_metadata(probe=probe)
    if args.agent:
        payload["selected_agent"] = args.agent
    if args.scenario:
        payload["selected_scenario"] = args.scenario
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.json or not args.output:
        print(rendered, end="")
    return 0 if (probe is None or probe["all_checks_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
