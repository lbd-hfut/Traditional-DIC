"""Durably run one external client scenario without provider-specific logic.

This helper only owns process execution and evidence capture.  Scenario
meaning, expected tool calls, and scoring remain in the existing M2 contract.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SECRET_WORDS = ("token", "secret", "password", "api_key", "apikey", "authorization", "credential")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: str) -> str:
    lowered = value.lower()
    return "[REDACTED]" if any(word in lowered for word in SECRET_WORDS) else value


def _redacted_argv(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for item in argv:
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
        else:
            result.append(_redact(item))
            if any(word in item.lower() for word in SECRET_WORDS):
                redact_next = True
    return result


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: float) -> str:
    """Terminate only the fresh process group created for this attempt."""
    if process.poll() is not None:
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_seconds)
            return "sigterm"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=grace_seconds)
            return "sigkill"
    except ProcessLookupError:
        return "already_exited"


def run_capture(
    argv: Sequence[str],
    *,
    output_dir: Path,
    client: str,
    scenario: str,
    attempt: int = 1,
    cwd: Path | None = None,
    env_additions: dict[str, str] | None = None,
    timeout_seconds: float = 120.0,
    grace_seconds: float = 2.0,
) -> dict[str, object]:
    """Run ``argv`` and persist incremental stdout/stderr plus metadata."""
    if not argv:
        raise ValueError("argv must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "command.txt"
    prompt_path = output_dir / "prompt.txt"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    metadata_path = output_dir / "metadata.json"
    command_path.write_text(" ".join(_redacted_argv(argv)) + "\n", encoding="utf-8")
    if not prompt_path.exists():
        prompt_path.write_text("", encoding="utf-8")

    child_env = os.environ.copy()
    for key, value in (env_additions or {}).items():
        child_env[key] = value
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    process_alive_at_timeout = False
    termination = "none"
    exit_code: int | None = None
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            env=child_env,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        (output_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process_alive_at_timeout = process.poll() is None
            termination = _terminate_group(process, grace_seconds)
            exit_code = process.returncode
    ended_at = _utc_now()
    classification = "TIMEOUT" if timed_out else ("COMPLETED" if exit_code == 0 else "EXIT_NONZERO")
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "client": client,
        "scenario": scenario,
        "attempt": attempt,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        "command": _redacted_argv(argv),
        "cwd": str(cwd) if cwd is not None else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code": exit_code,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "process_alive_at_timeout": process_alive_at_timeout,
        "termination": termination,
        "classification": classification,
        "environment_keys_added": sorted((env_additions or {}).keys()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        separator = raw_args.index("--")
        option_args = raw_args[:separator]
        command = raw_args[separator + 1 :]
    else:
        option_args = raw_args
        command = []
    args = parser.parse_args(option_args)
    additions: dict[str, str] = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error("--env values must use KEY=VALUE")
        additions[key] = value
    result = run_capture(
        command,
        output_dir=args.output_dir,
        client=args.client,
        scenario=args.scenario,
        cwd=args.cwd,
        env_additions=additions,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
