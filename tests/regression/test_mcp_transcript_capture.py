from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from qualification.mcp.run_client_scenario import run_capture


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")



def test_capture_persists_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    result = run_capture(
        [sys.executable, "-c", "import sys; print('OUT-1', flush=True); print('ERR-1', file=sys.stderr, flush=True); print('OUT-2', flush=True); print('ERR-2', file=sys.stderr, flush=True)"],
        output_dir=tmp_path / "success",
        client="dummy",
        scenario="Q00",
        cwd=Path.cwd(),
    )
    assert result["classification"] == "COMPLETED"
    assert result["exit_code"] == 0
    assert "OUT-1" in _read(tmp_path / "success" / "stdout.log")
    assert "OUT-2" in _read(tmp_path / "success" / "stdout.log")
    assert "ERR-1" in _read(tmp_path / "success" / "stderr.log")
    assert "ERR-2" in _read(tmp_path / "success" / "stderr.log")
    assert (tmp_path / "success" / "metadata.json").is_file()


def test_capture_preserves_nonzero_exit_without_timeout(tmp_path: Path) -> None:
    result = run_capture(
        [sys.executable, "-c", "raise SystemExit(7)"],
        output_dir=tmp_path / "nonzero",
        client="dummy",
        scenario="Q00",
    )
    assert result["classification"] == "EXIT_NONZERO"
    assert result["exit_code"] == 7
    assert result["timed_out"] is False


def test_capture_times_out_and_terminates_process_group(tmp_path: Path) -> None:
    result = run_capture(
        [sys.executable, "-c", "import time; print('BEFORE', flush=True); time.sleep(60)"],
        output_dir=tmp_path / "timeout",
        client="dummy",
        scenario="Q00",
        timeout_seconds=0.15,
        grace_seconds=0.2,
    )
    assert result["classification"] == "TIMEOUT"
    assert result["timed_out"] is True
    assert result["process_alive_at_timeout"] is True
    assert "BEFORE" in _read(tmp_path / "timeout" / "stdout.log")


def test_capture_terminates_child_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        f"p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    result = run_capture(
        [sys.executable, "-c", code],
        output_dir=tmp_path / "child",
        client="dummy",
        scenario="Q00",
        timeout_seconds=0.3,
        grace_seconds=0.2,
    )
    assert result["classification"] == "TIMEOUT"
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("child process survived process-group cleanup")
