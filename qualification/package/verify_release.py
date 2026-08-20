#!/usr/bin/env python3
"""Build and qualify a Traditional-DIC release candidate locally or in CI.

The script intentionally tests a wheel, never an editable checkout.  It keeps
the GitHub workflow small and leaves a compact JSON record that can be uploaded
alongside the exact distribution artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_TOOLS = {
    "traditional_dic_capabilities",
    "traditional_dic_inspect",
    "traditional_dic_validate",
    "traditional_dic_run",
    "traditional_dic_status",
    "traditional_dic_summarize",
}
REQUIRED_WHEEL = {
    "traditional_dic/__init__.py",
    "traditional_dic/_runtime.py",
    "traditional_dic/case.py",
    "traditional_dic/config_resolver.py",
    "traditional_dic/run_contract.py",
    "traditional_dic/cli.py",
    "traditional_dic/mcp_server.py",
}
REQUIRED_SDIST = {"pyproject.toml", "CMakeLists.txt"}


class QualificationFailure(RuntimeError):
    """A release gate failed with a stable category."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, cwd: Path, env: dict[str, str], category: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise QualificationFailure(
            category,
            f"{' '.join(argv)} exited {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
    return completed


def json_output(argv: list[str], *, cwd: Path, env: dict[str, str], category: str) -> dict[str, Any]:
    result = run(argv, cwd=cwd, env=env, category=category)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationFailure(category, f"expected JSON from {' '.join(argv)}: {result.stdout}") from exc


def assert_contract_files(workspace: Path) -> None:
    missing = [name for name in ("manifest.json", "status.json", "metrics.json", "result.json") if not (workspace / name).is_file()]
    if missing:
        raise QualificationFailure("F4_CONTRACT_FAILURE", f"missing F4 files in {workspace}: {missing}")


def wheel_audit(wheel: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_WHEEL - names
        if missing or not any(re.fullmatch(r"traditional_dic/_traditional_dic[^/]*\.so", name) for name in names):
            raise QualificationFailure("WHEEL_CONTENT_FAILURE", f"wheel missing required payload: {sorted(missing)}")
        if not any(name.startswith("traditional_dic/resources/config/") and name.endswith(".yaml") for name in names):
            raise QualificationFailure("WHEEL_CONTENT_FAILURE", "wheel has no packaged configuration YAML")
        unwanted = [name for name in names if name.startswith(("case/", "tests/", "qualification/", "build/", ".git/"))]
        if unwanted:
            raise QualificationFailure("WHEEL_CONTENT_FAILURE", f"wheel contains unwanted paths: {unwanted[:5]}")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        metadata = archive.read(metadata_name).decode("utf-8")
        entries = archive.read(entry_name).decode("utf-8")
    for entry in ("traditional-dic = traditional_dic.cli:main", "traditional-dic-mcp = traditional_dic.mcp_server:main"):
        if entry not in entries:
            raise QualificationFailure("ENTRYPOINT_FAILURE", f"wheel entry point missing: {entry}")
    if "Provides-Extra: mcp" not in metadata or "Requires-Dist: mcp" not in metadata:
        raise QualificationFailure("MCP_INSTALL_FAILURE", "MCP optional extra missing from wheel metadata")
    return {"files": len(names), "metadata": metadata_name, "entry_points": entry_name}


def sdist_audit(sdist: Path) -> dict[str, Any]:
    with tarfile.open(sdist, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
    stripped = {name.split("/", 1)[1] for name in names if "/" in name}
    missing = REQUIRED_SDIST - stripped
    required_prefixes = ("cmake/", "include/", "src/", "bindings/", "python/traditional_dic/")
    if missing or any(not any(name.startswith(prefix) for name in stripped) for prefix in required_prefixes):
        raise QualificationFailure("SDIST_FAILURE", "sdist is not self-contained for a native rebuild")
    unwanted = [name for name in stripped if name.startswith(("case/", "tests/", "qualification/", "build/"))]
    if unwanted:
        raise QualificationFailure("SDIST_FAILURE", f"sdist contains excluded payload: {unwanted[:5]}")
    return {"files": len(names)}


def installed_probe(python: str, executable: str, case_root: Path, work: Path, env: dict[str, str]) -> dict[str, Any]:
    installed_cwd = work / "installed-cwd"
    installed_cwd.mkdir(parents=True, exist_ok=True)
    clean_env = dict(env)
    clean_env.pop("PYTHONPATH", None)
    origin_code = """
import importlib.metadata
import pathlib
import traditional_dic
import traditional_dic._traditional_dic
import traditional_dic.case, traditional_dic.config_resolver, traditional_dic.workflows
import traditional_dic.run_contract, traditional_dic.cli, traditional_dic.capabilities
package = pathlib.Path(traditional_dic.__file__).resolve()
extension = pathlib.Path(traditional_dic._traditional_dic.__file__).resolve()
prefix = pathlib.Path(__import__('sys').prefix).resolve()
assert prefix in package.parents and 'site-packages' in str(package)
assert prefix in extension.parents and 'site-packages' in str(extension)
print(package)
print(extension)
print(importlib.metadata.version('traditional-dic'))
"""
    origins = run([python, "-c", origin_code], cwd=installed_cwd, env=clean_env, category="BASE_INSTALL_FAILURE").stdout.splitlines()
    run([executable, "--version"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    run([executable, "--help"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    capabilities = json_output([executable, "capabilities", "--format", "json"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    if capabilities["capability_contract"]["stereo_3d"]["solver"] != "subset":
        raise QualificationFailure("CAPABILITY_RESTRICTION_FAILURE", "CLI capability contract changed")
    ring = case_root / "mono_DIC" / "ring"
    stereo = case_root / "stereo_DIC" / "plate_center_load"
    inspect = json_output([executable, "inspect", "--workflow", "subset-2d", "--case", str(ring), "--format", "json"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    if inspect["workflow_kind"] != "subset_2d":
        raise QualificationFailure("CLI_SMOKE_FAILURE", "installed CLI inspect returned wrong workflow")
    validate = json_output([executable, "validate", "--workflow", "stereo-3d", "--case", str(stereo), "--format", "json"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    if not validate["valid"]:
        raise QualificationFailure("CLI_SMOKE_FAILURE", "installed CLI Stereo validation failed")
    subset = work / "cli-subset"
    subset_run = json_output([executable, "run", "--workflow", "subset-2d", "--case", str(ring), "--output", str(subset), "--format", "json"], cwd=installed_cwd, env=clean_env, category="SUBSET_SMOKE_FAILURE")
    if subset_run["execution_status"] not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        raise QualificationFailure("SUBSET_SMOKE_FAILURE", str(subset_run))
    assert_contract_files(subset)
    status = json_output([executable, "status", str(subset), "--format", "json"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    summary = json_output([executable, "summarize", str(subset), "--format", "json"], cwd=installed_cwd, env=clean_env, category="CLI_SMOKE_FAILURE")
    return {"package_origin": origins[0], "extension_origin": origins[1], "version": origins[2], "subset": subset_run, "status": status, "summary": summary}


def mcp_probe(python: str, mcp_executable: str, case_root: Path, work: Path, env: dict[str, str]) -> dict[str, Any]:
    """Run official SDK stdio tests in a child so it cannot see checkout imports."""
    probe = work / "mcp_probe.py"
    payload = {
        "server": mcp_executable,
        "ring": str(case_root / "mono_DIC" / "ring"),
        "stereo": str(case_root / "stereo_DIC" / "plate_center_load"),
        "multi": str(case_root / "multi_DIC" / "CylinderDIC"),
        "mesh_workspace": str(work / "mcp-mesh"),
    }
    probe.write_text(
        """import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
data=json.loads(sys.argv[1])
def content(result): return getattr(result, 'structured_content', getattr(result, 'structuredContent', None))
async def main():
  params=StdioServerParameters(command=data['server'], args=[])
  async with stdio_client(params) as streams:
    read, write=streams
    async with ClientSession(read, write) as session:
      await session.initialize()
      tools=await session.list_tools()
      names={tool.name for tool in tools.tools}
      expected={'traditional_dic_capabilities','traditional_dic_inspect','traditional_dic_validate','traditional_dic_run','traditional_dic_status','traditional_dic_summarize'}
      assert names == expected, names
      capabilities=content(await session.call_tool('traditional_dic_capabilities', {}))
      assert capabilities['capability_contract']['stereo_3d']['solver']=='subset'
      inspected=content(await session.call_tool('traditional_dic_inspect', {'workflow':'subset_2d','case':data['ring']}))
      assert inspected['workflow_kind']=='subset_2d'
      valid=content(await session.call_tool('traditional_dic_validate', {'workflow':'stereo_3d','case':data['stereo']}))
      assert valid['valid']
      for workflow, case, solver in [('stereo_3d',data['stereo'],'mesh'),('multiview_3d',data['multi'],'mesh'),('multiview_3d',data['multi'],'both')]:
        rejected=content(await session.call_tool('traditional_dic_validate', {'workflow':workflow,'case':case,'overrides':{'solver':solver}}))
        assert not rejected['valid'] and any(item['code']=='UNSUPPORTED_SOLVER_FOR_WORKFLOW' for item in rejected['errors'])
      mesh=content(await session.call_tool('traditional_dic_run', {'workflow':'mesh_2d','case':data['ring'],'output_root':data['mesh_workspace'],'element_types':['T3'],'dense_samples_per_axis':5}))
      assert mesh['execution_status'] in {'SUCCESS','SUCCESS_WITH_WARNINGS'}, mesh
      status=content(await session.call_tool('traditional_dic_status', {'workspace':data['mesh_workspace']}))
      summary=content(await session.call_tool('traditional_dic_summarize', {'workspace':data['mesh_workspace']}))
      print(json.dumps({'tools':sorted(names),'mesh':mesh,'status':status,'summary':summary}))
asyncio.run(main())
""",
        encoding="utf-8",
    )
    clean_env = dict(env)
    clean_env.pop("PYTHONPATH", None)
    result = run([python, str(probe), json.dumps(payload)], cwd=work, env=clean_env, category="MCP_STDIO_FAILURE")
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationFailure("MCP_STDIO_FAILURE", result.stdout) from exc
    assert_contract_files(work / "mcp-mesh")
    return parsed


def http_probe(python: str, mcp_executable: str, work: Path, env: dict[str, str]) -> dict[str, Any]:
    """Exercise installed Streamable HTTP on a private localhost port."""
    probe = work / "http_probe.py"
    probe.write_text(
        """import asyncio, json, socket, subprocess, sys
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
server=sys.argv[1]
sock=socket.socket(); sock.bind(('127.0.0.1', 0)); port=sock.getsockname()[1]; sock.close()
process=subprocess.Popen([server,'--transport','streamable-http','--host','127.0.0.1','--port',str(port),'--path','/mcp'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
async def main():
  url=f'http://127.0.0.1:{port}/mcp'
  last=None
  for _ in range(40):
    try:
      async with streamable_http_client(url) as streams:
        read, write=streams
        async with ClientSession(read, write) as session:
          await session.initialize()
          tools=await session.list_tools()
          result=await session.call_tool('traditional_dic_capabilities', {})
          payload=getattr(result,'structured_content',getattr(result,'structuredContent',None))
          assert len(tools.tools)==6 and payload['capability_contract']['multiview_3d']['solver']=='subset'
          print(json.dumps({'tool_count':len(tools.tools),'host':'127.0.0.1','port':port}))
          return
    except Exception as exc: last=exc; await asyncio.sleep(.25)
  raise last or RuntimeError('HTTP server did not start')
try:
  asyncio.run(main())
finally:
  process.terminate()
  try: process.wait(timeout=10)
  except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
  assert process.poll() is not None
""",
        encoding="utf-8",
    )
    clean_env = dict(env)
    clean_env.pop("PYTHONPATH", None)
    result = run([python, str(probe), mcp_executable], cwd=work, env=clean_env, category="MCP_HTTP_FAILURE")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationFailure("MCP_HTTP_FAILURE", result.stdout) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, help="existing direct-build artifacts to qualify")
    parser.add_argument("--build", action="store_true", help="build direct artifacts from --source-root before qualification")
    args = parser.parse_args()
    source = args.source_root.resolve()
    work = args.work_root.resolve()
    case_root = args.case_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    started = time.time()
    report: dict[str, Any] = {"schema_version": "1.0", "phase": "R2", "release_candidate_eligible": False, "gates": {}}
    try:
        if args.dist_dir and args.build:
            raise QualificationFailure("BUILD_FAILURE", "use either --dist-dir or --build, not both")
        if args.build:
            dist = work / "dist"
            dist.mkdir(exist_ok=True)
            run([sys.executable, "-m", "build", "--outdir", str(dist)], cwd=source, env=env, category="BUILD_FAILURE")
        elif args.dist_dir:
            dist = args.dist_dir.resolve()
        else:
            raise QualificationFailure("BUILD_FAILURE", "provide --build or --dist-dir")
        wheel = next(dist.glob("*.whl"), None)
        sdist = next(dist.glob("*.tar.gz"), None)
        if wheel is None or sdist is None:
            raise QualificationFailure("BUILD_FAILURE", "expected one wheel and one sdist")
        report["artifacts"] = {"wheel": {"filename": wheel.name, "sha256": sha256(wheel)}, "sdist": {"filename": sdist.name, "sha256": sha256(sdist)}}
        report["gates"]["wheel_audit"] = wheel_audit(wheel)
        report["gates"]["sdist_audit"] = sdist_audit(sdist)
        extracted = work / "sdist-source"
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(extracted, filter="data")
        sdist_source = next(path for path in extracted.iterdir() if path.is_dir())
        sdist_wheels = work / "sdist-wheel"
        sdist_wheels.mkdir()
        run([sys.executable, "-m", "build", "--wheel", "--outdir", str(sdist_wheels)], cwd=sdist_source, env=env, category="WHEEL_FROM_SDIST_FAILURE")
        sdist_wheel = next(sdist_wheels.glob("*.whl"), None)
        if sdist_wheel is None:
            raise QualificationFailure("WHEEL_FROM_SDIST_FAILURE", "no wheel built from sdist")
        report["gates"]["wheel_from_sdist"] = {"filename": sdist_wheel.name, "sha256": sha256(sdist_wheel), "audit": wheel_audit(sdist_wheel)}
        if sdist_wheel.name != wheel.name:
            raise QualificationFailure("WHEEL_FROM_SDIST_FAILURE", "sdist wheel tag/name differs from direct wheel")
        # The supported Conda environment provides the declared base runtime
        # dependencies.  Install only the exact wheel artifact here, then use
        # pip check as the dependency declaration gate; MCP remains absent
        # until the explicit extra is requested below.
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=work, env=env, category="BASE_INSTALL_FAILURE")
        run([sys.executable, "-m", "pip", "check"], cwd=work, env=env, category="BASE_INSTALL_FAILURE")
        executable_path = Path(sys.executable).parent / "traditional-dic"
        if not executable_path.is_file():
            raise QualificationFailure("ENTRYPOINT_FAILURE", "traditional-dic was not installed")
        report["gates"]["base_install"] = installed_probe(sys.executable, str(executable_path), case_root, work, env)
        run([sys.executable, "-m", "pip", "install", f"{wheel}[mcp]"], cwd=work, env=env, category="MCP_INSTALL_FAILURE")
        run([sys.executable, "-m", "pip", "check"], cwd=work, env=env, category="MCP_INSTALL_FAILURE")
        mcp_executable_path = Path(sys.executable).parent / "traditional-dic-mcp"
        if not mcp_executable_path.is_file():
            raise QualificationFailure("ENTRYPOINT_FAILURE", "traditional-dic-mcp was not installed")
        report["gates"]["mcp_stdio"] = mcp_probe(sys.executable, str(mcp_executable_path), case_root, work, env)
        report["gates"]["mcp_http"] = http_probe(sys.executable, str(mcp_executable_path), work, env)
        report["release_candidate_eligible"] = True
        report["failed_gate"] = None
        print("RELEASE_ELIGIBLE=true")
        return 0
    except QualificationFailure as exc:
        report["failed_gate"] = exc.category
        report["error"] = str(exc)
        print(f"RELEASE_ELIGIBLE=false failed_gate={exc.category}", file=sys.stderr)
        return 1
    finally:
        report["duration_seconds"] = round(time.time() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
