# Traditional-DIC MCP server

Traditional-DIC provides an optional, vendor-neutral MCP server as a sibling
adapter to the stable `traditional-dic` CLI. It calls the normalized Python
contracts directly; it does not invoke the CLI as a subprocess.

Install the optional integration extra from the built wheel when the official
MCP Python SDK is required:

```bash
python -m pip install '/path/to/traditional_dic-0.1.0-cp311-cp311-linux_x86_64.whl[mcp]'
```

The server uses stdio by default:

```bash
traditional-dic-mcp
# development checkout fallback:
PYTHONPATH=python python -m traditional_dic.mcp_server
```

See [installation.md](installation.md) for the qualified Conda runtime.  An
installed `traditional-dic-mcp` runs without `PYTHONPATH` and does not require
the repository as its working directory.

When a development MCP client launches that module as a subprocess, configure
the child explicitly with an absolute `PYTHONPATH` pointing to the checkout's
`python/` directory (and, where supported, the checkout as `cwd`).  The
official stdio client intentionally uses an allow-listed environment and does
not inherit an arbitrary parent `PYTHONPATH`.  Installed
`traditional-dic-mcp` remains the preferred invocation.

It exposes exactly six tools:

- `traditional_dic_capabilities`
- `traditional_dic_inspect`
- `traditional_dic_validate`
- `traditional_dic_run`
- `traditional_dic_status`
- `traditional_dic_summarize`

Use capabilities, inspect, and validate before an expensive run. MCP workflow
identifiers are `subset_2d`, `mesh_2d`, `stereo_3d`, and `multiview_3d`.
Stereo and Multiview are Subset-only; Mesh is supported only as standalone
2D. MCP runs require an absolute output workspace outside the source case and
return the F4 `manifest.json`, `status.json`, `metrics.json`, and `result.json`
contract. Existing workspaces should be read with status and summarize rather
than rerun.

Streamable HTTP is available explicitly with `--transport streamable-http` and
binds to `127.0.0.1` by default. Public deployment requires its own
authentication, authorization, network, and path-security controls; M1 does
not make the server internet-facing by default.

Local qualification status: the official Python SDK `mcp==2.0.0` stdio path
and localhost Streamable HTTP path both completed the six-tool protocol smoke
and cheap Subset/Mesh workflow checks in the project environment. Source-tree
stdio launchers must still pass the absolute `PYTHONPATH` explicitly as
described above.
