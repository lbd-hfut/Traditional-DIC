# Traditional-DIC local MCP

Traditional-DIC provides an optional local MCP adapter alongside the stable
`traditional-dic` CLI. It calls the same normalized Python Case/Config/
Workflow/F4 contracts directly; it does not shell out to the CLI.

## Install and launch

Create the supported Conda runtime described in [installation.md](installation.md), install the wheel, then add the optional MCP dependency from the same wheel:

```bash
python -m pip install '/path/to/traditional_dic-0.1.0-cp311-cp311-linux_x86_64.whl[mcp]'
traditional-dic-mcp
```

STDIO is the default and recommended local transport. Installed entry points work without `PYTHONPATH` and from any working directory. A development-checkout fallback is:

```bash
PYTHONPATH=python python -m traditional_dic.mcp_server
```

## Public protocol

The server exposes exactly six tools:

- `traditional_dic_capabilities`
- `traditional_dic_inspect`
- `traditional_dic_validate`
- `traditional_dic_run`
- `traditional_dic_status`
- `traditional_dic_summarize`

Use capabilities, inspect, and validate before an expensive run. Canonical workflow IDs are `subset_2d`, `mesh_2d`, `stereo_3d`, and `multiview_3d`. Stereo and Multiview are Subset-only; Mesh is standalone 2D only. `run` requires an explicit output workspace outside the source case and returns the F4 `manifest.json`, `status.json`, `metrics.json`, and `result.json` contract.

See [agent-guide.md](agent-guide.md) for exact tool inputs, call order, error handling, and conservative Agent behavior.

## Localhost HTTP fallback

Streamable HTTP is retained only for local testing/fallback:

```bash
traditional-dic-mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
```

Keep the host at `127.0.0.1`. Traditional-DIC does not provide Remote MCP,
server deployment, accounts, authentication, authorization, or multi-user
service behavior.
