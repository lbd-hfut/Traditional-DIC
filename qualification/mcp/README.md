# Traditional-DIC MCP client qualification

This is release/development evidence for M2, not production runtime code.  It
qualifies real MCP clients against the same repository-root `SKILL.md` and the
same six Traditional-DIC MCP tools.  It neither invokes providers from pytest
nor adds an Agent SDK dependency.

Each client receives the matching canonical prompt in `scenarios/`.  Record a
compact, redacted transcript in an external release log or `results/`, with the
tool calls, arguments, outcomes, final decision, transport, server identity,
and client version.  Do not record credentials, environment dumps, or large
provider logs.

The preferred local transport is stdio.  For source-tree qualification the
server configuration must set the absolute package path explicitly:

```text
command: /home/a306/miniconda3/envs/tradic/bin/python
args: [-m, traditional_dic.mcp_server]
env: {PYTHONPATH: /home/a306/01project/Traditional-DIC/python}
cwd: /home/a306/01project/Traditional-DIC
```

For live scenarios, use `run_client_scenario.py` with one fresh client session
per scenario. Raw incremental logs belong under a temporary transcript root;
compact redacted result JSON belongs under `results/<client>/`. The helper is
client-neutral and records timestamps, exit status, timeout state, process-group
cleanup, and the exact (redacted) argv without collecting provider secrets.

The paths above are qualification evidence only, not portable client setup
instructions.  See `docs/mcp.md` for generic installation and transport use.

Level 1 covers discovery and read-only use; Level 2 covers one isolated Subset
and one isolated Mesh run; Level 3 covers capability restrictions, existing
workspace reuse, and non-bypass behavior.  A client is fully qualified only
when all fourteen scenarios pass without a critical violation.
