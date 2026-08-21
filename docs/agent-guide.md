# Traditional-DIC Agent guide

Traditional-DIC is a local scientific tool. Prefer local MCP over shell use:

```text
Agent → local MCP / STDIO → Traditional-DIC F1/F2/F3/F4 → local files and CPU
```

STDIO is the primary transport. Streamable HTTP is an optional localhost-only test/fallback transport; it is not a Remote MCP or public service.

## Connection and fallback order

1. Use connected local MCP tools.
2. If MCP is unavailable, use the installed `traditional-dic` CLI.
3. Only in a development checkout, use `PYTHONPATH=python python -m traditional_dic ...`.

Do not install dependencies automatically during a scientific task. An installed package should run without `PYTHONPATH` or a repository working directory.

## The exact six public tools

No stage-level solver tools are public. The complete registry is:

1. `traditional_dic_capabilities()` — no inputs; returns server/package version, supported workflow IDs, and the canonical capability mapping.
2. `traditional_dic_inspect(workflow, case, paths_config=None, config=None, calibration_config=None, overrides=None)` — read-only F1/F2 resolution; returns resolved case, config, warnings, and capabilities.
3. `traditional_dic_validate(workflow, case, paths_config=None, config=None, calibration_config=None, overrides=None)` — no solver execution; returns structured validity, errors, warnings, and normalized inputs.
4. `traditional_dic_run(workflow, case, output_root, paths_config=None, config=None, calibration_config=None, overrides=None, run_id=None, element_types=None, dense_samples_per_axis=25, skip_calibration=False, compute_fields=False, resume=False)` — validates and synchronously runs one workflow in an explicit external workspace.
5. `traditional_dic_status(workspace)` — reads F4 operational status without recomputing.
6. `traditional_dic_summarize(workspace)` — returns F4 metrics, artifacts, quality, warnings, and errors without recomputing.

`workflow` is one of `subset_2d`, `mesh_2d`, `stereo_3d`, or `multiview_3d`. `case`, configs, and `output_root` are paths visible to the local server. `overrides` is a mapping of dotted F2 keys to values. `output_root` is mandatory for `run` and must be outside the source case.

For Mesh, `element_types` is a list such as `["T3"]`; for Stereo,
`skip_calibration` and `compute_fields` control the documented stages; for
Multiview, `resume` reuses completed pairwise fields.

## Required operating sequence

1. Call **capabilities** before assuming solver availability.
2. Call **inspect** to understand frames, ROI, cameras, calibration, and effective configuration.
3. Call **validate** before an expensive run.
4. Call **run** with a unique external output directory only when validation is valid.
5. Call **status** to read execution state.
6. Call **summarize** to read metrics and artifacts.

For an existing workspace, begin with `status` and `summarize`; do not rerun merely to learn its state.

## Capability and safety rules

| Workflow | Solver rule |
| --- | --- |
| `subset_2d` | Subset supported |
| `mesh_2d` | Mesh supported |
| `stereo_3d` | Subset only; Mesh must be rejected |
| `multiview_3d` | Subset only; Mesh and Both must be rejected |

- Never bypass restrictions through legacy helpers or native bindings.
- Do not mutate source cases; use an explicit workspace outside them.
- Do not manually orchestrate low-level correlation, reconstruction, or stitching stages.
- Read F4 metadata instead of inferring success from a log line or a single metric.
- Execution and scientific quality are separate. `SUCCESS_WITH_WARNINGS` may be an executed run with quality warnings; `valid_fraction < 1` alone is not execution failure.

## Failure handling

Case/config errors are structured. Common codes are `MISSING_CASE_ROOT`,
`MISSING_DIRECTORY`, `INSUFFICIENT_IMAGES`, `MISSING_ROI`, `MISSING_CAMERAS`,
`IMAGE_DIMENSION_MISMATCH`, `CALIBRATION_PAIR_MISMATCH`, and
`UNSUPPORTED_SOLVER_FOR_WORKFLOW`. Quality warnings include
`ZERO_VALID_POINTS` and `NONFINITE_METRIC`.

When a tool reports an error:

1. Preserve the reported code/message/path.
2. Correct the actual case, config, or requested solver rather than guessing.
3. Repeat inspect/validate.
4. Rerun only if the user wants a new computation.

If `run` validation fails, it returns `execution_status: VALIDATION_FAILED` and `quality_status: NOT_EVALUATED`; it does not start a solver. Contract read failures are evidence of an incomplete or inconsistent workspace and should not be repaired by inventing metadata.

## Local MCP launch

After installing the MCP extra:

```bash
traditional-dic-mcp
```

For a local HTTP test only:

```bash
traditional-dic-mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
```

Do not bind public interfaces, deploy a server, add authentication, or describe this as a remote compute service. The companion [mcp.md](mcp.md) covers installation and transport details.
