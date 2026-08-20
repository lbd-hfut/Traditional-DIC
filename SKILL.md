---
name: traditional-dic
description: Use the Traditional-DIC MCP or command-line contract safely for 2D, Stereo 3D, and Multiview 3D digital image correlation workflows.
metadata:
  short-description: MCP-first, CLI-compatible Traditional-DIC workflow guidance
---

# Traditional-DIC

This is an agent-facing usage contract. When Traditional-DIC MCP tools are
connected, prefer them; otherwise use the stable `traditional-dic` CLI and its
normalized run contract. Do not reimplement case discovery,
configuration merging, workflow orchestration, or result interpretation in the
agent.

The generic MCP tool sequence is:

```text
traditional_dic_capabilities
→ traditional_dic_inspect
→ traditional_dic_validate
→ traditional_dic_run
→ traditional_dic_status
→ traditional_dic_summarize
```

The six MCP tools are transport-level siblings of the CLI commands and call
the same F1–F4 Python contracts. MCP `run` requires an explicit external
`output_root`; it is synchronous and may be long-running.

## Supported capabilities

Begin an unfamiliar task with:

```bash
traditional-dic capabilities --format json
```

The current contract is:

- `subset-2d`: standalone 2D Subset-DIC.
- `mesh-2d`: standalone 2D Mesh-DIC.
- `stereo-3d`: Stereo 3D-DIC with Subset correspondence only.
- `multiview-3d`: Multiview 3D-DIC with Subset pairwise correspondence only.

Always trust the capabilities response for the installed version. Do not
assume that a workflow or solver is supported when the command can answer.

Mesh-DIC is a standalone 2D capability. Never request Mesh-DIC for
`stereo-3d` or `multiview-3d`, and never request `solver=both` for
`multiview-3d`. Do not bypass this restriction by calling legacy Python helpers
or native bindings directly.

## Invocation and environment

For an MCP-connected agent, use the six tools above with canonical workflow
identifiers: `subset_2d`, `mesh_2d`, `stereo_3d`, and `multiview_3d`. MCP is
local by default and references case paths accessible to the server. Do not
add client-vendor-specific setup to this workflow contract.

If MCP is unavailable, the preferred shell interface is the installed command:

```bash
traditional-dic <command> ...
```

For a normal installed deployment, use the documented Conda runtime and the
installed `traditional-dic` / `traditional-dic-mcp` entry points without
setting `PYTHONPATH`.  The source-tree fallback below is only for checkout
development.

Check availability with `command -v traditional-dic`. In a repository
checkout where the package has not been installed, use the development
fallback instead of installing dependencies automatically:

```bash
PYTHONPATH=python python -m traditional_dic <command> ...
```

Use the project's documented interpreter/environment when one is provided.
If the CLI or compiled package is unavailable, report the environment problem;
do not automatically install or upgrade CMake, OpenCV, NumPy,
scikit-build-core, or other dependencies.

The module form and installed command have the same semantics. Commands are
non-interactive and must not prompt for confirmation.

## Standard workflow

For a new computation, use this order:

1. `capabilities` — discover support and solver restrictions.
2. `inspect` — resolve the case, frame roles, camera order, ROI, and effective
   configuration without running a solver.
3. `validate` — fail closed on missing or ambiguous inputs/configuration.
4. `run` — execute the selected normalized workflow.
5. `status` — read operational execution state from the run workspace.
6. `summarize` — read key metrics, quality state, warnings, and artifacts.

For an existing workspace, start with `status` and `summarize`; do not rerun
the workflow unless the user explicitly asks for a new run.

## Inspect and validate

Use the public workflow names exactly:

```bash
traditional-dic inspect \
  --workflow subset-2d \
  --case case/mono_DIC/ring \
  --format json

traditional-dic validate \
  --workflow stereo-3d \
  --case case/stereo_DIC/plate_center_load \
  --format json
```

`inspect` exposes the resolved case and configuration. It is the source of
truth for reference/deformed roles, camera ordering, calibration inputs, ROI
semantics, and effective values. Do not teach yourself filesystem rules such
as “first image”, “last image”, or fixed camera names.

`validate` performs input/config validation without scientific execution. If
it exits nonzero, report the structured errors and correct the actual case or
configuration. Do not guess, rename, reorder, or silently substitute inputs.

For Stereo and Multiview, validation should be treated as mandatory before an
expensive run. It is strongly recommended for 2D workflows as well.

## Run

Prefer an explicit isolated output root:

```bash
traditional-dic run \
  --workflow subset-2d \
  --case case/mono_DIC/ring \
  --output /tmp/traditional-dic-run \
  --format json
```

Standalone Mesh-DIC is selected by the workflow name, not a solver flag:

```bash
traditional-dic run \
  --workflow mesh-2d \
  --case <case> \
  --output <workspace> \
  --format json
```

Stereo and Multiview are also selected by workflow name and are implicitly
Subset-only:

```bash
traditional-dic run \
  --workflow stereo-3d \
  --case <case> \
  --output <workspace> \
  --format json

traditional-dic run \
  --workflow multiview-3d \
  --case <case> \
  --output <workspace> \
  --format json
```

Do not manually orchestrate Stereo fields, Multiview camera pairs, masks,
reconstruction, or stitching. The workflow facade owns those stages. Do not use `python examples/subset_2d.py`, `python examples/mesh_2d.py`,
`python examples/stereo_3d.py`, or `python examples/multiview_3d.py` as the
normal agent interface; those remain developer/backward-compatibility
wrappers.

## Configuration overrides

Use F2 overrides only when the key is known and intentional:

```bash
traditional-dic inspect \
  --workflow subset-2d \
  --case case/mono_DIC/ring \
  --set subset.radius=41 \
  --format json
```

Use the exact same `--set` values for `validate` and `run`. The safe pattern is
`inspect → validate → run`. If validation reports
`UNSUPPORTED_SOLVER_FOR_WORKFLOW`, stop; do not try an internal solver path.
Do not maintain a second catalog of configuration keys in the agent or guess
undocumented defaults.

## Run workspace and artifacts

After execution, treat the explicit workspace as authoritative. Read these
F4 files rather than inferring state from CSV names or directory presence:

```text
manifest.json  what ran, inputs/config identity, and registered artifacts
status.json    operational execution state and failures/warnings
metrics.json   compact scientific and quality metrics
result.json    primary/secondary artifact references
```

Use:

```bash
traditional-dic status /tmp/traditional-dic-run --format json
traditional-dic summarize /tmp/traditional-dic-run --format json
```

When locating a displacement, strain, 3D point, or stitched-surface output,
use the result/manifest artifact registry. Do not hardcode legacy paths such
as `result/subset/...` or `result/mesh/...` when the contract provides an
artifact descriptor.

## Status, quality, and exit codes

Execution status and scientific quality are separate. A run may be
`SUCCESS` while some scientific points have `valid=false` or
`valid_fraction < 1`. Use `metrics.json` and `quality_status` to discuss such
quality conditions; do not call a completed process a crash merely because
some points are invalid.

The final execution statuses are:

- `SUCCESS`: completed with expected outputs and no material warnings.
- `SUCCESS_WITH_WARNINGS`: completed with usable outputs and warnings.
- `PARTIAL_SUCCESS`: incomplete workflow; some usable output may remain.
- `VALIDATION_FAILED`: case/config validation failed before scientific work.
- `EXECUTION_FAILED`: the workflow or control contract could not complete.

CLI exit codes are:

```text
0  SUCCESS or SUCCESS_WITH_WARNINGS
2  usage/argument error
3  validation failure
4  PARTIAL_SUCCESS or execution failure
5  run-contract or result-read failure
```

On nonzero exit, inspect structured JSON on stderr and, if a workspace was
created, read `status.json`. For `PARTIAL_SUCCESS`, report completed and failed
stages and which artifacts remain usable. For `SUCCESS_WITH_WARNINGS`, report
usable completion plus warnings. Surface `ZERO_VALID_POINTS` and
`NONFINITE_METRIC` prominently without inventing a scientific explanation.

The CLI is JSON-first for automation. In JSON mode, successful machine output
is one JSON document on stdout; structured errors and workflow progress go to
stderr. Parse JSON semantically and do not regex human text.

## Safety and scientific boundary

- Prefer a new explicit `--output` workspace and keep source cases read-only.
- Never perform destructive cleanup such as deleting `result/`, cleaning a
  case directory, or overwriting source images as part of normal operation.
- Quote paths and pass arguments directly to the CLI. Do not use `eval`, shell
  code generation, or unsafe wildcard assumptions.
- Do not automatically retry a failed run with changed scientific parameters.
  Retry only for a clearly transient environment failure or at the user's
  request.
- Do not infer accuracy, physical failure, material behavior, or experimental
  validity from execution status alone. Scientific interpretation requires the
  user's context and the reported metrics.
- Normally do not call `traditional_dic.subset()`, `traditional_dic.mesh()`,
  low-level Stereo APIs, Multiview APIs, native bindings, or C++ directly.
  Use them only for explicit library-level development/debugging or while
  maintaining the CLI/facade itself.

## Agent decision rule

```text
What can Traditional-DIC do?       → capabilities
What will this case/config resolve?→ inspect
Can it run safely?                 → validate
Compute a new result?               → inspect → validate → run
What happened to an existing run?  → status → summarize
Where are the outputs?              → summarize → manifest/result artifacts
```

The CLI/F1–F5 contracts are the source of truth. If this skill conflicts with
the installed CLI behavior, stop relying on the conflicting instruction and
follow the machine-readable CLI contract.
