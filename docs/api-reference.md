# Python API reference

This reference documents the recommended stable control-plane API. Import from the named modules; do not rely on every symbol re-exported by `traditional_dic.__init__`. Modules such as `subset`, `mesh`, `stereo`, `multiview`, and the pybind extension expose advanced scientific implementation APIs and are not the preferred orchestration layer.

## `traditional_dic.case`

### Data types

| Type | Fields | Purpose |
| --- | --- | --- |
| `CaseSpec(workflow_kind, case_root=None, paths_config=None, case_key=None, case_config={})` | unresolved source choices | serializable input specification |
| `FrameRef(role, path, index=None)` | role, image path, optional deformation index | resolved image role |
| `ROIInput(mode, path=None, frame_name=None)` | ROI mode and optional source | normalized ROI selection |
| `CalibrationInput(index, paths, label=None, camera_id=None)` | ordered calibration paths | calibration pairing |
| `CameraRef(index, camera_id, name, path, reference, deformed, roi=None)` | camera and frame roles | Multiview camera input |
| `ResolvedCase(workflow_kind, case_root, frames=(), roi=..., cameras=(), calibration_inputs=(), calibration_metadata=(), metadata={}, output_roots={})` | fully resolved input contract | F1 output consumed by workflows |
| `ValidationIssue(code, message, path=None)` | structured issue | validation diagnostic |
| `ValidationResult(valid, errors=(), warnings=(), resolved=None)` | validation result | use `to_dict()` for transport output |

`CaseResolutionError(code, message, *, details=None)` is raised for fail-closed case errors. Important codes include `UNSUPPORTED_WORKFLOW`, `MISSING_CASE_ROOT`, `MISSING_DIRECTORY`, `INSUFFICIENT_IMAGES`, `MISSING_ROI`, `MISSING_CAMERAS`, `IMAGE_DIMENSION_MISMATCH`, `CALIBRATION_PAIR_MISMATCH`, and `UNSUPPORTED_SOLVER_FOR_WORKFLOW` where applicable.

### `resolve_case`

```python
resolve_case(workflow_kind: str | CaseSpec, *, case_root=None, paths_config=None,
             case_key=None, repository_root=None, case_config=None) -> ResolvedCase
```

Resolve the recommended F1 contract. Supply either `case_root`, or `paths_config` plus an optional `case_key`. `case_config` is a mapping form of the named case entry. It reads inputs only and raises `CaseResolutionError` on invalid layout.

```python
case = resolve_case("subset_2d", case_root="/data/ring")
```

### Specialized resolvers

```python
resolve_mono_case(case_root, *, workflow_kind="subset_2d", images_dir=".", repository_root=None)
resolve_stereo_case(case_root, *, config=None, repository_root=None)
resolve_multiview_case(case_root, *, config=None, repository_root=None)
```

These return `ResolvedCase` using their respective layout rules. Use them only when the layout is already known; `resolve_case` is the general public entry point.

### Read-only helpers

```python
inspect_case(*args, repository_root=None, **kwargs) -> ValidationResult
validate_case(*args, repository_root=None, **kwargs) -> ValidationResult
```

Both return structured errors/warnings rather than raising resolution errors. `inspect_case` is intended for discovery; `validate_case` has the same normalized result shape for a pre-run gate.

## `traditional_dic.config_resolver`

### Types

`ResolvedConfig(workflow_kind, values, source_files, provenance, capabilities, warnings=(), validation={})` is the F2 contract. `values` is the normalized mapping, `source_files` identifies YAML sources, `provenance` identifies leaf sources, and `capabilities` records the permitted solver. `ConfigIssue(code, message, path=None)` is a structured warning. `ConfigResolutionError(code, message, *, path=None, details=None)` is the fail-closed exception.

### `resolve_config`

```python
resolve_config(workflow_kind, *, config=None, config_path=None, overrides=None,
               subset_config=None, subset_config_path=None,
               calibration_config=None, calibration_config_path=None,
               repository_root=None) -> ResolvedConfig
```

Start from the packaged workflow YAML; a supplied mapping or custom YAML overlays it; `overrides` then applies dotted keys. Stereo and Multiview resolve nested Subset configuration; Stereo also resolves calibration configuration. Invalid keys, invalid values, and non-Subset 3D solver requests raise `ConfigResolutionError`.

```python
cfg = resolve_config("subset_2d", overrides={"subset.radius": 20})
stereo_cfg = resolve_config("stereo_3d", calibration_config="my-calibration.yaml")
```

### Read-only helpers

```python
inspect_config(*args, **kwargs) -> dict[str, object]
validate_config(*args, **kwargs) -> dict[str, object]
```

They return `{ "valid": bool, "config" | "errors": ... }`, preserving structured config errors rather than raising them.

## `traditional_dic.workflows`

All four facade functions require matching `ResolvedCase` and `ResolvedConfig` workflow kinds. They execute science, write an F4 workspace, and return `WorkflowRunResult`.

`WorkflowRunResult(workflow_kind, output_root, artifacts={}, warnings=(), run_id=None, manifest_path=None, status_path=None, metrics_path=None, result_path=None, contract=None)` locates generated results. `WorkflowContext` is an advanced validation/context type; use the facade functions unless extending the workflow layer.

```python
run_subset_2d(resolved_case, resolved_config, *, repository_root=None,
              output_root=None, visualization_root=None, run_id=None)
run_mesh_2d(resolved_case, resolved_config, *, repository_root=None,
            output_root=None, visualization_root=None, element_types=None,
            dense_samples_per_axis=25, run_id=None)
run_stereo_3d(resolved_case, resolved_config, *, repository_root=None,
              output_root=None, visualization_root=None, calibrate=None,
              compute_fields=None, reconstruct=None, run_id=None)
run_multiview_3d(resolved_case, resolved_config, *, repository_root=None,
                 output_root=None, visualization_root=None, resume=False,
                 run_id=None)
```

`output_root` is strongly recommended and becomes the F4 workspace. If omitted, legacy solver roots from the resolved case are used. Mesh `element_types` accepts a sequence such as `["T3"]`; Stereo controls calibration/field/reconstruction stages; Multiview `resume=True` reuses pairwise work. Stereo and Multiview are fixed Subset-only workflows.

## `traditional_dic.run_contract`

The supported read API is:

```python
load_manifest(workspace) -> dict
load_status(workspace) -> dict
load_metrics(workspace) -> dict
load_result(workspace) -> dict
load_run(workspace) -> dict[str, dict]
inspect_run(workspace) -> dict[str, dict]
summarize_run(workspace) -> dict
validate_run_contract(workspace) -> dict[str, dict]
```

All `workspace` parameters accept `str | pathlib.Path`. `load_run` and
`inspect_run` validate all four files first. `summarize_run` gives a compact
transport-neutral view with execution state, quality state, metrics, artifacts,
warnings, and errors. They raise `RunContractError` for missing, inconsistent,
or invalid metadata.

`ArtifactDescriptor` and `RunWorkspace` are advanced contract-construction types. `finalize_run_contract`, `write_execution_failure`, and `derive_metrics` are workflow-maintainer APIs, not normal application APIs.

## `traditional_dic.capabilities`

```python
capability_contract() -> dict[str, object]
f4_capability_contract() -> dict[str, object]
```

`capability_contract()` returns the detached canonical mapping used by CLI and MCP. `f4_capability_contract()` returns its established compact manifest projection. Call the former before selecting a solver programmatically.

## Advanced/native API boundary

The C++ extension is `traditional_dic._traditional_dic`, built from
`bindings/python/module.cpp` and `bindings/python/bind_*.cpp`. It backs Python
algorithm modules such as `subset.py`, `mesh.py`, `stereo.py`, and
`multiview.py`. These modules expose useful implementation functions and
dataclasses, but their stage-level sequencing is not the stable user contract.
Use Case/Config/Workflow/F4 APIs for human applications; use the native layer
only when maintaining scientific algorithms or bindings.

CLI and MCP are documented in [user-guide.md](user-guide.md) and
[agent-guide.md](agent-guide.md).
