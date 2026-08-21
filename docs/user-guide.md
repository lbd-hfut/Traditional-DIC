# Traditional-DIC user guide

## Choose a workflow

Traditional-DIC uses one normalized route for command-line and Python use:

```text
case resolution (F1) → configuration resolution (F2) → workflow facade (F3)
→ manifest/status/metrics/result (F4)
```

| Workflow | Use | Solver |
| --- | --- | --- |
| `subset-2d` | standalone point/subset correlation | Subset |
| `mesh-2d` | standalone finite-element-style 2D field | Mesh: `T3`, `Q4`, `Q8` |
| `stereo-3d` | calibrated two-camera reconstruction | Subset only |
| `multiview-3d` | multi-camera reconstruction and stitching | Subset only |

Stereo and Multiview do not accept Mesh. Confirm the installed capability contract with `traditional-dic capabilities --format json`.

## Install and verify

Follow [installation.md](installation.md), then run:

```bash
traditional-dic --help
traditional-dic capabilities --format json
```

For MCP support install the wheel's `[mcp]` extra. Installed operation should not set `PYTHONPATH`.

## Cases and ROI

A case is an input directory, not a workspace. Always put output in a separate directory with `--output` or `output_root`.

```text
case/
├── mono_DIC/<case>/              # images; mono convention: first/reference,
│                                  # middle/deformed, last/ROI
├── stereo_DIC/<case>/
│   ├── cam1/  cam2/              # reference/deformed image series
│   ├── calibrate1/  calibrate2/  # paired calibration images
│   └── ROI.bmp                   # explicit ROI by default
└── multi_DIC/<case>/
    ├── images/<camera-id>/       # at least two camera directories
    ├── calibrate_images/
    └── calibration/
```

`config/case_paths.yaml` provides named repository examples. For external work pass an absolute `--case`. Multiview supports ROI modes `auto`, `last_image`, `explicit_image`, and `none`; Stereo uses an explicit ROI image. An ROI/mask identifies analysis support; it is not an intensity image.

## CLI

Public commands are `capabilities`, `inspect`, `validate`, `run`, `status`, and `summarize`. JSON stdout is script-safe; scientific progress is sent to stderr.

| Exit code | Meaning |
| --- | --- |
| 0 | success or success with warnings |
| 2 | command-line usage error |
| 3 | case/config validation or invalid argument |
| 4 | partial or execution failure |
| 5 | F4 run-contract error |

### Inspect and validate

```bash
traditional-dic inspect --workflow mesh-2d --case /path/to/case --format json
traditional-dic validate --workflow mesh-2d --case /path/to/case --format json
```

`inspect` resolves inputs and effective configuration without a solver. `validate` reports whether that normalized contract can run.

### Run 2D workflows

```bash
traditional-dic run --workflow subset-2d --case /path/to/case \
  --output /tmp/dic-subset --format json

traditional-dic run --workflow mesh-2d --case /path/to/case \
  --element T3 --dense-samples-per-axis 25 \
  --output /tmp/dic-mesh --format json
```

### Run 3D workflows

```bash
traditional-dic run --workflow stereo-3d --case /path/to/stereo-case \
  --compute-fields --output /tmp/dic-stereo --format json

traditional-dic run --workflow multiview-3d --case /path/to/multiview-case \
  --resume --output /tmp/dic-multiview --format json
```

`--skip-calibration` lets Stereo reuse calibration products; `--compute-fields` requests its three Subset fields; `--resume` permits Multiview reuse of completed pairwise fields. None enables Mesh for a 3D workflow.

### Read an existing workspace

```bash
traditional-dic status /tmp/dic-subset --format json
traditional-dic summarize --run /tmp/dic-subset --format json
```

## Configuration

The packaged baselines are `subset_2d.yaml`, `mesh_2d.yaml`, `stereo_3d.yaml`, `multiview_3d.yaml`, and `calibration.yaml`. Pass `--config` for a workflow YAML and `--calibration-config` for Stereo calibration. Use repeatable dotted overrides only for deliberate changes:

```bash
traditional-dic validate --workflow subset-2d --case /path/to/case \
  --set subset.radius=20 --set optimization.max_iterations=30 --format json
```

Resolution starts from the workflow baseline, overlays a supplied mapping or custom YAML, then applies overrides. Stereo and Multiview also resolve nested Subset configuration. Invalid keys and invalid solver choices fail closed; use `inspect` to see effective values and provenance.

| YAML | Main sections | Notes |
| --- | --- | --- |
| `subset_2d.yaml` | `subset`, `shape_function`, `optimization`, `correlation`, `interpolation`, `initialization`, `seed_selection`, `reliability_propagation`, `strain` | standalone Subset baseline |
| `mesh_2d.yaml` | `mesh`, `mesh_generation`, `optimization`, `interpolation`, `initialization`, `strain` | standalone Mesh baseline; element types are T3/Q4/Q8 |
| `stereo_3d.yaml` | `workflow`, `solver`, `configs`, `reconstruction`, `strain` | resolves nested Subset and calibration YAML; solver is Subset-only |
| `multiview_3d.yaml` | `self_calibration`, `camera_pair_selection`, `maskGen`, `pairwise_2d_dic`, `pairwise_3d_dic`, `scale`, `surface_stitch`, `strain`, `triangulation` | pairwise correspondence/reconstruction is Subset-only |
| `calibration.yaml` | `calibration`, `board`, `detection`, `mono_calibration`, `stereo_calibration`, `self_calibration`, `scale` | calibration baseline used by Stereo and supporting workflows |

## Python facade

The full signatures are in [api-reference.md](api-reference.md). The recommended pattern is:

```python
from traditional_dic.case import resolve_case
from traditional_dic.config_resolver import resolve_config
from traditional_dic.workflows import run_mesh_2d
from traditional_dic.run_contract import summarize_run

case = resolve_case("mesh_2d", case_root="/path/to/case")
config = resolve_config("mesh_2d")
run = run_mesh_2d(case, config, output_root="/tmp/dic-mesh", element_types=["T3"])
print(summarize_run(run.output_root)["execution_status"])
```

Replace the workflow name and facade function for Subset, Stereo, or Multiview. Do not manually orchestrate calibration, pairwise fields, reconstruction, or stitching unless you intentionally maintain low-level internals.

## F4 outputs and result interpretation

```text
workspace/
├── manifest.json  # identity, resolved case/config, artifact descriptors
├── status.json    # execution state, stages, warnings, errors
├── metrics.json   # numerical summary and quality warnings
├── result.json    # primary/secondary artifact references
└── workflow-specific artifacts
```

Execution states include `SUCCESS`, `SUCCESS_WITH_WARNINGS`, `PARTIAL_SUCCESS`, `VALIDATION_FAILED`, and `EXECUTION_FAILED`. Quality is separate: `QUALITY_OK` or `QUALITY_WARNINGS`. `valid_fraction < 1` is not by itself execution failure; zero valid points and non-finite metrics are quality warnings.

## Troubleshooting

| Symptom/code | Action |
| --- | --- |
| Native extension loader failure | Activate the documented Conda runtime; do not use arbitrary system Python. |
| `MISSING_CASE_ROOT` / `MISSING_DIRECTORY` | Correct the case path or `case_paths.yaml` key. |
| `INSUFFICIENT_IMAGES`, `MISSING_ROI`, `MISSING_CAMERAS` | Repair case layout, then inspect and validate again. |
| `IMAGE_DIMENSION_MISMATCH` / `CALIBRATION_PAIR_MISMATCH` | Use compatible images and paired calibration series. |
| `UNSUPPORTED_SOLVER_FOR_WORKFLOW` | Keep Stereo/Multiview on Subset; use Mesh only through `mesh-2d`. |
| `ZERO_VALID_POINTS` / `NONFINITE_METRIC` | Inspect metrics and inputs; do not treat this as a successful scientific result. |
| MCP command unavailable | Install the same wheel with its `[mcp]` extra. |

For package-level failures, see local qualification in the [development guide](development.md).
