# Architecture

Traditional-DIC is organized as a set of shared numerical building blocks plus
workflow modules for 2D correspondence, 3D reconstruction, and postprocessing.

## High-Level Data Flow

```text
Image / Mask input
  -> image preprocessing and interpolation
  -> initialization
  -> 2D correspondence
       -> Subset-DIC
       -> Mesh-DIC
  -> calibration and geometry
  -> stereo / multi-view reconstruction
  -> 3D displacement
  -> postprocess
```

Subset-DIC and Mesh-DIC are the two main 2D correspondence workflows. They
share core image, mask, interpolation, correlation, and initialization modules,
but their solver logic and validation routes are intentionally separate.

## Shared Core Modules

Core data containers:

- `dic::Image`: row-major single-channel floating-point image.
- `dic::Mask`: binary ROI/validity mask.
- `dic::Displacement2D`: 2D result record with displacement, affine gradients,
  correlation/quality, solver status, and validity flag.
- `dic::Displacement3D`: 3D displacement result record.

Shared numerical modules:

- `correlation`: SSD, ZNSSD, and ZNCC vector criteria.
- `interpolation`: B-spline image precompute and subpixel interpolation.
- `initialization`: integer search, SIFT matching, subset subpixel refinement,
  and seed/node initialization utilities.
- `config`: YAML parsing for runtime workflow configuration.

Shared modules must stay generic. They should not contain Subset-DIC-specific
propagation logic or Mesh-DIC-specific FEM logic.

## Subset-DIC Boundary

Subset-DIC owns:

- Circular subset sampling.
- ROI-truncated subset behavior.
- Mirror padding policy for subset solve near image/ROI boundaries.
- Seed candidate generation and quality ranking.
- ncorr-style reduced-grid reliability propagation.
- Subset optimizer dispatch over:
  - first-order vs second-order shape functions
  - SSD vs ZNSSD objectives
  - ICGN vs Forward Gauss-Newton optimizers

Current implemented subset optimizer:

```text
first_order + znssd + icgn
```

Other combinations have explicit dispatch scaffolding and are being completed
incrementally. See `docs/subset_dic.md`.

Key files:

- `include/dic/subset/subset_config.hpp`
- `src/subset/subset_dic.cpp`
- `src/subset/seed/seed_selector.cpp`
- `src/subset/seed/reliability_propagation.cpp`
- `src/subset/solver/icgn.cpp`
- `src/subset/solver/forward_gauss_newton.cpp`

## Mesh-DIC Boundary

Mesh-DIC owns:

- ROI/manual mesh generation and mesh topology.
- T3/Q4/Q8 element shape functions.
- Global-to-local coordinate mapping.
- FEM/inform sampling over the image domain.
- SIFT/integer node displacement initialization.
- Global sparse stiffness/residual assembly.
- Mesh objective/optimizer dispatch over:
  - SSD vs ZNSSD objectives
  - global ICGN vs Forward Gauss-Newton optimizers
- Optional displacement-gradient regularization through
  `regularization_alpha`.

Mesh topology generation, nodal initialization, and global solving are separate
steps and should not be mixed.

Current mesh production route uses auto ROI mesh generation, SIFT node
initialization, and SSD-like global solve paths. ZNSSD global mesh paths are
scaffolded for completion. See `docs/mesh_dic.md`.

Key files:

- `include/dic/mesh/mesh_config.hpp`
- `src/mesh/mesh_dic.cpp`
- `src/mesh/generation/roi_mesh_generator.cpp`
- `src/mesh/coordinate/**`
- `src/mesh/element/**`
- `src/mesh/solver/fem_assembler.cpp`

## 3D And Postprocess Boundary

Calibration, geometry, stereo, multi-view, reconstruction, and postprocess
modules consume correspondence-style data but should not own 2D DIC optimizer
logic.

These modules include:

- `src/calibration/**`
- `src/geometry/**`
- `src/reconstruction/**`
- `src/postprocess/**`

Keep changes to these modules separate from Subset-DIC and Mesh-DIC optimizer
work unless a task explicitly targets 3D reconstruction or strain/postprocess.

## Configuration Boundary

Configuration is YAML-driven. Important files:

- `config/subset_2d.yaml`
- `config/mesh_2d.yaml`
- `src/config/yaml_parser.cpp`

Currently the parser is strongest for Subset-DIC config. Mesh diagnostic tools
also accept command-line arguments for ring-case validation. When extending
configuration, keep the defaults conservative and preserve existing behavior.

Important solver fields:

```yaml
shape_function:
  order: first_order | second_order | 1 | 2
optimization:
  method: icgn | forward_gauss_newton | global_icgn
correlation:
  criterion: znssd | ssd
interpolation:
  degree: 1 | 3 | 5
```

Subset uses `CorrelationCriterionKind::ZNSSD` by default. Mesh uses
`CorrelationCriterionKind::SSD` by default to preserve the current global solver
baseline.

## Build Boundary

Use the single configured `build` directory. Do not create extra build folders
for normal development.

Typical full build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build'
```

Typical test run:

```powershell
build\traditional_dic_tests.exe
```

At this checkpoint, the known full-test baseline is:

```text
115 tests total
106 passing
9 known failures in subset/mesh/initialization
```

Any change to that baseline must be explained in the handoff or pull request.

## Repository Hygiene

Do not commit generated outputs:

- `build/`
- `case/2D/ring/auto_mesh_quality_target35/`
- `case/2D/ring/padding_roi_truncation_test/`
- Python `__pycache__/`
- compiled Python extension binaries

Keep Subset-DIC and Mesh-DIC optimizer work on separate branches where possible.
Shared module edits must be justified and verified against both workflows.
