# Subset-DIC Optimizer Completion Task

This document is the handoff brief for the teammate or AI coding agent assigned
to complete the Subset-DIC optimizer family. Read this file before editing code.

## Scope

Only work on Subset-DIC optimizer completion. Do not refactor Mesh-DIC,
reconstruction, postprocess, calibration, Python packaging, or unrelated tools.

Allowed areas:

- `include/dic/subset/**`
- `src/subset/**`
- `include/dic/initialization/**`, only when the change is required by subset seed/subpixel dispatch
- `src/initialization/**`, only for subset initializer dispatch
- `src/config/yaml_parser.cpp`, only for subset YAML fields
- `config/subset_2d.yaml`
- `tests/subset/**`
- `tests/initialization/**`, only for subset initializer behavior
- `tools/subset_dic_diagnostic.cpp`, only for subset diagnostics

Avoid changing shared modules unless necessary:

- `include/dic/interpolation/**` and `src/interpolation/**`
- `include/dic/core/**` and `src/core/**`
- `include/dic/correlation/**` and `src/correlation/**`

If a shared module must change, explain why in the commit message and run the
mesh-related tests listed below to make sure Mesh-DIC was not broken.

## Current State

Subset-DIC is no longer just a skeleton. The current pipeline already includes:

- ROI-truncated circular subsets via `truncate_roi_subsets`.
- Mirror image padding before subset solve near ROI/image boundaries.
- ncorr-style reduced-grid propagation spacing: config `spacing` is a gap count,
  and the actual full-resolution stride is `spacing + 1`.
- Reliable propagation over all reduced-grid points, with no
  `reliability_propagation.max_points` cap.
- Seed selection from ROI/k-means candidates.
- Integer-search initialization with optional pyramid search.
- Seed subpixel refinement through configured subset solver.
- Shared B-spline interpolation and image precompute.
- Reference gradient cache support for the existing Subset ICGN path.
- Explicit first-order affine shape-function parameters:
  `u, v, du_dx, du_dy, dv_dx, dv_dy`.
- A working first-order ZNSSD ICGN implementation in
  `src/subset/solver/icgn.cpp`.
- Dispatch scaffolding for objective/shape/optimizer combinations:
  `CorrelationCriterionKind::{SSD, ZNSSD}`,
  `SubsetShapeFunctionMethod::{FirstOrder, SecondOrder}`,
  `SubsetOptimizationMethod::{ICGN, ForwardGaussNewton}`.
- YAML parsing for `correlation.criterion` into `SubsetConfig::objective` and
  seed subpixel refinement objective.

Current important files:

- Public config: `include/dic/subset/subset_config.hpp`
- Shared seed config enums: `include/dic/initialization/seed_config.hpp`
- Main controller: `src/subset/subset_dic.cpp`
- Seed selection: `src/subset/seed/seed_selector.cpp`
- Reliability propagation: `src/subset/seed/reliability_propagation.cpp`
- ICGN solver: `src/subset/solver/icgn.cpp`
- Forward GN solver: `src/subset/solver/forward_gauss_newton.cpp`
- First-order shape function: `src/subset/shape/first_order.cpp`
- Second-order placeholder: `src/subset/shape/second_order.cpp`
- YAML parser: `src/config/yaml_parser.cpp`
- Diagnostic executable source: `tools/subset_dic_diagnostic.cpp`

The existing implemented optimizer is:

```text
shape function: first_order
objective:      znssd
optimizer:      icgn
status:         implemented
```

The following combinations currently have explicit placeholder dispatch and
must be completed:

```text
first_order  + ssd   + icgn
second_order + znssd + icgn
second_order + ssd   + icgn
first_order  + znssd + forward_gauss_newton
first_order  + ssd   + forward_gauss_newton
second_order + znssd + forward_gauss_newton
second_order + ssd   + forward_gauss_newton
```

## Required Implementation

Complete the optimizer family without changing the high-level Subset-DIC
pipeline.

1. Preserve the existing first-order ZNSSD ICGN behavior.
   Do not rewrite it wholesale unless a focused bug fix is necessary.

2. Implement SSD ICGN for first-order affine shape functions.
   Use the same sampling, B-spline interpolation, convergence handling, and
   result status conventions as the current ZNSSD ICGN path.

3. Implement second-order shape-function support for ICGN.
   Add the additional second-order warp parameters in a way that does not break
   `Displacement2D` consumers. If the result struct cannot yet store all
   parameters cleanly, keep public displacement output compatible and document
   where second-order parameters are held internally.

4. Implement Forward Gauss-Newton for first-order SSD and ZNSSD.
   This path should re-evaluate the deformed-image gradient/Jacobian as required
   by a forward-additive formulation.

5. Implement Forward Gauss-Newton for second-order SSD and ZNSSD.

6. Keep ROI-truncated subset behavior consistent for all variants.
   The masked and unmasked solve paths must produce equivalent behavior away
   from ROI boundaries.

7. Keep seed refinement and propagation dispatch connected to config:

```yaml
shape_function:
  order: first_order | second_order | 1 | 2
optimization:
  method: icgn | forward_gauss_newton
correlation:
  criterion: znssd | ssd
initialization:
  subpixel_refinement:
    shape_function: first_order | second_order | 1 | 2
    optimizer: icgn | forward_gauss_newton
    criterion: znssd | ssd
```

8. Do not introduce multithreading in this task.
   The current subset path is intentionally single-threaded while correctness
   is being aligned with ncorr.

## Ncorr Alignment Notes

Use `C:\02Project\Study\ncorr_2D_matlab` as the reference implementation.
Read these files before changing optimizer math:

- `ncorr_alg_rgdic.cpp`
- `ncorr_alg_dicanalysis.m`

Important interpretation:

- ncorr `spacing` is not image downsampling.
- It defines the reduced-grid propagation sampling interval.
- In this codebase, config spacing is treated as ncorr's gap count, so the
  full-resolution stride is `spacing + 1`.
- Do not downsample reference/deformed images as a substitute for propagation
  spacing.

Pay special attention to:

- Reduced grid and seed placement.
- Thread diagram/region/nodelist concepts.
- ROI and subset truncation.
- Reference gradient and B-spline precomputed lookup usage.
- `precompute QK_B_QKT_buffer` style reuse.
- ZNSSD residual and Hessian construction.

## Tests To Add Or Update

Add focused tests for every completed combination.

Required unit coverage:

- ICGN first-order SSD recovers a known translation.
- ICGN second-order ZNSSD returns valid results on a synthetic second-order warp.
- ICGN second-order SSD returns valid results on a synthetic second-order warp.
- Forward GN first-order ZNSSD recovers a known translation.
- Forward GN first-order SSD recovers a known translation.
- Forward GN second-order ZNSSD/SSD run without placeholder status.
- YAML dispatch selects the intended solver/objective/shape-function path.
- Masked ROI-truncated solve remains stable near padded boundaries.

Existing relevant tests:

- `tests/subset/test_icgn.cpp`
- `tests/subset/test_forward_gauss_newton.cpp`
- `tests/subset/test_subset_dic.cpp`
- `tests/subset/test_region.cpp`
- `tests/subset/test_yaml_config.cpp`
- `tests/initialization/test_integer_search.cpp`

## Ring Case Validation

Use the ring case after unit tests pass.

Input images:

```text
case/2D/ring/001.bmp   reference speckle
case/2D/ring/002.bmp   deformed speckle
case/2D/ring/003.bmp   ROI mask
```

Subset output directory:

```text
case/2D/ring/padding_roi_truncation_test/subset
```

Run:

```powershell
build\subset_dic_diagnostic.exe `
  case\2D\ring\001.bmp `
  case\2D\ring\002.bmp `
  case\2D\ring\003.bmp `
  case\2D\ring\padding_roi_truncation_test\subset\displacements.csv `
  case\2D\ring\padding_roi_truncation_test\subset\config.yaml

C:\Users\lbd\miniconda3\python.exe `
  case\2D\ring\padding_roi_truncation_test\subset\visualize.py
```

Expected current baseline with radius `37` and spacing `3`:

```text
total_points=102400
valid_points=48868
invalid_points=53532
```

After changing optimizer math, compare:

- Valid point count.
- `stats.json`.
- `subset_field_mag.png`.
- `subset_field_u.png`.
- `subset_field_v.png`.
- Overall visual trend in
  `case/2D/ring/padding_roi_truncation_test/overview.png`.

Do not commit generated case outputs unless the project owner explicitly asks.

## Build And Test Workflow

Use the existing `build` directory. Do not create `build-ninja`, `build-tests`,
or other parallel build folders.

Build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --target traditional_dic_tests subset_dic_diagnostic'
```

Run focused subset tests:

```powershell
build\traditional_dic_tests.exe --gtest_filter=Icgn.*:ForwardGaussNewtonSolver.*:SubsetDIC.*:ReliabilityPropagation.*:SeedSelector.*:SubsetRegion.*:YamlConfig.*:SubsetInitializer.*
```

Run full tests before final handoff:

```powershell
build\traditional_dic_tests.exe
```

At the time of this handoff, the full suite baseline is:

```text
115 tests total
106 passing
9 known failures in subset/mesh/initialization
```

If you change this baseline, explain exactly which tests changed and why.

## Git Workflow

The project owner will provide the GitHub repository URL or an invitation.
Use that URL for clone/fetch/push.

Recommended workflow:

```powershell
git clone <github-url> Traditional-DIC-subset
cd Traditional-DIC-subset
git checkout main
git pull --ff-only
git checkout -b feature/subset-optimizer-completion
```

During work:

```powershell
git status
git diff
git add <only subset-related files>
git commit -m "Complete subset optimizer dispatch variants"
```

Before push:

```powershell
git pull --rebase origin main
git push -u origin feature/subset-optimizer-completion
```

Open a pull request from `feature/subset-optimizer-completion` into `main`.

Pull request description must include:

- Implemented optimizer combinations.
- Files changed.
- Tests run and results.
- Ring case stats and screenshots if generated.
- Any known limitations.
- Confirmation that Mesh-DIC files were not changed, or a clear explanation if
  shared modules had to change.

Do not force-push over someone else's branch. Do not merge your own pull
request unless the project owner asks you to.

## Hard Constraints

- Keep the high-level Subset-DIC API stable.
- Keep the implementation single-threaded.
- Do not remove existing ICGN ZNSSD functionality.
- Do not change Mesh-DIC behavior.
- Do not rewrite B-spline interpolation unless a tested bug fix is required.
- Do not commit generated `build/`, `case/2D/ring/...` outputs, Python caches,
  or binary artifacts.
