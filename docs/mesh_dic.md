# Mesh-DIC Optimizer Completion Task

This document is the handoff brief for the teammate or AI coding agent assigned
to complete the Mesh-DIC optimizer family and improve global-field smoothness.
Read this file before editing code.

## Scope

Only work on Mesh-DIC optimizer completion and mesh regularization tuning. Do
not refactor Subset-DIC, reconstruction, postprocess, calibration, Python
packaging, or unrelated tools.

Allowed areas:

- `include/dic/mesh/**`
- `src/mesh/**`
- `config/mesh_2d.yaml`
- `tests/mesh/**`
- `tools/mesh_*diagnostic.cpp`
- `tools/visualize_mesh_displacement_field.py`, only for mesh result
  visualization

Avoid changing shared modules unless necessary:

- `include/dic/interpolation/**` and `src/interpolation/**`
- `include/dic/core/**` and `src/core/**`
- `include/dic/correlation/**` and `src/correlation/**`
- `include/dic/initialization/**` and `src/initialization/**`

If a shared module must change, explain why in the commit message and run the
subset-related tests listed below to make sure Subset-DIC was not broken.

## Current State

Mesh-DIC is no longer just a skeleton. The current pipeline already includes:

- ROI/mask-based automatic annular mesh generation for T3, Q4, and Q8.
- Manual mesh loading from `nodes_*.txt` and `elements_*.txt`.
- Mesh topology separated from displacement initialization.
- Global-to-local coordinate mapping for T3/Q4/Q8 elements.
- FEM sampling/inform construction over image pixels.
- B-spline image precompute shared with the rest of the project.
- Reference image gradient extraction and stiffness/Hessian assembly.
- Sparse global solve utilities.
- SIFT-based node initialization with robust match filtering and nodal
  interpolation.
- Integer-search fallback initialization for nodes not initialized by SIFT.
- Optional mirror image padding for interpolation/search near image boundaries.
- Current global solver dispatch between `global_icgn` and
  `global_forward_gn`.
- Config field `regularization_alpha` for displacement-gradient
  regularization.
- Config field `CorrelationCriterionKind objective` with current default `SSD`.
- Objective-aware mesh solver dispatch scaffolding.

Current important files:

- Public config: `include/dic/mesh/mesh_config.hpp`
- Main controller: `src/mesh/mesh_dic.cpp`
- Mesh model: `include/dic/mesh/mesh.hpp`, `src/mesh/mesh.cpp`
- Elements: `src/mesh/element/t3.cpp`, `q4.cpp`, `q8.cpp`
- Global-to-local mapping: `src/mesh/coordinate/**`
- Assembly and global solvers: `src/mesh/solver/fem_assembler.cpp`
- Solver headers: `include/dic/mesh/solver/**`
- Auto ROI mesh generation: `src/mesh/generation/roi_mesh_generator.cpp`
- Ring diagnostics:
  - `tools/mesh_auto_generation_diagnostic.cpp`
  - `tools/mesh_sift_pipeline_diagnostic.cpp`
  - `tools/visualize_mesh_displacement_field.py`

The current implemented production route is:

```text
mesh generation: ROI auto mesh
element types:   T3, Q4, Q8
initialization:  SIFT node initialization + fill missing nodes
objective:       SSD-like global residual path
optimizer:       global_icgn or forward_gauss_newton dispatch
regularization:  alpha is wired into assembly but needs tuning/validation
```

The following combinations need completion or validation:

```text
objective ssd   + optimizer global_icgn
objective ssd   + optimizer forward_gauss_newton
objective znssd + optimizer global_icgn
objective znssd + optimizer forward_gauss_newton
```

The SSD paths currently have working code in `src/mesh/solver/fem_assembler.cpp`
and are used by the ring diagnostics. The ZNSSD paths currently route through
explicit placeholders in `src/mesh/mesh_dic.cpp`; complete these without
breaking the current SSD baseline.

## Required Implementation

Complete the Mesh-DIC optimizer family without changing the high-level Mesh-DIC
pipeline.

1. Preserve the current SSD global solver behavior.
   The existing ring-case trend must remain recognizable after your changes.

2. Implement ZNSSD global ICGN.
   Use element sampling and B-spline interpolation consistently with the SSD
   path, but normalize reference/deformed intensity vectors according to ZNSSD.

3. Implement ZNSSD forward Gauss-Newton.
   The forward-additive path should reassemble the appropriate Jacobian/Hessian
   terms when the warped/deformed state changes.

4. Confirm and clean up SSD Global ICGN vs SSD Forward GN semantics.
   If `global_icgn` is still internally a forward-additive update with a reused
   Hessian, document that clearly and correct it if the math is wrong.

5. Add explicit status/reporting for unimplemented mesh solver paths if needed.
   `MeshDIC::compute()` currently returns only `std::vector<Displacement2D>`.
   If you need solver-level status, add it minimally and keep existing callers
   compatible.

6. Keep mesh generation separate from displacement solve.
   Do not mix topology generation with optimizer math.

7. Keep initialization separate from global solve.
   SIFT and integer-search node initialization should remain modular.

8. Do not introduce multithreading in this task.
   Correctness and smoothness validation come first.

## Regularization Alpha Tuning Task

The current mesh field is less smooth than results produced by
`YangMechanicsGroupUTAustin/2D_FE_Global_DIC.git` on comparable cases. Add and
validate a regularization-alpha tuning workflow.

Reference repository for comparison:

```text
https://github.com/YangMechanicsGroupUTAustin/2D_FE_Global_DIC.git
```

Do not copy code blindly from that repository. Use it to understand expected
global DIC smoothness, regularization behavior, and result diagnostics.

Required alpha work:

1. Confirm where `regularization_alpha` enters the stiffness/Hessian assembly.
   Start with `src/mesh/mesh_dic.cpp` and `src/mesh/solver/fem_assembler.cpp`.

2. Add a diagnostic sweep over alpha values, for example:

```text
0
1e-8
1e-7
1e-6
1e-5
1e-4
1e-3
```

3. For each alpha, compute:

- Mean displacement magnitude.
- Max displacement magnitude.
- Standard deviation of neighboring nodal displacement differences.
- Optional element-gradient smoothness metric.
- Visual field overlays for `|U|`, `u`, and `v`.

4. Prefer a small nonzero alpha if it improves smoothness without flattening
   the physical displacement trend.

5. Update `config/mesh_2d.yaml` comments if a recommended alpha range is found.
   Do not silently change the default value unless the project owner approves.

6. Save generated alpha-sweep outputs under:

```text
case/2D/ring/padding_roi_truncation_test/mesh_alpha_sweep
```

Generated outputs are for review only and should not be committed unless the
project owner explicitly asks.

## Tests To Add Or Update

Add focused tests for every completed optimizer path.

Required unit coverage:

- SSD Global ICGN still recovers zero displacement.
- SSD Global ICGN still recovers synthetic subpixel translation.
- SSD Forward GN still converges on a synthetic Q4 case.
- ZNSSD Global ICGN recovers zero displacement.
- ZNSSD Global ICGN recovers synthetic subpixel translation.
- ZNSSD Forward GN recovers zero displacement.
- ZNSSD Forward GN recovers synthetic subpixel translation.
- T3/Q4/Q8 global-to-local tests remain passing.
- Regularization alpha reduces a synthetic high-frequency displacement
  perturbation without destroying mean displacement.

Existing relevant tests:

- `tests/mesh/test_mesh_gauss_newton.cpp`
- `tests/mesh/test_mesh_icgn_synthetic.cpp`
- `tests/mesh/test_local_icgn.cpp`
- `tests/mesh/test_global_to_natural.cpp`
- `tests/mesh/test_assembler.cpp`
- `tests/mesh/test_t3.cpp`
- `tests/mesh/test_q4.cpp`
- `tests/mesh/test_q8.cpp`

If shared interpolation or initialization code changes, also run:

- `tests/subset/test_icgn.cpp`
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

Correct mesh route:

1. Generate the auto mesh from the ROI mask.
2. Use the generated `auto_mesh_quality_target35` T3/Q4/Q8 nodes and elements.
3. Run SIFT node initialization and global mesh solve.
4. Generate field visualizations.
5. Update the overview image.

Do not use external mesh files from another project unless explicitly asked.

Run auto mesh generation:

```powershell
build\mesh_auto_generation_diagnostic.exe `
  case\2D\ring\003.bmp `
  case\2D\ring\auto_mesh_quality_target35 `
  35 18 55
```

Expected mesh generation baseline:

```text
target_element_size=35
radial_divisions=8
circumferential_divisions=108
T3 nodes=972 elements=1728
Q4 nodes=972 elements=864
Q8 nodes=2808 elements=864
```

Run T3/Q4/Q8 displacement:

```powershell
build\mesh_sift_pipeline_diagnostic.exe `
  case\2D\ring\001.bmp `
  case\2D\ring\002.bmp `
  case\2D\ring\auto_mesh_quality_target35\T3\nodes_T3.txt `
  case\2D\ring\auto_mesh_quality_target35\T3\elements_T3.txt `
  case\2D\ring\padding_roi_truncation_test\mesh\T3 `
  T3 5 1 icgn sift 1e-3

build\mesh_sift_pipeline_diagnostic.exe `
  case\2D\ring\001.bmp `
  case\2D\ring\002.bmp `
  case\2D\ring\auto_mesh_quality_target35\Q4\nodes_Q4.txt `
  case\2D\ring\auto_mesh_quality_target35\Q4\elements_Q4.txt `
  case\2D\ring\padding_roi_truncation_test\mesh\Q4 `
  Q4 5 1 icgn sift 1e-3

build\mesh_sift_pipeline_diagnostic.exe `
  case\2D\ring\001.bmp `
  case\2D\ring\002.bmp `
  case\2D\ring\auto_mesh_quality_target35\Q8\nodes_Q8.txt `
  case\2D\ring\auto_mesh_quality_target35\Q8\elements_Q8.txt `
  case\2D\ring\padding_roi_truncation_test\mesh\Q8 `
  Q8 5 1 icgn sift 1e-3
```

Generate visualizations:

```powershell
C:\Users\lbd\miniconda3\python.exe tools\visualize_mesh_displacement_field.py `
  --nodes case\2D\ring\auto_mesh_quality_target35\T3\nodes_T3.txt `
  --elements case\2D\ring\auto_mesh_quality_target35\T3\elements_T3.txt `
  --u case\2D\ring\padding_roi_truncation_test\mesh\T3\final_U.csv `
  --out-dir case\2D\ring\padding_roi_truncation_test\mesh\T3\field_visualization `
  --etype T3 --width 1280 --height 1280

C:\Users\lbd\miniconda3\python.exe tools\visualize_mesh_displacement_field.py `
  --nodes case\2D\ring\auto_mesh_quality_target35\Q4\nodes_Q4.txt `
  --elements case\2D\ring\auto_mesh_quality_target35\Q4\elements_Q4.txt `
  --u case\2D\ring\padding_roi_truncation_test\mesh\Q4\final_U.csv `
  --out-dir case\2D\ring\padding_roi_truncation_test\mesh\Q4\field_visualization `
  --etype Q4 --width 1280 --height 1280

C:\Users\lbd\miniconda3\python.exe tools\visualize_mesh_displacement_field.py `
  --nodes case\2D\ring\auto_mesh_quality_target35\Q8\nodes_Q8.txt `
  --elements case\2D\ring\auto_mesh_quality_target35\Q8\elements_Q8.txt `
  --u case\2D\ring\padding_roi_truncation_test\mesh\Q8\final_U.csv `
  --out-dir case\2D\ring\padding_roi_truncation_test\mesh\Q8\field_visualization `
  --etype Q8 --width 1280 --height 1280
```

Update combined overview:

```powershell
C:\Users\lbd\miniconda3\python.exe `
  case\2D\ring\padding_roi_truncation_test\make_overview.py
```

Recent mesh displacement baseline after the current rebuilt diagnostics:

```text
T3 nodes=972  mag_mean=0.550529  mag_max=1.272315
Q4 nodes=972  mag_mean=0.550386  mag_max=1.269631
Q8 nodes=2808 mag_mean=0.537294  mag_max=1.320356
```

Compare:

- `final_U.csv` statistics.
- `field_visualization/*_field_mag_mesh_overlay.png`.
- `field_visualization/*_field_u_mesh_overlay.png`.
- `field_visualization/*_field_v_mesh_overlay.png`.
- Overall visual trend in
  `case/2D/ring/padding_roi_truncation_test/overview.png`.

Do not commit generated case outputs unless the project owner explicitly asks.

## Build And Test Workflow

Use the existing `build` directory. Do not create `build-ninja`, `build-tests`,
or other parallel build folders.

Build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --target traditional_dic_tests mesh_auto_generation_diagnostic mesh_sift_pipeline_diagnostic mesh_dic_diagnostic'
```

Run focused mesh tests:

```powershell
build\traditional_dic_tests.exe --gtest_filter=MeshDICIntegration.*:MeshICGN.*:LocalICGN.*:FEMAssembler.*:G2L_*:Q4Element.*:Q8Element.*:T3Element.*:ROIMeshGenerator.*
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
git clone <github-url> Traditional-DIC-mesh
cd Traditional-DIC-mesh
git checkout main
git pull --ff-only
git checkout -b feature/mesh-optimizer-regularization
```

During work:

```powershell
git status
git diff
git add <only mesh-related files>
git commit -m "Complete mesh optimizer dispatch and regularization tuning"
```

Before push:

```powershell
git pull --rebase origin main
git push -u origin feature/mesh-optimizer-regularization
```

Open a pull request from `feature/mesh-optimizer-regularization` into `main`.

Pull request description must include:

- Implemented optimizer combinations.
- Regularization alpha sweep results.
- Files changed.
- Tests run and results.
- Ring case statistics and screenshots if generated.
- Any known limitations.
- Confirmation that Subset-DIC files were not changed, or a clear explanation if
  shared modules had to change.

Do not force-push over someone else's branch. Do not merge your own pull
request unless the project owner asks you to.

## Hard Constraints

- Keep the high-level Mesh-DIC API stable.
- Keep mesh generation separate from solving.
- Keep displacement initialization separate from solving.
- Preserve the current SSD ring-case baseline unless the change is intentional
  and explained.
- Do not change Subset-DIC behavior.
- Do not rewrite B-spline interpolation unless a tested bug fix is required.
- Do not commit generated `build/`, `case/2D/ring/...` outputs, Python caches,
  or binary artifacts.
