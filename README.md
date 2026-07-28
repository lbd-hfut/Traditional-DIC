# Traditional-DIC

Traditional-DIC is a C++17 digital image correlation workspace for traditional
2D/3D DIC workflows. The current development focus is correctness and algorithm
parity for Subset-DIC and Mesh-DIC before broad API polish or parallelization.

The project currently builds a C++ core library, diagnostic executables, and a
GoogleTest test runner. Python bindings exist as an optional layer, but the
active validation path is C++ diagnostics plus focused Python visualization
scripts.

## Current Status

Implemented or partially implemented:

- Floating-point grayscale `dic::Image` and binary `dic::Mask` containers.
- Image normalization helpers and path-based image/mask loading.
- Shared B-spline precompute/interpolation for degree 1, 3, and 5.
- SSD, ZNSSD, and ZNCC vector correlation criteria.
- Integer-search initialization with ROI/mask support and optional pyramid
  search.
- SIFT feature matching and mesh nodal initialization support when OpenCV is
  available.
- Subset-DIC seed selection, ROI truncation, mirror padding, ncorr-style
  reduced-grid spacing, and reliability propagation.
- First-order ZNSSD ICGN for Subset-DIC.
- Subset optimizer dispatch scaffolding for first/second-order, SSD/ZNSSD,
  ICGN/Forward GN combinations.
- Mesh-DIC ROI auto mesh generation for T3/Q4/Q8 ring cases.
- Mesh global-to-local mapping, element sampling, assembly, SIFT node
  initialization, and SSD-like global solver paths.
- Mesh optimizer dispatch scaffolding for SSD/ZNSSD and global ICGN/Forward GN.
- Calibration, geometry, reconstruction, and postprocess modules with ongoing
  implementation work.

Known active completion work:

- Complete the remaining Subset-DIC optimizer variants.
- Complete ZNSSD Mesh-DIC global solver paths.
- Validate and tune Mesh-DIC `regularization_alpha` for smoother global fields.
- Keep Subset-DIC and Mesh-DIC changes isolated unless a shared module fix is
  explicitly required.

## Repository Layout

```text
include/dic/core              Image, Mask, ROI, result containers, common types
include/dic/interpolation     B-spline precompute and interpolator interfaces
include/dic/correlation       SSD, ZNSSD, ZNCC criteria
include/dic/initialization    Integer search, SIFT, subset initializer configs
include/dic/subset            Subset-DIC configs, shapes, solvers, propagation
include/dic/mesh              Mesh-DIC configs, mesh data, elements, solvers
include/dic/calibration       Camera and calibration models
include/dic/geometry          Projection and triangulation utilities
include/dic/reconstruction    Stereo, multi-view, 3D shape/displacement
include/dic/postprocess       Filtering, strain, coordinate transforms

src/                          C++ implementations
tests/                        GoogleTest tests
tools/                        Diagnostic executables and visualization helpers
config/                       YAML example configs
docs/                         Architecture and handoff documentation
case/                         Local validation data and generated diagnostics
bindings/, python/            Optional Python binding/API layer
```

## Key Documentation

Start here:

- `docs/architecture.md`: module boundaries and shared data flow.
- `docs/image_preprocessing.md`: `Image`, `Mask`, B-spline precompute, padding.
- `docs/correlation.md`: SSD/ZNSSD/ZNCC criteria and solver objective rules.
- `docs/initialization.md`: integer search, SIFT, seed/node initialization.
- `docs/subset_dic.md`: Subset-DIC optimizer completion handoff.
- `docs/mesh_dic.md`: Mesh-DIC optimizer and regularization handoff.
- `docs/subset_mesh_function_alignment.md`: notes on shared subset/mesh behavior.

## Dependencies

Required:

- CMake 3.16+
- C++17 compiler
- Eigen
- yaml-cpp

Common development setup on this workstation uses:

- Visual Studio Build Tools / MSVC
- Ninja generator in `build/`
- OpenCV from conda for SIFT and image IO paths
- GoogleTest for `traditional_dic_tests`

Optional:

- pybind11 for Python bindings
- NumPy/Pillow for visualization scripts

Use `C:\Users\lbd\miniconda3\python.exe` for scripts that require NumPy or
Pillow.

## Build

Use the existing single `build` directory. Do not create `build-ninja`,
`build-tests`, or other parallel build folders for normal work.

Full build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build'
```

Common focused build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --target traditional_dic_tests subset_dic_diagnostic mesh_auto_generation_diagnostic mesh_sift_pipeline_diagnostic'
```

If the build directory must be recreated, configure with tests enabled and the
same dependency locations used by the project owner. Do that only deliberately,
because generated build artifacts are not part of source control.

## Tests

Run the full test executable:

```powershell
build\traditional_dic_tests.exe
```

Current known baseline:

```text
115 tests total
106 passing
9 known failures in subset/mesh/initialization
```

Focused Subset-DIC run:

```powershell
build\traditional_dic_tests.exe --gtest_filter=Icgn.*:ForwardGaussNewtonSolver.*:SubsetDIC.*:ReliabilityPropagation.*:SeedSelector.*:SubsetRegion.*:YamlConfig.*:SubsetInitializer.*
```

Focused Mesh-DIC run:

```powershell
build\traditional_dic_tests.exe --gtest_filter=MeshDICIntegration.*:MeshICGN.*:LocalICGN.*:FEMAssembler.*:G2L_*:Q4Element.*:Q8Element.*:T3Element.*:ROIMeshGenerator.*
```

If a change modifies shared modules such as interpolation, image loading,
correlation, or initialization, run both focused sets.

## Ring Case Diagnostics

The active 2D ring validation case uses:

```text
case/2D/ring/001.bmp   reference speckle image
case/2D/ring/002.bmp   deformed speckle image
case/2D/ring/003.bmp   ROI mask
```

Subset output:

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

Current subset baseline with radius `37` and config spacing `3`:

```text
total_points=102400
valid_points=48868
invalid_points=53532
```

Mesh auto generation:

```powershell
build\mesh_auto_generation_diagnostic.exe `
  case\2D\ring\003.bmp `
  case\2D\ring\auto_mesh_quality_target35 `
  35 18 55
```

Expected mesh generation baseline:

```text
radial_divisions=8
circumferential_divisions=108
T3 nodes=972 elements=1728
Q4 nodes=972 elements=864
Q8 nodes=2808 elements=864
```

Mesh displacement uses the generated `case/2D/ring/auto_mesh_quality_target35`
nodes/elements and writes to:

```text
case/2D/ring/padding_roi_truncation_test/mesh/T3
case/2D/ring/padding_roi_truncation_test/mesh/Q4
case/2D/ring/padding_roi_truncation_test/mesh/Q8
```

Update the combined overview after generating subset and mesh visualizations:

```powershell
C:\Users\lbd\miniconda3\python.exe `
  case\2D\ring\padding_roi_truncation_test\make_overview.py
```

Generated case outputs are ignored and should not be committed unless the
project owner explicitly asks.

## Development Rules

- Keep Subset-DIC and Mesh-DIC optimizer work on separate branches when
  assigning work to multiple people or AI agents.
- Do not change unrelated modules while completing one optimizer family.
- Do not add multithreading while numerical correctness is still being aligned.
- Do not reinterpret subset spacing as image downsampling. In the current
  ncorr-style subset propagation, config spacing is a gap count and the actual
  full-resolution stride is `spacing + 1`.
- Do not rewrite shared B-spline, image, mask, or correlation modules for one
  workflow without checking the other workflow.
- Do not commit generated build files, case outputs, Python caches, or compiled
  binaries.

## Git Hygiene

Check status before staging:

```powershell
git status --short
git diff
```

Stage only files related to the task:

```powershell
git add <files>
git commit -m "<concise summary>"
```

For collaborator branches:

```powershell
git checkout main
git pull --ff-only
git checkout -b feature/<task-name>
git push -u origin feature/<task-name>
```

Open a pull request into `main`. Include implemented combinations, files
changed, tests run, ring-case stats if applicable, and known limitations.

## License

See `LICENSE`.
