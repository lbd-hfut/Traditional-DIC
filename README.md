# Traditional-DIC

Current Status: Core Infrastructure In Progress

Traditional-DIC is a C++17/Python implementation workspace for traditional digital image correlation workflows. The project now has the main module boundaries in place, a buildable C++ core, focused tests for early numerical infrastructure, and placeholder interfaces for the larger Subset-DIC, Mesh-DIC, and 3D reconstruction pipeline.

## Project Structure

- `include/dic/core`: image loading, preprocessing, ROI, masks, common types, and result containers.
- `include/dic/interpolation`: unified B-spline precompute/interpolation interfaces with configurable degree.
- `include/dic/correlation`: SSD, ZNSSD, and ZNCC criteria.
- `include/dic/initialization`: integer search, SIFT-ready initialization, feature matching, and subset-based mesh initialization.
- `include/dic/subset`: 2D Subset-DIC orchestration, shape functions, ICGN solver, and propagation.
- `include/dic/mesh`: 2D Mesh-DIC data structures, elements, coordinate conversion, sparse assembly, and forward Gauss-Newton.
- `include/dic/calibration`, `geometry`, `reconstruction`: stereo/multiview camera models, triangulation, 3D shape, and displacement.
- `bindings/python` and `python/traditional_dic`: pybind11 entry points and high-level Python API placeholders.

## Dependencies

Required: CMake 3.16+, C++17 compiler, Eigen. Optional: OpenCV, pybind11, GoogleTest.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Tests

When GoogleTest is available, configure with tests enabled and run CTest:

```bash
cmake -S . -B build-ninja-tests -DTRADITIONAL_DIC_BUILD_TESTS=ON
cmake --build build-ninja-tests
ctest --test-dir build-ninja-tests --output-on-failure
```

The current test suite covers image construction/loading behavior, preprocessing normalization, B-spline precompute and gradients, correlation criteria, and placeholder coverage for the remaining module interfaces.

## Python Bindings

The `_traditional_dic` pybind11 module is optional. The binding structure is present, while most high-level Python workflows still raise `NotImplementedError` until the corresponding C++ algorithms are implemented.

## Implemented So Far

- Single-channel floating-point image container and path-based image loading.
- Image preprocessing policies for max-intensity, global mean/std, and ROI mean/std normalization.
- Shared B-spline precompute configuration for Subset-DIC and Mesh-DIC.
- B-spline coefficient/precompute path with local polynomial blocks and gradient maps.
- Runtime B-spline interpolation and analytic gradients at arbitrary subpixel coordinates.
- SSD, ZNSSD, and ZNCC correlation criteria.
- Subset-DIC region/nodelist utilities, circular subset extraction, and local Cholesky solve helpers adapted from the standalone SubsetDIC project.
- CMake integration for the core library, optional tests, optional Python bindings, benchmarks, and local diagnostic tools.

## Still In Progress

The larger DIC algorithms are intentionally staged behind interfaces. SIFT matching, integer search completion, Subset ICGN, Subset forward Gauss-Newton, second-order subset shape functions, reliability propagation, mesh global Gauss-Newton, element-level assembly details, calibration, triangulation refinement, 3D reconstruction, strain, filtering, and coordinate transforms still need production implementations and non-placeholder tests.

## Recommended Development Order

1. Finish integer search against the shared image/interpolation/correlation path.
2. Implement first-order Subset ICGN with focused synthetic displacement tests.
3. Add reliability propagation and seed selection tests.
4. Complete Q4/Q8 coordinate conversion, element sampling, and sparse assembly.
5. Implement mesh global Gauss-Newton on small synthetic fields.
6. Bring stereo triangulation and reconstruction orchestration out of placeholder status.
7. Wire the Python API to implemented C++ workflows as each backend becomes stable.
