# Traditional-DIC

Current Status: Project Skeleton / Interfaces Only

Traditional-DIC is a C++17/Python engineering skeleton for traditional digital image correlation workflows. The current goal is clean module boundaries, compile-ready interfaces, and collaboration-friendly TODOs rather than full numerical algorithms.

## Project Structure

- `include/dic/core`: image, ROI, masks, common types, and result containers.
- `include/dic/interpolation`: unified B-spline interpolation interfaces with configurable degree.
- `include/dic/correlation`: SSD, ZNSSD, and ZNCC criteria.
- `include/dic/initialization`: integer search, SIFT-ready initialization, feature matching, and subset-based mesh initialization.
- `include/dic/subset`: 2D Subset-DIC orchestration, shape functions, ICGN solver, and propagation.
- `include/dic/mesh`: 2D Mesh-DIC data structures, elements, coordinate conversion, sparse assembly, and forward Gauss-Newton.
- `include/dic/calibration`, `geometry`, `reconstruction`: stereo/multiview camera models, triangulation, 3D shape, and displacement.
- `bindings/python` and `python/traditional_dic`: pybind11 and Python API placeholders.

## Dependencies

Required: CMake 3.16+, C++17 compiler, Eigen. Optional: OpenCV, pybind11, GoogleTest.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Python Bindings

The `_traditional_dic` pybind11 module is optional and currently exposes placeholder bindings. High-level Python files raise `NotImplementedError` until the C++ backend is implemented.

## Not Implemented Yet

B-spline pre-fitting, ZNSSD numerical details, SIFT matching, ICGN, mesh global Gauss-Newton, calibration, triangulation refinement, 3D reconstruction, strain, filtering, and coordinate transforms are TODO placeholders.

## Recommended Development Order

1. Core image loading and validation.
2. B-spline interpolation with tested gradients.
3. ZNSSD and integer search.
4. First-order Subset ICGN.
5. Reliability propagation.
6. Q4/Q8 mesh coordinate conversion and sparse assembly.
7. Stereo triangulation and reconstruction orchestration.
