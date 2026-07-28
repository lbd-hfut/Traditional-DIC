# Image Reading And Preprocessing

Traditional-DIC uses shared image and interpolation modules for Subset-DIC and
Mesh-DIC. Solver-specific policies belong in the solver modules, not in the
core image container.

## Image Container

`dic::Image` is a row-major, single-channel floating-point image container.

Key API:

- `Image(path)`
- `Image(path, ImageLoadOptions)`
- `Image(width, height, data)`
- `width()`, `height()`, `size()`, `empty()`
- `contains(x, y)`
- `at(x, y)`, `set(x, y, value)`

Default file loading uses:

```cpp
ImageColorMode::Grayscale
ImageIntensityScale::Unit
```

That means speckle images loaded through `dic::Image(path)` are converted to
grayscale and scaled to unit intensity range. Use
`ImageIntensityScale::Preserve` only when an algorithm or diagnostic explicitly
needs source intensity scale.

## Mask Container

`dic::Mask` stores binary ROI/validity information.

Key API:

- `Mask(path)`
- `Mask(width, height)`
- `valid(x, y)`
- `set(x, y, bool)`
- `fill(bool)`

ROI masks should be loaded as `dic::Mask`, not as normalized `dic::Image`.
Nonzero pixels are treated as valid ROI pixels.

## Optional Image Normalization

The image module provides explicit normalization helpers:

- `normalize_max_intensity`
- `normalize_global_mean_std`
- `normalize_roi_mean_std`
- `normalize_image`

These are separate from default image loading. Use them only when the workflow
configuration or experiment requires the additional normalization step.

## B-Spline Precompute

Subset-DIC and Mesh-DIC share the B-spline interpolation module:

- `BSplinePrecomputeConfig`
- `BSplineImagePreprocessor`
- `BSplinePrecomputedImage`
- `BSplineInterpolator`

Supported degrees:

```text
1  Linear
3  Cubic
5  Quintic
```

The default config uses quintic interpolation and a minimum border of 3 pixels.
The exact prefilter path uses symmetric padding and FFT deconvolution. Lazy
precompute skips storing every local polynomial block and builds blocks on
demand.

`BSplinePrecomputedImage` stores:

- image size
- precompute config
- coefficient image
- QK matrix
- optional per-pixel local polynomial blocks
- `gradient_x`
- `gradient_y`

## Full vs Lazy Precompute

Use full precompute when many repeated value/gradient queries are expected and
memory cost is acceptable:

```cpp
BSplineImagePreprocessor preprocessor(config);
BSplinePrecomputedImage precomputed = preprocessor.compute(image);
BSplineInterpolator interp(&precomputed);
```

Use lazy precompute when memory pressure matters or when only a subset of local
blocks will be queried:

```cpp
BSplinePrecomputedImage precomputed = preprocessor.compute_lazy(image);
BSplineInterpolator interp(&precomputed);
```

Current subset seed selection and reliability propagation use lazy/deformed
precompute in performance-sensitive paths. Mesh-DIC uses shared B-spline
precompute before global assembly and solve.

## Reference Gradients

For inverse-compositional solvers, reference-image gradients should be fixed and
reused. Current Subset ICGN uses a reference gradient cache and reuses
steepest-descent/Hessian data across iterations.

For forward-additive solvers, deformed/current gradients at warped coordinates
must be evaluated as the current displacement changes:

```cpp
BSplineInterpolator::gradient(x, y)
```

Do not change gradient conventions in the shared interpolation module for only
one solver. If a solver needs a different convention, keep that adaptation local
to the solver and add tests.

## Padding Policy

Padding is a workflow policy, not a property of `dic::Image`.

Subset-DIC:

- Uses mirror padding for reference/deformed images when
  `truncate_roi_subsets` requires stable boundary interpolation/search.
- Uses zero padding for ROI masks so padded pixels are outside the original ROI.
- Computes a recommended padding radius from subset/search/B-spline settings.

Mesh-DIC:

- Supports `MeshConfig::mirror_image_padding`.
- Shifts solver node coordinates by the padding amount during solve, then writes
  results back in original image coordinates.
- Uses the same B-spline precompute module after padding.

Shared helpers live in:

- `include/dic/subset/padding.hpp`
- `src/subset/padding.cpp`

Although the helper currently lives under `subset`, it is used by both subset
and mesh code. Treat it as shared behavior.

## ROI Boundary Policy

Subset-DIC supports ROI-truncated subsets:

- `truncate_roi_subsets: true` keeps valid samples inside the ROI and allows
  boundary subsets when enough valid samples remain.
- `min_valid_sample_ratio` and `min_valid_samples` define acceptance thresholds.

Mesh-DIC should handle ROI/mask validity at element sampling/correlation time,
not by changing mesh topology in the solver. Mesh generation still owns topology
creation and filtering.

## Diagnostics

Ring-case inputs:

```text
case/2D/ring/001.bmp   reference speckle
case/2D/ring/002.bmp   deformed speckle
case/2D/ring/003.bmp   ROI mask
```

Use the conda Python when visualization scripts need NumPy/Pillow:

```powershell
C:\Users\lbd\miniconda3\python.exe <script.py>
```

Do not commit generated image outputs unless the project owner explicitly asks.

## Tests

Relevant tests:

- `tests/core/test_image.cpp`
- `tests/core/test_roi_mask.cpp`
- `tests/core/test_interpolation.cpp`
- `tests/subset/test_icgn.cpp`
- `tests/subset/test_region.cpp`
- `tests/mesh/test_mesh_gauss_newton.cpp`
- `tests/mesh/test_mesh_icgn_synthetic.cpp`

Any change to interpolation, image loading, mask loading, or padding must be
checked against both Subset-DIC and Mesh-DIC paths.
