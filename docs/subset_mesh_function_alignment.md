# SubsetDIC And MeshDIC Function Alignment

This note records the local differences between `SubsetDIC`, `MeshDIC`, and the
current `Traditional-DIC` image interpolation path. It is intended as a decision
aid before selecting one convention as the project-wide implementation.

## Shared Target Contract

The unified project should treat image preparation as:

1. Load speckle images as grayscale and scale pixel intensities by the full-image maximum gray value.
2. Optionally apply algorithm-level normalization, such as ROI mean/std.
3. Run full-image B-spline preprocessing on that final normalized image.
4. Use the same B-spline precomputed local polynomial blocks for interpolation and gradients.
5. Load ROI images through mask APIs only; ROI masks are not intensity-normalized.

`Traditional-DIC` already has the common API shape through `dic::Image`,
`dic::Mask`, `dic::BSplinePrecomputeConfig`, `dic::BSplineImagePreprocessor`,
and `dic::BSplineInterpolator`.

## Image Normalization Differences

SubsetDIC:

- `tests/run_cases.py` loads `001.bmp` and `002.bmp` into `float64` but keeps the
  original grayscale scale.
- ROI is loaded separately as `(roi_raw > 128).astype(np.uint8)`.
- `src/subsetdic/dic.py` converts arrays to `float64`/Fortran layout and then
  directly computes B-spline coefficients.

MeshDIC:

- `src/mesh_dic/solver.py` loads images as grayscale `float64`.
- ROI is loaded as a bool mask with `roi_mask_img > 0`.
- Before B-spline preprocessing, both reference and deformed images are
  normalized with `(img - mean_roi) / std_roi`.

Traditional-DIC:

- `dic::Image(path)` now scales speckle images by the full-image maximum gray value.
- `normalize_image()` separately exposes max-intensity, global mean/std, and ROI
  mean/std policies for array-backed workflows and solver-specific normalization.

## B-Spline Coefficient Differences

SubsetDIC:

- `src/subsetdic/bcoef.py` implements quintic coefficients only.
- Padding is symmetric: `np.pad(..., mode='symmetric')`.
- Coefficients are computed by row and column FFT deconvolution.
- Default `border.bcoef` is 20.

MeshDIC:

- `src/cpp/bspline.cpp` also implements quintic coefficient deconvolution.
- Padding is replicate/edge padding through `cv::BORDER_REPLICATE`.
- The engine clamps border to at least 3.
- Default `bcoef_border` is 3.

Traditional-DIC:

- Supports degree 1, 3, and 5 through `BSplineDegree`.
- Defaults to symmetric padding plus FFT deconvolution prefiltering.
- Clamps the B-spline border to at least 3.
- Keeps the deterministic edge-padded coefficient path only as an explicit
  compatibility mode with `use_exact_prefilter = false`.

## Local Polynomial And LUT Differences

SubsetDIC:

- Builds QK in Python with samples `[-2, -1, 0, 1, 2, 3]`.
- Precomputes `QK @ coefficient_window @ QK.T` for all valid windows.
- C++ `interp_qbs_lut()` consumes this LUT and evaluates
  `y_powers @ block @ x_powers`.
- Solver code applies `border_bcoef - 2` as the LUT offset.

MeshDIC:

- Builds the same QK sample convention in C++.
- Does not materialize a flat LUT for all pixels; it computes
  `QK @ coefficient_window @ QK.T` on demand.
- `compute_QK_B_QKT()` uses `top = y + border - 2` and
  `left = x + border - 2`.

Traditional-DIC:

- Stores per-pixel local polynomial blocks in `BSplinePrecomputedImage`.
- `BSplineInterpolator::value()` evaluates the local polynomial using the same
  `y_powers @ block @ x_powers` convention.
- The precomputed-block interface is the unified internal representation. It
  keeps SubsetDIC's LUT performance model while exposing a cleaner C++ object
  for MeshDIC.

## Gradient Differences

SubsetDIC:

- `extract_gradients_from_lut()` computes gradients at pixel center
  `dx = dy = 0.5`.
- The current SubsetDIC C++ solver reads precomputed reference gradients from
  these maps.

MeshDIC:

- `BsplineEngine::compute_gradients()` returns `temp(0, 1)` and `temp(1, 0)`,
  which are derivatives at local coordinate `dx = dy = 0`.
- In the current MeshDIC Python solver, reference gradients used by global solve
  are actually `_fd7_gradients(ref)`, not `compute_gradients()`.
- The forward GN path uses `interpolate_with_grad()` on the deformed image at
  warped positions.

Traditional-DIC:

- `BSplineImagePreprocessor::compute()` currently extracts gradient maps at
  `dx = dy = 0.5`, matching the SubsetDIC reference comparison.
- `BSplineInterpolator::gradient(x, y)` evaluates analytic gradients at arbitrary
  subpixel coordinates.
- ICGN-style solvers keep the precomputed reference-image gradient maps.
- Forward Gauss-Newton uses `BSplineInterpolator::gradient(x, y)` on the
  current/deformed image at every warped coordinate on every iteration, because
  those warped coordinates change with the displacement update.

## Recommended Unification Options

Selected default

- Full-image max-intensity image read.
- Optional normalization chosen per solver.
- Symmetric padding plus exact FFT prefilter.
- Degree selectable as 1/3/5, default quintic.
- Pixel-center precomputed gradient maps.
- Arbitrary-coordinate `value()` and `gradient()` for warped sampling.

Compatibility profile

- Full-image max-intensity image read plus optional ROI mean/std normalization.
- Replicate padding compatibility mode.
- Keep on-demand local block construction as an optimization option.
- Use arbitrary-coordinate gradient for forward GN.
- Keep FD7 gradients only as an explicit legacy/reference option.
