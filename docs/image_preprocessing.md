# Image Reading And Preprocessing

Traditional-DIC uses one image preparation contract for Subset-DIC and Mesh-DIC.

## Common Image Input

Images enter the C++ core as `dic::Image`, a row-major single-channel floating-point image.
Path-based loading is handled by `dic::Image(path, ImageLoadOptions)`.

Recommended defaults:

- Load as grayscale.
- Scale speckle image intensities to unit range during `dic::Image(path)` loading by dividing
  every pixel by the full-image maximum gray value.
- Preserve source intensity values only when a caller explicitly requests
  `ImageIntensityScale::Preserve`.
- Load ROI images through `dic::Mask(path)`, not `dic::Image(path)`. ROI masks are not intensity
  normalized; non-zero pixels are treated as valid ROI pixels.
- Apply normalization as a separate step so algorithm choices are visible in configuration.

## Common B-Spline Precompute

Both Subset-DIC and Mesh-DIC should perform full-image B-spline preprocessing before correlation.
The shared configuration is `dic::BSplinePrecomputeConfig`, exposed through:

- `SubsetConfig::image_precompute`
- `MeshConfig::image_precompute`

Supported interpolation degrees are:

- `BSplineDegree::Linear`
- `BSplineDegree::Cubic`
- `BSplineDegree::Quintic`

The default coefficient path uses symmetric padding, FFT deconvolution prefiltering, and a
minimum border width of 3 pixels.

The preprocessor returns `BSplinePrecomputedImage`, which owns:

- padded coefficient image
- QK matrix
- per-pixel local polynomial blocks
- `gradient_x`
- `gradient_y`

For ICGN-style solvers, reference-image gradients should be read from this same precomputed data.
For Forward Gauss-Newton, current/deformed-image gradients at warped points must be evaluated
each iteration with `BSplineInterpolator::gradient(x, y)` because the warped coordinates change
with the displacement update.

## Algorithm-Specific Policy

Subset-DIC and Mesh-DIC may still choose different default normalization policies, ROI handling,
seed initialization, element/subset sampling, and solver parameters. Those are algorithm-level
policies. They should not fork the image-gradient source.
