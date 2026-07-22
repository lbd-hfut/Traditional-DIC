# Correlation

Current Status: Basic Criteria Implemented.

The correlation module compares sampled subset intensity vectors. It is used by
initialization, Subset-DIC, and Mesh-DIC, but it does not sample images or decide
ROI membership by itself.

## Criteria

```text
SSD
  sum(w * (f - g)^2)
  Lower is better. Identical vectors return 0.

ZNSSD
  zero-mean, weighted unit-norm SSD
  Lower is better. Perfect normalized match returns 0.

ZNCC
  zero-mean, weighted normalized cross-correlation
  Higher is better. Perfect positive correlation returns 1.
```

## ROI / Mask Samples

Subset windows can cross an ROI or mask boundary. In that case, the caller must
pass one weight per sampled pixel:

```text
weight = 1.0  valid ROI / mask sample
weight = 0.0  non-ROI sample ignored by correlation
0 < weight < 1 optional fractional boundary coverage
```

The unweighted API is equivalent to passing all-one weights.

Weighted ZNSSD and ZNCC compute the mean, norm, residual, and score only through
the weighted samples:

```text
mean_f = sum(w * f) / sum(w)
mean_g = sum(w * g) / sum(w)

norm_f = sqrt(sum(w * (f - mean_f)^2))
norm_g = sqrt(sum(w * (g - mean_g)^2))

ZNSSD = sum(w * (normalized_f - normalized_g)^2)
ZNCC  = sum(w * normalized_f * normalized_g)
```

This matches the intended Ncorr-style behavior: points inside a subset but
outside the ROI/mask do not affect the correlation value.

## Validation Rules

Correlation inputs must satisfy:

```text
reference.size == deformed.size
weights.size == reference.size when weights are supplied
all values are finite
weights are nonnegative
sum(weights) > 0
ZNSSD/ZNCC weighted variance is nonzero
```

## TODO

```text
ROI / Mask -> subset sampling -> weight vector construction
minimum valid sample count
minimum valid sample fraction
weighted residual vector output for nonlinear solvers
SIMD/OpenMP batch evaluation
```
