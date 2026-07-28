# Correlation

The correlation module compares sampled intensity vectors. It does not sample
images, choose ROI pixels, run Gauss-Newton, or decide propagation acceptance by
itself. Callers build the reference/deformed vectors and optional weights, then
call a criterion.

## Implemented Criteria

Implemented public classes:

- `dic::SSDCorrelation`
- `dic::ZNSSDCorrelation`
- `dic::ZNCCCorrelation`

Common interface:

```cpp
double evaluate(const Eigen::VectorXd& reference,
                const Eigen::VectorXd& deformed) const;

double evaluate(const Eigen::VectorXd& reference,
                const Eigen::VectorXd& deformed,
                const Eigen::VectorXd& weights) const;
```

## Meaning

SSD:

```text
SSD = sum(w * (f - g)^2)
```

Lower is better. Identical vectors return `0`.

ZNSSD:

```text
mean_f = sum(w * f) / sum(w)
mean_g = sum(w * g) / sum(w)
norm_f = sqrt(sum(w * (f - mean_f)^2))
norm_g = sqrt(sum(w * (g - mean_g)^2))

ZNSSD = sum(w * ((f - mean_f) / norm_f - (g - mean_g) / norm_g)^2)
```

Lower is better. Perfect normalized match returns `0`. A perfect inverse
normalized match approaches `4`.

ZNCC:

```text
ZNCC = sum(w * ((f - mean_f) / norm_f) * ((g - mean_g) / norm_g))
```

Higher is better. Perfect positive normalized correlation returns `1`.

## ROI And Mask Weights

Subset windows and mesh elements may cross ROI/mask boundaries. The caller must
represent validity using one weight per sampled pixel:

```text
weight = 1.0  valid ROI/mask sample
weight = 0.0  invalid sample ignored by correlation
0 < weight < 1 optional fractional coverage
```

The unweighted API is equivalent to all-one weights.

Current Subset-DIC ROI truncation is mostly handled before vector evaluation:
invalid samples can be excluded from the sample set. The weighted API remains
available for code paths that prefer fixed-size vectors or fractional weights.

Mesh-DIC element sampling should use weights when element samples are clipped by
ROI/mask boundaries, but mesh generation itself must stay topology-only.

## Validation Rules

All criteria validate:

- `reference.size() == deformed.size()`
- `weights.size() == reference.size()` when weights are supplied
- inputs contain finite values
- weights are finite and nonnegative
- weighted calls include at least one nonzero weight

ZNSSD and ZNCC additionally require nonzero weighted variance/norm for both
vectors.

## Solver Objective Selection

The correlation criteria are lower-level vector functions. Optimizer dispatch
uses a smaller objective enum:

```cpp
enum class CorrelationCriterionKind {
    SSD,
    ZNSSD
};
```

Current intended solver objectives:

```text
Subset-DIC optimizers: SSD or ZNSSD
Mesh-DIC optimizers:   SSD or ZNSSD
Seed quality ranking:  ZNCC, ZNSSD, or SSD
```

Do not treat ZNCC as a Gauss-Newton objective unless the optimizer math and
configuration are explicitly extended. ZNCC is currently used as a score-style
criterion for initialization and diagnostics.

## Current Integration Points

Subset-DIC:

- Existing first-order ICGN implementation optimizes a ZNSSD-style normalized
  residual internally.
- SSD and second-order variants are scaffolded through explicit solver dispatch.
- Reliability propagation uses the result `correlation` field as a ZNSSD-like
  lower-is-better acceptance score.

Mesh-DIC:

- Current global solver path is SSD-like.
- ZNSSD global mesh paths are scaffolded for completion.
- Regularization is handled separately through `regularization_alpha`; it is not
  part of the correlation criterion.

Initialization:

- Integer search uses score/evaluation logic over candidate integer
  displacements.
- Seed selection supports `zncc`, `znssd`, and `ssd` quality metrics with the
  correct pass direction.

## Tests

Relevant tests:

- `tests/core/test_znssd.cpp`
- `tests/subset/test_yaml_config.cpp`
- `tests/subset/test_icgn.cpp`
- `tests/subset/test_forward_gauss_newton.cpp`
- `tests/mesh/test_mesh_gauss_newton.cpp`
- `tests/mesh/test_mesh_icgn_synthetic.cpp`

When implementing new optimizer objectives, add tests at the solver level, not
only at the vector criterion level.
