# Initialization

Traditional-DIC uses initialization modules to produce a reasonable starting
displacement before nonlinear Subset-DIC or Mesh-DIC solvers run.

Initialization is shared in concept, but Subset-DIC and Mesh-DIC use it at
different granularity:

- Subset-DIC initializes seed points for reliability propagation.
- Mesh-DIC initializes nodal displacements before the global FEM solve.

## Shared Types

Important config and result types:

- `SeedInitializationConfig`
- `SeedIntegerSearchConfig`
- `SeedSubpixelRefinementConfig`
- `SeedSelectionConfig`
- `InitialDisplacement`
- `FeatureMatcherConfig`
- `FeatureMatch`

Important enums:

```cpp
SeedInitializationMethod::{IntegerSearch, SIFT}
SubsetShapeFunctionMethod::{FirstOrder, SecondOrder}
SubsetOptimizationMethod::{ICGN, ForwardGaussNewton}
CorrelationCriterionKind::{SSD, ZNSSD}
SeedQualityMetric::{ZNCC, ZNSSD, SSD}
```

## Integer Search

`IntegerSearchInitializer` estimates integer-pixel displacement at a point.

Public entry points:

- `estimate(reference, deformed, point)`
- `estimate_with_interpolators(...)`
- `estimate_with_mask(...)`
- `estimate_with_mask_interpolators(...)`

Configuration:

```yaml
initialization:
  integer_search:
    subset_radius: 37
    search_radius: 30
    pyramid_enabled: true
    pyramid_scale: 4
    pyramid_refinement_radius: 4
```

Current behavior:

- Uses a circular support window.
- Supports ROI/mask-aware sampling.
- Supports pyramid search for larger displacement ranges.
- Can use shared B-spline interpolators when provided by the caller.
- Returns an `InitialDisplacement` with `u`, `v`, confidence, and validity.

Integer search is an initializer. It must not become a fallback inside
reliability propagation after a failed ICGN update unless that behavior is
explicitly reintroduced and tested. The current ncorr-alignment direction is to
propagate only through reliable solver results.

## Subpixel Refinement

`SubsetInitializer` wraps integer search and optional subpixel refinement.

Public entry points:

- `estimate(reference, deformed, point)`
- `estimate_with_interpolators(...)`
- `estimate_with_mask(...)`
- `estimate_with_mask_interpolators(...)`

Configuration:

```yaml
initialization:
  subpixel_refinement:
    enabled: true
    shape_function: first_order
    optimizer: icgn
    criterion: znssd
    subset_radius: 37
    max_iterations: 30
    convergence_threshold: 1.0e-3
```

Current implemented numerical refinement:

```text
first_order + znssd + icgn
```

Other combinations have explicit dispatch scaffolding:

```text
first_order  + ssd   + icgn
second_order + znssd + icgn
second_order + ssd   + icgn
first_order  + znssd + forward_gauss_newton
first_order  + ssd   + forward_gauss_newton
second_order + znssd + forward_gauss_newton
second_order + ssd   + forward_gauss_newton
```

When subpixel refinement succeeds, `SubsetInitializer` returns the refined
displacement and affine parameters. When it fails, it returns the integer-search
result.

## SIFT Feature Matching

`FeatureMatcher` computes sparse feature matches between the reference and
deformed images.

Configuration:

```yaml
max_features: 4000
ratio_threshold: 0.75
robust_mad_factor: 5.0
```

Each `FeatureMatch` stores:

- reference point
- deformed point
- displacement
- confidence
- validity flags for ratio/mutual/robust filtering

SIFT requires OpenCV support in the build.

## SIFT Initializer

`SIFTInitializer` estimates displacement at an arbitrary point by interpolating
nearby sparse feature matches.

Configuration:

```yaml
interpolation_neighbors: 8
interpolation_radius: 180.0
```

Public entry points:

- `estimate(reference, deformed, point)`
- `estimate_from_matches(matches, point)`

This initializer is especially useful for Mesh-DIC node initialization, where a
global set of sparse matches can be reused for many nodes.

## Subset-DIC Seed Selection

Subset-DIC uses `SeedSelector` to find reliable seed points automatically.

Current behavior:

- Generates candidates inside ROI/mask.
- Uses k-means-style sampling over ROI points for hollow/ring regions.
- Applies margin and texture checks.
- Evaluates candidates with `SubsetInitializer`.
- Keeps candidates that pass the configured quality metric.
- Chooses the passing candidate with the largest displacement norm.

Configuration:

```yaml
seed_selection:
  method: roi_kmeans
  seed_count: 64
  threads: 1
  quality_metric: znssd
  max_znssd: 0.2
  min_zncc: 0.7
  max_ssd: 0.05
  min_displacement_norm: 0.0
  min_texture_std: 0.02
  kmeans_iterations: 20
  kmeans_sample_limit: 20000
```

The `threads` field exists in config but the current implementation is
single-threaded. Do not add multithreading as part of optimizer-correctness
work.

Quality metric pass directions:

```text
zncc   pass if quality >= min_zncc
znssd  pass if quality <= max_znssd
ssd    pass if quality <= max_ssd
```

## Reliability Propagation Input

Subset reliability propagation starts from selected `PropagationSeed` records:

```cpp
struct PropagationSeed {
    Eigen::Vector2d point;
    Displacement2D displacement;
};
```

The selected seed already includes subpixel displacement and affine parameters.
Propagation snaps off-grid seeds to the nearest ncorr-style reduced-grid point,
then runs ICGN at that aligned point before pushing it into the priority queue.

Spacing convention:

```text
config reliability_propagation.spacing = ncorr gap count
full-resolution stride = spacing + 1
```

Do not interpret subset spacing as image downsampling.

## Mesh-DIC Node Initialization

Mesh-DIC initializes nodal displacement before global solving.

Current route in `src/mesh/mesh_dic.cpp`:

1. Convert mesh nodes/elements to flat arrays.
2. Optionally mirror-pad images and shift solver node coordinates.
3. Build shared B-spline precompute/interpolators.
4. Use SIFT node initialization if `SeedInitializationMethod::SIFT`.
5. For nodes not initialized by SIFT, run integer search.
6. Fill missing nodal values from nearest valid initialized nodes.
7. Run the global mesh solver.

Mesh SIFT config:

```yaml
initialization:
  method: sift
  sift_node_initialization:
    max_features: 4000
    ratio_threshold: 0.75
    interpolation_neighbors: 8
    interpolation_radius: 180.0
    robust_mad_factor: 5.0
```

Keep mesh node initialization separate from mesh generation and global solve.

## Configuration Notes

Subset config:

- `config/subset_2d.yaml`
- Current ring-case config:
  `case/2D/ring/padding_roi_truncation_test/subset/config.yaml`

Mesh config:

- `config/mesh_2d.yaml`

The current YAML parser explicitly supports subset fields including
`correlation.criterion`. Mesh diagnostics still rely heavily on command-line
arguments, so do not assume every mesh YAML field is parsed unless you verify
the code path.

## Tests

Relevant tests:

- `tests/initialization/test_integer_search.cpp`
- `tests/initialization/test_sift_initializer.cpp`
- `tests/subset/test_seed_selector.cpp`
- `tests/subset/test_subset_dic.cpp`
- `tests/subset/test_icgn.cpp`
- `tests/subset/test_yaml_config.cpp`
- `tests/mesh/test_local_icgn.cpp`
- `tests/mesh/test_mesh_gauss_newton.cpp`
- `tests/mesh/test_mesh_icgn_synthetic.cpp`

Focused run:

```powershell
build\traditional_dic_tests.exe --gtest_filter=IntegerSearch.*:SubsetInitializer.*:SIFTInitializer.*:SeedSelector.*:ReliabilityPropagation.*:SubsetDIC.*:LocalICGN.*:MeshDICIntegration.*:MeshICGN.*
```

At this checkpoint, the full suite baseline is:

```text
115 tests total
106 passing
9 known failures in subset/mesh/initialization
```

If initialization changes alter that baseline, document exactly which tests
changed and why.
