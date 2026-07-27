# Initialization

Traditional-DIC treats seed-point initialization as a two-stage process shared by
Subset-DIC and Mesh-DIC.

## Integer Search

The integer stage searches a circular reference subset over an integer
displacement window and scores each candidate with ZNCC.

Configuration keys:

- `initialization.integer_search.subset_radius`: circular subset radius used by
  the integer search.
- `initialization.integer_search.search_radius`: maximum integer displacement
  searched in both image directions.

## Subpixel Refinement

The subpixel stage refines the integer displacement with a configured subset
shape function and optimizer. The solver dispatch supports these combinations:

- `shape_function: first_order`
- `shape_function: second_order`
- `optimizer: icgn`
- `optimizer: forward_gauss_newton`

The currently implemented numerical refinement path is first-order ICGN. The
second-order shape function and forward Gauss-Newton optimizer are selectable
and return stable placeholders until their numerical kernels are implemented.

Configuration keys:

- `initialization.subpixel_refinement.enabled`
- `initialization.subpixel_refinement.shape_function`
- `initialization.subpixel_refinement.optimizer`
- `initialization.subpixel_refinement.subset_radius`
- `initialization.subpixel_refinement.max_iterations`
- `initialization.subpixel_refinement.convergence_threshold`

The subpixel subset radius is intentionally separate from the integer-search
subset radius, because robust integer matching and local ICGN refinement often
benefit from different support sizes.

## Seed Selection

Subset-DIC seed selection is automatic. `SeedSelector` generates a configured
number of candidate seed points uniformly over the ROI, evaluates each point
with the same seed initialization and subpixel refinement settings used by the
main Subset-DIC solve, and then chooses the best seed without manual user
selection.

The first implementation is single-threaded. Candidate ranking is:

1. Keep candidates whose configured quality metric passes the threshold.
2. From those candidates, choose the one with the largest displacement norm.

Supported quality metric directions are:

- `zncc`: higher is better, pass with `quality >= min_zncc`.
- `znssd`: lower is better, pass with `quality <= max_znssd`.
- `ssd`: lower is better, pass with `quality <= max_ssd`.

Configuration keys:

- `seed_selection.seed_count`
- `seed_selection.threads`
- `seed_selection.quality_metric`
- `seed_selection.max_znssd`
- `seed_selection.min_zncc`
- `seed_selection.max_ssd`
- `seed_selection.min_displacement_norm`
