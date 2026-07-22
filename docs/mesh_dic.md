# Mesh-DIC

Current Status: Project Skeleton / Interfaces Only.

TODO: Document algorithms, inputs, outputs, validation datasets, and development checkpoints.

## Mesh Generation

```text
ROI / Mask
-> Mesh Generator
-> T3 / Q4 / Q8 Mesh
-> Displacement Initialization
-> Mesh Solver
```

Mesh Generation and Mesh displacement initialization are separate steps.

Mesh Generation initializes:

```text
node coordinates
element topology
connectivity
```

Displacement Initialization initializes:

```text
u0
v0
```

Do not mix mesh topology creation with Gauss-Newton, ICGN, triangulation, or strain calculation.

## Mesh Solvers

Mesh-DIC solver code is split by optimization formulation:

```text
mesh/solver/
  global_icgn.hpp/.cpp
      FE-based Global IC-GN.
      Reference image gradients and the Hessian/stiffness matrix are
      precomputed from the reference configuration and reused across
      iterations. Each iteration updates only the residual/right-hand side,
      solves for the nodal increment, and updates nodal displacement.

  global_gauss_newton.hpp/.cpp
      Forward-additive Global Gauss-Newton.
      The Jacobian/Hessian may be reassembled at each iteration using the
      current warped/deformed image state.

  assembler.hpp/.cpp
      Shared sparse Hessian/right-hand-side assembly utilities.

  linear_solver.hpp/.cpp
      Shared sparse linear system solver.
```

The default Mesh-DIC optimization method is now `global_icgn`, because
FE-based Global IC-GN is a standard and efficient path for 2D global DIC
when reference gradients and the Hessian can be reused.

Future validation should compare:

```text
Global IC-GN
vs
Forward Global Gauss-Newton
```

on zero displacement, integer translation, subpixel translation, affine
deformation, noise robustness, Hessian reuse, and convergence behavior.
