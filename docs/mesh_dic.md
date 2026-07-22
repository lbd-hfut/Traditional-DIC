# Mesh Dic

Current Status: Project Skeleton / Interfaces Only.

TODO: Document algorithms, inputs, outputs, validation datasets, and development checkpoints.

## Mesh Generation

```text
ROI / Mask
↓
Mesh Generator
↓
T3 / Q4 / Q8 Mesh
↓
Displacement Initialization
↓
Global Gauss-Newton
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
