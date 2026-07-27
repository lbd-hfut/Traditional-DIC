# Second-order Subset-DIC / IC-GN2 Technical Design

This note studies the two local DIC papers currently stored in
`docs/reference_paper` and maps their IC-GN2 details onto the current
Traditional-DIC code structure.

Primary sources:

- Gao et al., 2015, "High-efficiency and high-accuracy digital image
  correlation for three-dimensional measurement", Optics and Lasers in
  Engineering 65, 73-80.
- Bai et al., 2017, "A novel 2nd-order shape function based digital image
  correlation method for large deformation measurements", Optics and Lasers in
  Engineering 90, 48-58.

The current stage is design only. Do not implement the formulas below until the
mathematical convention and validation tests are accepted.

## 1. Problem definition

Subset-DIC estimates the local mapping between a reference subset in image
`f` and a deformed/target image `g`. For a subset center

```text
c = (xc, yc)
```

use local coordinates

```text
xi  = x - xc
eta = y - yc
r   = [xi, eta]^T.
```

The warp returns an absolute local coordinate in the target subset. The global
sampling coordinate in `g` is therefore

```text
Xg = xc + W_x(r; p)
Yg = yc + W_y(r; p).
```

Equivalently, the local displacement is

```text
d(r; p) = W(r; p) - r.
```

Traditional-DIC should use the displacement-parameter convention already used
by `FirstOrderShapeFunction`:

```text
W_x = xi  + u + ux xi + uy eta
W_y = eta + v + vx xi + vy eta
```

so identity is

```text
p = 0
W(r; 0) = r
```

and `ux`, `vy` are displacement gradients, not full mapping Jacobian entries.
The full mapping Jacobian at identity is `I`, so internally formulas that build
operator matrices must insert the identity terms explicitly.

## 2. Second-order subset warp

### 2.1 Gao/Bai paper convention

Both Gao 2015 Eq. (4) and Bai 2017 Eq. (11)/(26) use the same second-order
displacement expansion. In paper local coordinates `(x, y)`:

```text
x' = x + U(x, y)
y' = y + V(x, y)

U(x, y) = u + ux x + uy y + (1/2) uxx x^2 + uxy x y + (1/2) uyy y^2
V(x, y) = v + vx x + vy y + (1/2) vxx x^2 + vxy x y + (1/2) vyy y^2
```

Important details confirmed from the papers:

- coordinates are local subset/polar coordinates, not full image coordinates;
- `x'`, `y'` are absolute warped local coordinates, not only displacement;
- pure quadratic terms include `1/2`;
- the mixed term is `uxy x y` or `vxy x y`, with no `1/2` and no extra factor
  of 2;
- first-order parameters are displacement gradients;
- identity has all 12 displacement parameters equal to zero, while the mapping
  rows contain `1 + ux` and `1 + vy`;
- `uxx`, `uxy`, `uyy`, `vxx`, `vxy`, `vyy` have units pixel displacement per
  pixel squared.

Gao's printed parameter order is

```text
p_Gao = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]^T.
```

Bai uses the same printed order:

```text
p_Bai = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]^T.
```

### 2.2 Recommended Traditional-DIC convention

For consistency with the existing first-order solver, use:

```text
p = [u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy]^T.
```

Then

```text
W_x(xi, eta; p) =
    xi + u + ux xi + uy eta
       + 0.5 uxx xi^2 + uxy xi eta + 0.5 uyy eta^2

W_y(xi, eta; p) =
    eta + v + vx xi + vy eta
       + 0.5 vxx xi^2 + vxy xi eta + 0.5 vyy eta^2.
```

The 2 x 12 shape Jacobian `dW/dp` in this project order is

```text
dW/dp =
[ 1  0  xi eta 0  0   0.5 xi^2 xi eta 0.5 eta^2 0        0      0        ]
[ 0  1  0  0   xi eta 0        0      0         0.5 xi^2 xi eta 0.5 eta^2]
```

## 3. Jacobian and Hessian

Let the reference gradient at the reference subset sample be

```text
grad_f = [fx, fy].
```

The steepest-descent row for one sample is

```text
J(xi, eta) = grad_f * dW/dp
```

or, in the recommended project order:

```text
J = [
    fx,
    fy,
    fx xi,
    fx eta,
    fy xi,
    fy eta,
    0.5 fx xi^2,
    fx xi eta,
    0.5 fx eta^2,
    0.5 fy xi^2,
    fy xi eta,
    0.5 fy eta^2
].
```

For IC-GN, Gao Eq. (7)-(9) and Bai Eq. (7) linearize the increment on the
reference subset. Therefore:

- the gradient is the reference image gradient `grad f`, not `grad g`;
- the reference gradient does not change during iterations;
- `dW/dp` depends only on local subset coordinates;
- steepest-descent images can be precomputed before the iteration loop;
- the Hessian can be precomputed before the iteration loop.

Current code status:

- `src/subset/shape/first_order.cpp` already follows the displacement
  convention above for affine warps.
- `src/subset/shape/second_order.cpp` is only a placeholder.
- `src/subset/solver/icgn.cpp` currently computes a central-difference
  reference gradient for the first-order path. IC-GN2 should instead prefer
  `BSplineInterpolator::gradient()` or precomputed B-spline gradient images for
  consistency with the interpolation model.

## 4. ZNSSD formulation

Gao 2015 uses ZNSSD in Eq. (1), (6), (7), and (8). Bai 2017 uses ZNSSD in
Eq. (31) for numerical verification and states cubic B-spline interpolation is
used for gray value and partial derivatives.

For sample set `Omega`, define

```text
f_i     = f(x_i, y_i)
g_i(p)  = g(xc + W_x(r_i; p), yc + W_y(r_i; p))

f_bar = (1/N) sum_i f_i
g_bar = (1/N) sum_i g_i(p)

f_s = sqrt(sum_i (f_i - f_bar)^2)
g_s = sqrt(sum_i (g_i(p) - g_bar)^2)
```

ZNSSD residual:

```text
e_i(p) = (f_i - f_bar) / f_s - (g_i(p) - g_bar) / g_s
C_ZNSSD(p) = sum_i e_i(p)^2.
```

Gao's IC-GN linearization places the increment on the reference side:

```text
C_ZNSSD(delta_p) =
sum_i {
  [ f(W(r_i; delta_p)) - f_bar ] / f_s
  - [ g(W(r_i; p)) - g_bar ] / g_s
}^2.
```

First-order Taylor expansion about `delta_p = 0`:

```text
f(W(r_i; delta_p)) ~= f_i + J_i delta_p
```

so the normal equations are equivalent to

```text
H delta_p = -b

H = sum_i (J_i^T J_i) / f_s^2
b = sum_i J_i^T [
      (f_i - f_bar) / f_s - (g_i(p) - g_bar) / g_s
    ] / f_s
```

Gao prints the same structure with the residual written as

```text
f_i - f_bar - (f_s / g_s) [g_i(p) - g_bar]
```

inside the summation. This is algebraically the same up to the common scale
used in the Hessian. The current first-order implementation uses

```text
H = (2 / f_s^2) sum_i J_i^T J_i
gradient = (2 / f_s) sum_i e_i J_i
delta = -H^-1 gradient
```

which cancels the factor 2 and is compatible with the normalized-residual
form above.

Does the Jacobian require additional gray-normalization correction terms?

- In the Gao IC-GN derivation, no extra derivative of `f_bar`, `g_bar`,
  `f_s`, or `g_s` appears in the steepest-descent image. `f_bar` and `f_s`
  are constants, while `g_bar` and `g_s` are recomputed for the current
  warped target subset but not differentiated in the IC linearization.
- This matches the common Pan/Ncorr style implementation where the reference
  normalized subset and normalized target subset are compared and the
  reference steepest-descent images are scaled by `1 / f_s`.
- If a future implementation differentiates `g_bar` or `g_s`, it would no
  longer be the Gao/Bai IC-GN formula and the Hessian would not remain fixed.

## 5. Gao 2015 IC-GN2

### 5.1 Why a special second-order operator is needed

Affine IC-GN can update by composing homogeneous 3 x 3 matrices:

```text
W(p) <- W(p) o W(delta_p)^-1.
```

A direct second-order polynomial warp is not closed under composition:
composing two quadratic maps generally produces cubic and quartic terms.
Gao therefore lifts the quadratic coordinate monomials into a 6-vector and
uses a 6 x 6 matrix operator, truncating higher-order terms.

### 5.2 Gao lift and operator

Use Gao's basis order

```text
m_G = [x^2, x y, y^2, x, y, 1]^T.
```

Gao defines the lifted output as

```text
m'_G = [x'^2, x' y', y'^2, x', y', 1]^T = G_Gao(p) m_G + o(h^2).
```

The last three rows are the original second-order shape function:

```text
row_x = [0.5 uxx, uxy, 0.5 uyy, 1 + ux, uy, u]
row_y = [0.5 vxx, vxy, 0.5 vyy, vx, 1 + vy, v]
row_1 = [0, 0, 0, 0, 0, 1].
```

The first three rows are obtained by expanding `x'^2`, `x'y'`, and `y'^2`
and ignoring terms above second order:

```text
x'^2  = (1 + S1) x^2 + S2 xy + S3 y^2 + S4 x + S5 y + S6 + o(h^2)
x'y'  = S7 x^2 + (1 + S8) xy + S9 y^2 + S10 x + S11 y + S12 + o(h^2)
y'^2  = S13 x^2 + S14 xy + (1 + S15) y^2 + S16 x + S17 y + S18 + o(h^2)
```

with

```text
S1  = 2 ux + ux^2 + u uxx
S2  = 2 u uxy + 2 (1 + ux) uy
S3  = uy^2 + u uyy
S4  = 2 u (1 + ux)
S5  = 2 u uy
S6  = u^2

S7  = 0.5 (v uxx + 2 (1 + ux) vx + u vxx)
S8  = uy vx + ux vy + v uxy + u vxy + vy + ux
S9  = 0.5 (v uyy + 2 uy (1 + vy) + u vyy)
S10 = v + v ux + u vx
S11 = u + v uy + u vy
S12 = u v

S13 = vx^2 + v vxx
S14 = 2 v vxy + 2 vx (1 + vy)
S15 = 2 vy + vy^2 + v vyy
S16 = 2 v vx
S17 = 2 v (1 + vy)
S18 = v^2
```

The update is

```text
G_new = G_Gao(p) * inv(G_Gao(delta_p)).
```

Then extract the updated 12 parameters from rows 4 and 5:

```text
u_new   = G_new(4, 6)
ux_new  = G_new(4, 4) - 1
uy_new  = G_new(4, 5)
uxx_new = 2 G_new(4, 1)
uxy_new = G_new(4, 2)
uyy_new = 2 G_new(4, 3)

v_new   = G_new(5, 6)
vx_new  = G_new(5, 4)
vy_new  = G_new(5, 5) - 1
vxx_new = 2 G_new(5, 1)
vxy_new = G_new(5, 2)
vyy_new = 2 G_new(5, 3)
```

The row/column indices above are 1-based in the written formula. C++ should use
0-based indices.

Gao's method is not a strict polynomial composition of the original quadratic
warp. It is an approximate lifted operator that remains second-order by
truncation. Bai argues that the particular truncation used in Gao's `x'y'`
component may lose stable second-order precision for large or local non-uniform
deformation.

## 6. Bai 2017 improved IC-GN2

### 6.1 Bai's criticism of Gao

Bai states that the ordinary 3 x 6 second-order shape operator is suitable for
FA-GN2 and FC-GN2 but cannot be used directly in IC-GN2 because it is not
invertible. Gao enlarged it to a 6 x 6 matrix, but Bai identifies a mathematical
weakness in the lifted `x'y'` term.

Bai's Eq. (25)-(27) shows:

```text
x' y' = (x + U)(y + V) + higher-order terms.
```

Gao ignores cubic and quartic terms such as `x^3`, `y^3`, `x^2 y`, `x y^2`,
`x^4`, etc. Bai argues this is reasonable only when deformation is bounded and
small enough. If deformation is relatively large or local deformation is
non-uniform, `U + V` may be of an order that makes the discarded terms harm the
claimed `o(x^2 + y^2)` accuracy. In Bai's words, Gao's operator may lose
second-order precision in local non-uniform deformation.

### 6.2 Bai lift and operator

Bai uses the basis

```text
m_B = [1, x, y, x^2, x y, y^2]^T.
```

The original coordinate rows are

```text
E(p) =
[ 1, 0,      0,      0,         0,       0        ]
[ u, 1+ux,   uy,     0.5 uxx,   uxy,     0.5 uyy  ]
[ v, vx,     1+vy,   0.5 vxx,   vxy,     0.5 vyy  ].
```

Bai builds an invertible 6 x 6 operator

```text
G_Bai(p) = [ E(p) ]
           [ F'(p)]
         = [ A(p)  B(p)  ]
           [ C'(p) D'(p) ]
```

with `F'(p) = [C'(p) D'(p)]`.

The closed forms from Bai Eq. (A13)-(A14) are

```text
C'(p) =
[ u^2,  2 u ux + 4 u,       2 u uy       ]
[ u v,  ux v + u vx + 2 v,  uy v + u vy + 2 u ]
[ v^2,  2 v vx,             2 v vy + 4 v ]
```

and

```text
D'(p) =
[ ux^2 + u uxx + 4 ux + 1,
  2 u uxy + 2 ux uy + 4 uy,
  uy^2 + u uyy ]

[ 0.5 uxx v + ux vx + 0.5 u vxx + 2 vx,
  uxy v + u vxy + ux vy + uy vx + 2 ux + 2 vy + 1,
  0.5 uyy v + uy vy + 0.5 u vyy + 2 uy ]

[ vx^2 + v vxx,
  2 v vxy + 2 vx vy + 4 vx,
  vy^2 + v vyy + 4 vy + 1 ]
```

The paper also writes these as Gao's `C(p), D(p)` plus correction matrices:

```text
C'(p) = C(p) + [0, 2u, 0; 0, v, u; 0, 0, 2v]
D'(p) = D(p) + [2ux, 2uy, 0; vx, ux+vy, uy; 0, 2vx, 2vy].
```

This operator is called invertible/reversible by Bai because it is constructed
as a square lifted linear operator and its identity case is exactly the 6 x 6
identity. Strictly, a square matrix is not globally invertible for every
possible parameter vector; arbitrary large or degenerate parameters can still
make the determinant vanish. The practical IC-GN2 requirement is local
invertibility of the small incremental operator near identity, and the
implementation must check this numerically, just as affine IC-GN checks the
affine determinant.

The update is

```text
G_new = G_Bai(p) * inv(G_Bai(delta_p)).
```

Bai Eq. (23) prints `B(p) B^{-1}(delta_p) x`; from context and Eq. (20), the
intended object is the full 6 x 6 operator. This document therefore uses
`G_Bai`.

Extract the updated parameters from coordinate rows 2 and 3 in Bai basis:

```text
u_new   = G_new(2, 1)
ux_new  = G_new(2, 2) - 1
uy_new  = G_new(2, 3)
uxx_new = 2 G_new(2, 4)
uxy_new = G_new(2, 5)
uyy_new = 2 G_new(2, 6)

v_new   = G_new(3, 1)
vx_new  = G_new(3, 2)
vy_new  = G_new(3, 3) - 1
vxx_new = 2 G_new(3, 4)
vxy_new = G_new(3, 5)
vyy_new = 2 G_new(3, 6)
```

Again, written indices are 1-based.

### 6.3 Bai vs Gao in update behavior

Both methods:

- use the same physical 12-parameter second-order warp;
- use reference-side IC linearization;
- compose the current warp with an inverse incremental warp;
- use a lifted 6 x 6 matrix to make the inverse/composition computable;
- extract the next 12 parameters from the coordinate rows after composition.

Bai differs by changing the auxiliary lifted rows. The coordinate warp remains
quadratic, but the lifted operator is constructed to preserve second-order
precision more stably under large or locally non-uniform deformation. It adds
only a small fixed amount of arithmetic to build `C'` and `D'`; the matrix size
and solve/inverse cost remain the same as Gao's 6 x 6 update.

After either update, the extracted coordinate rows are strictly second-order in
the chosen lifted representation, so the stored 12-parameter warp remains a
quadratic shape function. This does not mean that the exact composition of two
ordinary quadratic maps has been represented without loss; it means the
composition has been projected through the Gao or Bai lifted second-order
operator.

## 7. C++ implementation form

### 7.1 Preferred operator representation

The scalar 12-parameter representation is convenient for API and result output.
For the IC update, a fixed 6 x 6 matrix operator is better:

```cpp
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat6  = Eigen::Matrix<double, 6, 6>;

Mat6 buildGaoOperator(const Vec12& p);
Mat6 buildBaiOperator(const Vec12& p);
Vec12 extractFromGaoOperator(const Mat6& G);
Vec12 extractFromBaiOperator(const Mat6& G);
```

The update should avoid explicit dynamic allocation and preferably avoid forming
an explicit inverse:

```cpp
// Gnew = Gp * inv(Gd)
// Compute X = Gd^{-T} * Gp^T, then Gnew = X^T.
Eigen::PartialPivLU<Mat6> lu(Gd);
Mat6 Gnew = lu.solve(Gp.transpose()).transpose();
```

`PartialPivLU` is robust for a general 6 x 6 matrix. If tests show the
incremental operator is always well-conditioned near identity, `FullPivLU`
can be used only as a diagnostic fallback because it is more expensive. The
Hessian solve should use fixed-size `LDLT` or `LLT`.

### 7.2 Tensor notation for the warp

For evaluating sample coordinates and incremental displacement, a compact
tensor view is useful:

```text
W(r) = r + t + D r + 0.5 Q(r, r)
```

where

```text
t = [u, v]^T
D = [ux uy; vx vy]

Q_x = [uxx uxy; uxy uyy]
Q_y = [vxx vxy; vxy vyy]
```

Then

```text
0.5 Q_x(r,r) = 0.5 uxx xi^2 + uxy xi eta + 0.5 uyy eta^2.
```

This tensor representation is clearer for evaluation and convergence checks,
while the 6 x 6 lifted matrix is clearer for IC composition. Use both as thin
views over the same `Vec12`; do not maintain duplicate mutable state.

## 8. Coordinates and convention checklist

Traditional-DIC should lock the following convention before implementation:

```text
local coordinate:       xi = x - xc, eta = y - yc
warp output:            absolute local target coordinate W(r)
global target sample:   c + W(r)
identity parameter:     all displacement parameters zero
identity mapping:       W(r; 0) = r
ux, vy:                 displacement gradient entries
mapping Jacobian:       I + displacement gradient
pure quadratic terms:   0.5 uxx xi^2 and 0.5 uyy eta^2
mixed term:             uxy xi eta, no 0.5, no factor 2
IC update:              compose with inverse incremental operator
```

Most implementation bugs in IC-GN2 come from mixing paper parameter order with
project parameter order or from treating `ux` and `vy` as full Jacobian entries.

## 9. Interpolation and gradient flow

Current relevant modules:

- `Image`: integer grayscale storage.
- `BSplineImagePreprocessor`: precomputes coefficients, local polynomial
  blocks, and gradient images.
- `BSplineInterpolator`: provides `value(x, y)` and `gradient(x, y)`.
- `ZNSSDCorrelation`: implements zero-mean/unit-norm vector comparison.
- `ICGNSolver`: contains a working first-order path and a second-order
  placeholder.
- `SubsetInitializer`, `IntegerSearchInitializer`, `SIFTInitializer`,
  seed/reliability modules: provide large-displacement initial estimates.

IC-GN2 precompute once per subset:

```text
reference sample coordinates and local coordinates
reference intensities f_i
reference mean f_bar
reference norm f_s
reference gradient grad_f_i
shape Jacobian dW/dp at each local coordinate
steepest-descent rows J_i
Hessian H
Hessian factorization
```

Each iteration:

```text
warp sample coordinates using p
interpolate deformed intensity g_i(p)
compute g_bar and g_s
compute residual e_i
compute b
solve delta_p
apply Bai or Gao inverse compositional update
check convergence
```

The deformed image gradient is not needed for IC-GN2. It would be needed for
FA-GN/FC-GN style updates.

The current `BSplineInterpolator` API is sufficient for IC-GN2:

- `value()` is needed each iteration for the deformed image;
- `gradient()` is enough for reference-gradient precomputation;
- precomputed local blocks avoid repeated coefficient construction when enabled.

## 10. Convergence criterion

Gao 2015 states the convergence condition as

```text
||delta_p|| = ||p_{n+1} - p_n|| <= 1e-3
```

and scales first-order and second-order terms by subset half-width `M`. The
text extraction confirms the first-order example begins as

```text
||delta_p1|| =
sqrt((delta_u)^2 + (delta_ux M)^2 + (delta_uy M)^2
     + (delta_v)^2 + ...)
```

The full printed second-order norm should be visually rechecked before copying
into code. The likely engineering intent is to measure maximum or RMS
incremental displacement over the subset rather than a raw parameter norm.

Bai 2017 reports comparisons after a fixed three iterations in simulation and
uses correlation coefficient distributions in the rubber experiment. It does
not give a more detailed stopping rule in the extracted text.

Recommended engineering criterion:

```text
max_incremental_displacement =
max_{r_i in subset} || W(r_i; delta_p) - r_i ||

converged if max_incremental_displacement < threshold_pixels
```

This naturally handles mixed units:

- displacement terms are pixels;
- first derivatives are pixel/pixel and become pixels when multiplied by
  `xi`, `eta`;
- second derivatives are pixel/pixel^2 and become pixels when multiplied by
  `xi^2`, `xi eta`, `eta^2`.

For speed, evaluate the max over the subset samples already stored for IC-GN2.
An optional cheaper bound can use the subset radius `M`, but sample evaluation
is simpler and less error-prone.

Also stop on:

```text
iteration >= max_iterations
non-finite p or delta_p
warped sample leaves image bounds
f_s or g_s <= epsilon
ill-conditioned Hessian or incremental operator
ZNSSD/correlation stagnation, if configured later
```

## 11. Initialization strategy

IC-GN2 is still a local optimizer. It improves local non-uniform deformation
modeling; it does not remove the need for a good initial displacement.

The papers:

- Gao focuses on stereo correspondence and mentions high-efficiency strategy:
  IC-GN2 for stereo left-right matching where viewpoint difference behaves like
  non-uniform deformation, IC-GN1 for temporal image sequences where deformation
  is closer to uniform.
- Bai validates large deformation on rubber tension but does not remove the
  need for a correlation initialization stage; it uses subset matching with
  B-spline interpolation and reports correlation coefficient fields.
- Gao cites Pan reliability-guided DIC; this project already has seed selection
  and reliability propagation modules.

Recommended Traditional-DIC staged flow:

```text
integer-pixel search or SIFT seed
-> first-order IC-GN refinement
-> second-order IC-GN2 refinement
-> reliability-guided propagation to neighboring subsets
```

This staged optimization is recommended because:

- integer/SIFT handles large translation;
- affine IC-GN captures translation, rotation, shear, and uniform strain with a
  smaller 6-parameter basin;
- IC-GN2 then refines non-uniform deformation after the solution is already
  close.

For stereo 3D matching with notable view-angle differences, Gao's discussion
supports using IC-GN2 in the final stereo correspondence step. For temporal
sequences with small frame-to-frame deformation, first-order may remain the
default for speed unless residual/correlation indicates underfitting.

## 12. Minimal Traditional-DIC design

The current code already exposes a shape-function interface and separate solver
modules. A minimal, non-invasive implementation should:

1. Complete `SecondOrderShapeFunction` with the project parameter order.
2. Add fixed-size IC-GN2 internals in `ICGNSolver`, rather than replacing the
   first-order path.
3. Keep `Displacement2D` output compatible by returning at least `(u, v)` and
   correlation. If second-order parameters need to be exposed, add an extended
   result type carefully and preserve Python API compatibility.
4. Keep ZNSSD residual code aligned with `ZNSSDCorrelation` normalization.
5. Add tests for:
   - identity warp;
   - pure translation;
   - affine-only equivalence when second-order terms are zero;
   - quadratic warp evaluation and Jacobian rows;
   - Gao/Bai operator identity and small inverse update;
   - synthetic known quadratic deformation.

Possible class layout:

```text
ShapeFunction
├── FirstOrderShapeFunction
└── SecondOrderShapeFunction

ICGNSolver
├── solve_first_order(...)
└── solve_second_order(...)
    ├── precompute_reference_12(...)
    ├── build_hessian_12(...)
    ├── solve_delta_12(...)
    └── inverse_compose_bai_12(...)
```

Do not over-template the public API yet. Internally, use fixed-size aliases:

```cpp
using Vec6  = Eigen::Matrix<double, 6, 1>;
using Mat6  = Eigen::Matrix<double, 6, 6>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
```

Hessian:

```cpp
Mat12 H = Mat12::Zero();
H.noalias() += J.transpose() * J; // per sample, or rank update
Eigen::LDLT<Mat12> ldlt(H);
```

Avoid per-iteration heap allocation by storing sample data in reserved vectors
or fixed-size structs:

```cpp
struct Sample12 {
    int x, y;
    double xi, eta;
    double f;
    Eigen::Matrix<double, 12, 1> sd;
};
```

## 13. Gao vs Bai comparison

| Item | Gao 2015 IC-GN2 | Bai 2017 IC-GN2 |
|---|---|---|
| physical shape function | Same 12-parameter second-order displacement warp | Same 12-parameter second-order displacement warp |
| parameter order in paper | `[u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]` | Same |
| inverse operator | 6 x 6 lifted operator `G(p)` using `[x^2, xy, y^2, x, y, 1]` | 6 x 6 lifted operator `G'(p)` using `[1, x, y, x^2, xy, y^2]` |
| composition | `G_new = G(p) inv(G(delta))` | `G_new = G'(p) inv(G'(delta))` |
| theoretical reversibility | Square lifted matrix; numerical invertibility still depends on matrix condition | Locally invertible near identity by construction; Bai calls it invertible/reversible, but implementation still checks numerically |
| second-order truncation | Expands `x'^2`, `x'y'`, `y'^2` and truncates higher-order terms | Uses auxiliary displacement functions and conjugate functions to build corrected lifted rows |
| known weakness | Bai argues `x'y'` may lose second-order precision for large/local non-uniform deformation | Designed to keep stable `o(x^2 + y^2)` second-order precision |
| computation | Build 6 x 6, solve/invert 6 x 6 | Same matrix size; slightly different arithmetic for `C'`, `D'` |
| numerical stability | Good in Gao's 3D-DIC tests; weaker in Bai's large/non-uniform tests | Better in Bai's simulations and 27% rubber tension experiment |
| large deformation ability | Better than first-order, but Bai reports possible loss of precision | Recommended by Bai for large and local non-uniform deformation |
| implementation complexity | Moderate; formulas `S1..S18` are verbose | Similar; `C'`, `D'` formulas are cleaner in block form |
| compatibility with current project | Compatible | Compatible and recommended |

Recommendation:

Traditional-DIC should adopt Bai 2017 as the formal IC-GN2 update operator,
because it preserves the same user-facing 12-parameter warp while addressing a
documented weakness in Gao's lifted operator. Gao should be kept as an optional
baseline in tests or an internal strategy enum during validation:

```text
SecondOrderUpdateOperator::Gao2015
SecondOrderUpdateOperator::Bai2017
```

After validation, Bai can be the default. Gao is still useful for reproducing
the older IC-GN2 paper and for A/B comparison with literature results.

## 14. Pseudocode

### 14.1 Main IC-GN2 loop

```cpp
SecondOrderResult solve_icgn2(reference, deformed, center, initial)
{
    using Vec12 = Eigen::Matrix<double, 12, 1>;
    using Mat12 = Eigen::Matrix<double, 12, 12>;

    std::vector<Sample12> samples;
    samples.reserve(maxSubsetSamples);

    // Precompute reference subset.
    for (int eta = -M; eta <= M; ++eta) {
        for (int xi = -M; xi <= M; ++xi) {
            if (!inside_subset_mask(xi, eta)) continue;

            int x = center.x + xi;
            int y = center.y + eta;

            Sample12 s;
            s.x = x;
            s.y = y;
            s.xi = xi;
            s.eta = eta;
            s.f = reference.at(x, y);

            Eigen::Vector2d grad = reference_interpolator.gradient(x, y);
            s.sd = steepest_descent_12(grad.x(), grad.y(), xi, eta);
            samples.push_back(s);
        }
    }

    double f_mean = mean(samples.f);
    double f_norm = sqrt(sum((s.f - f_mean)^2));
    if (f_norm <= eps) return numerical_failure;

    Mat12 H = Mat12::Zero();
    for (const Sample12& s : samples) {
        H.noalias() += s.sd * s.sd.transpose();
    }
    H *= 2.0 / (f_norm * f_norm);

    Eigen::LDLT<Mat12> ldlt(H);
    if (ldlt.info() != Success || !ldlt.isPositive()) {
        return numerical_failure;
    }

    Vec12 p = Vec12::Zero();
    p(0) = initial.u;
    p(1) = initial.v;

    for (int iter = 0; iter < max_iterations; ++iter) {
        double g_values[MAX_SAMPLES]; // or reserved vector
        double g_mean = 0.0;

        for (int i = 0; i < samples.size(); ++i) {
            Eigen::Vector2d local_w = warp_quadratic(samples[i].xi, samples[i].eta, p);
            double gx = center.x + local_w.x();
            double gy = center.y + local_w.y();
            if (!in_bounds(gx, gy, deformed)) return invalid_input;

            g_values[i] = deformed_interpolator.value(gx, gy);
            g_mean += g_values[i];
        }
        g_mean /= samples.size();

        double g_norm = sqrt(sum_i((g_values[i] - g_mean)^2));
        if (g_norm <= eps) return numerical_failure;

        Vec12 gradient = Vec12::Zero();
        double znssd = 0.0;
        for (int i = 0; i < samples.size(); ++i) {
            double e =
                (samples[i].f - f_mean) / f_norm -
                (g_values[i] - g_mean) / g_norm;

            gradient.noalias() += e * samples[i].sd;
            znssd += e * e;
        }
        gradient *= 2.0 / f_norm;

        Vec12 delta = -ldlt.solve(gradient);
        if (!delta.allFinite()) return numerical_failure;

        p = inverse_compose_bai(p, delta);
        if (!p.allFinite()) return numerical_failure;

        if (max_incremental_displacement(delta, samples) < convergence_threshold) {
            return success(p, znssd);
        }
    }

    return not_converged(p, last_znssd);
}
```

### 14.2 Steepest descent row

```cpp
Vec12 steepest_descent_12(double fx, double fy, double xi, double eta)
{
    Vec12 j;
    const double x2 = xi * xi;
    const double xy = xi * eta;
    const double y2 = eta * eta;

    j << fx,
         fy,
         fx * xi,
         fx * eta,
         fy * xi,
         fy * eta,
         0.5 * fx * x2,
         fx * xy,
         0.5 * fx * y2,
         0.5 * fy * x2,
         fy * xy,
         0.5 * fy * y2;
    return j;
}
```

### 14.3 Bai inverse composition

Project order:

```text
p = [u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy]
```

```cpp
Mat6 build_bai_operator(const Vec12& p)
{
    const double u   = p(0);
    const double v   = p(1);
    const double ux  = p(2);
    const double uy  = p(3);
    const double vx  = p(4);
    const double vy  = p(5);
    const double uxx = p(6);
    const double uxy = p(7);
    const double uyy = p(8);
    const double vxx = p(9);
    const double vxy = p(10);
    const double vyy = p(11);

    Mat6 G = Mat6::Zero();

    // Basis: [1, x, y, x^2, xy, y^2]^T
    G.row(0) << 1, 0,      0,      0,         0,       0;
    G.row(1) << u, 1 + ux, uy,     0.5 * uxx, uxy,     0.5 * uyy;
    G.row(2) << v, vx,     1 + vy, 0.5 * vxx, vxy,     0.5 * vyy;

    G.row(3) << u*u,
                2*u*ux + 4*u,
                2*u*uy,
                ux*ux + u*uxx + 4*ux + 1,
                2*u*uxy + 2*ux*uy + 4*uy,
                uy*uy + u*uyy;

    G.row(4) << u*v,
                ux*v + u*vx + 2*v,
                uy*v + u*vy + 2*u,
                0.5*uxx*v + ux*vx + 0.5*u*vxx + 2*vx,
                uxy*v + u*vxy + ux*vy + uy*vx + 2*ux + 2*vy + 1,
                0.5*uyy*v + uy*vy + 0.5*u*vyy + 2*uy;

    G.row(5) << v*v,
                2*v*vx,
                2*v*vy + 4*v,
                vx*vx + v*vxx,
                2*v*vxy + 2*vx*vy + 4*vx,
                vy*vy + v*vyy + 4*vy + 1;

    return G;
}

Vec12 extract_bai_parameters(const Mat6& G)
{
    Vec12 p;
    p << G(1,0),             // u
         G(2,0),             // v
         G(1,1) - 1.0,       // ux
         G(1,2),             // uy
         G(2,1),             // vx
         G(2,2) - 1.0,       // vy
         2.0 * G(1,3),       // uxx
         G(1,4),             // uxy
         2.0 * G(1,5),       // uyy
         2.0 * G(2,3),       // vxx
         G(2,4),             // vxy
         2.0 * G(2,5);       // vyy
    return p;
}

Vec12 inverse_compose_bai(const Vec12& p, const Vec12& delta)
{
    Mat6 Gp = build_bai_operator(p);
    Mat6 Gd = build_bai_operator(delta);

    Eigen::PartialPivLU<Mat6> lu(Gd);
    if (!lu.isInvertible()) {
        return Vec12::Constant(std::numeric_limits<double>::quiet_NaN());
    }

    Mat6 Gnew = lu.solve(Gp.transpose()).transpose();
    return extract_bai_parameters(Gnew);
}
```

`PartialPivLU::isInvertible()` availability depends on Eigen version. If it is
not available, check `lu.matrixLU().diagonal()` and/or use `FullPivLU` for a
diagnostic path. The implementation should not silently accept a singular or
ill-conditioned incremental operator.

### 14.4 Gao inverse composition baseline

For validation, implement Gao behind a strategy flag:

```cpp
Mat6 build_gao_operator(const Vec12& p_project_order);
Vec12 extract_gao_parameters(const Mat6& Gnew);

Vec12 inverse_compose_gao(const Vec12& p, const Vec12& delta)
{
    Mat6 Gp = build_gao_operator(p);
    Mat6 Gd = build_gao_operator(delta);
    Eigen::PartialPivLU<Mat6> lu(Gd);
    Mat6 Gnew = lu.solve(Gp.transpose()).transpose();
    return extract_gao_parameters(Gnew);
}
```

Use basis `[x^2, xy, y^2, x, y, 1]^T` and the `S1..S18` formulas in Section 5.

## 15. Open points before implementation

The following details should be explicitly validated before code changes:

- Gao's full printed second-order convergence norm should be rechecked from the
  PDF image if exact reproduction is required. The recommended max incremental
  displacement criterion is safer for engineering use.
- Bai Eq. (23) appears to use `B(p) B^{-1}(delta_p)` where surrounding formulas
  imply the full 6 x 6 `G'(p) G'^{-1}(delta_p)`. Treat this as a paper notation
  issue unless further source material says otherwise.
- Neither paper gives all low-level numerical safeguards for singular lifted
  operators, boundary handling, or weighted/masked subsets. These are project
  engineering choices and should be documented in tests.
- The project currently returns only `Displacement2D`. A public API design is
  needed if callers should access all second-order parameters.
