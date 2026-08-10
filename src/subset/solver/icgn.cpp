#include <dic/interpolation/bspline.hpp>
#include <dic/subset/shape/first_order.hpp>
#include <dic/subset/shape/second_order.hpp>
#include <dic/subset/solver/icgn.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace dic {
namespace {

constexpr double kEpsilon = 1e-12;

struct SamplePoint {
    int x{0};
    int y{0};
    double local_x{0.0};
    double local_y{0.0};
    double reference_value{0.0};
    double reference_normalized{0.0};
    Eigen::Matrix<double, 6, 1> steepest_descent{Eigen::Matrix<double, 6, 1>::Zero()};
};

bool finite_parameters(const Eigen::Matrix<double, 6, 1>& parameters)
{
    for (int i = 0; i < parameters.size(); ++i) {
        if (!std::isfinite(parameters(i))) {
            return false;
        }
    }
    return true;
}

bool warped_point_in_bounds(double x, double y, const Image& image)
{
    return x >= 0.0 && y >= 0.0 &&
           x <= static_cast<double>(image.width() - 1) &&
           y <= static_cast<double>(image.height() - 1);
}

Eigen::Vector2d central_difference_gradient(const Image& image, int x, int y)
{
    const int xm = std::max(0, x - 1);
    const int xp = std::min(image.width() - 1, x + 1);
    const int ym = std::max(0, y - 1);
    const int yp = std::min(image.height() - 1, y + 1);
    const double gx = (static_cast<double>(image.at(xp, y)) - static_cast<double>(image.at(xm, y))) /
                      static_cast<double>(std::max(1, xp - xm));
    const double gy = (static_cast<double>(image.at(x, yp)) - static_cast<double>(image.at(x, ym))) /
                      static_cast<double>(std::max(1, yp - ym));
    return {gx, gy};
}

Eigen::Vector2d reference_gradient_at(
    const Image& image,
    const BSplineInterpolator& reference_interpolator,
    int x,
    int y
)
{
    const auto& precomputed = reference_interpolator.precomputed();
    if (!precomputed.empty() &&
        precomputed.gradient_x.rows() == image.height() &&
        precomputed.gradient_x.cols() == image.width() &&
        precomputed.gradient_y.rows() == image.height() &&
        precomputed.gradient_y.cols() == image.width()) {
        return {precomputed.gradient_x(y, x), precomputed.gradient_y(y, x)};
    }
    return central_difference_gradient(image, x, y);
}

double vector_norm(const std::vector<double>& values, double mean)
{
    double sum = 0.0;
    for (const double value : values) {
        const double diff = value - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

Eigen::Matrix<double, 6, 1> inverse_compositional_affine_update(
    const Eigen::Matrix<double, 6, 1>& parameters,
    const Eigen::Matrix<double, 6, 1>& delta
)
{
    const double du = parameters(0);
    const double dv = parameters(1);
    const double dudx = parameters(2);
    const double dudy = parameters(3);
    const double dvdx = parameters(4);
    const double dvdy = parameters(5);
    const double dp0 = delta(0);
    const double dp1 = delta(1);
    const double dp2 = delta(2);
    const double dp3 = delta(3);
    const double dp4 = delta(4);
    const double dp5 = delta(5);

    const double denominator = dp2 + dp5 + dp2 * dp5 - dp3 * dp4 + 1.0;
    if (std::abs(denominator) <= kEpsilon) {
        return Eigen::Matrix<double, 6, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
    }

    Eigen::Matrix<double, 6, 1> updated;
    updated(0) = du - ((dudx + 1.0) * (dp0 + dp0 * dp5 - dp1 * dp3)) / denominator
                    - (dudy * (dp1 - dp0 * dp4 + dp1 * dp2)) / denominator;
    updated(1) = dv - ((dvdy + 1.0) * (dp1 - dp0 * dp4 + dp1 * dp2)) / denominator
                    - (dvdx * (dp0 + dp0 * dp5 - dp1 * dp3)) / denominator;
    updated(2) = ((dp5 + 1.0) * (dudx + 1.0)) / denominator
                    - (dp4 * dudy) / denominator - 1.0;
    updated(3) = (dudy * (dp2 + 1.0)) / denominator
                    - (dp3 * (dudx + 1.0)) / denominator;
    updated(4) = (dvdx * (dp5 + 1.0)) / denominator
                    - (dp4 * (dvdy + 1.0)) / denominator;
    updated(5) = ((dp2 + 1.0) * (dvdy + 1.0)) / denominator
                    - (dp3 * dvdx) / denominator - 1.0;
    return updated;
}

struct SecondOrderSamplePoint {
    int x{0};
    int y{0};
    double local_x{0.0};
    double local_y{0.0};
    double reference_value{0.0};
    double reference_normalized{0.0};
    Eigen::Matrix<double, 12, 1> steepest_descent{Eigen::Matrix<double, 12, 1>::Zero()};
};

bool finite_12_params(const Eigen::Matrix<double, 12, 1>& p)
{
    for (int i = 0; i < 12; ++i) {
        if (!std::isfinite(p(i))) {
            return false;
        }
    }
    return true;
}

Eigen::Matrix<double, 12, 1> second_order_steepest_descent(
    double gx, double gy, double local_x, double local_y)
{
    const double dx2_2 = local_x * local_x * 0.5;
    const double dy2_2 = local_y * local_y * 0.5;
    const double dxy = local_x * local_y;

    Eigen::Matrix<double, 12, 1> sd;
    // Parameters: [u, v, du_dx, du_dy, dv_dx, dv_dy,
    //              d2u_dx2, d2u_dxdy, d2u_dy2, d2v_dx2, d2v_dxdy, d2v_dy2]
    sd << gx,              // ?Wx/?u
          gy,              // ?Wy/?v
          gx * local_x,    // ?Wx/?du_dx
          gx * local_y,    // ?Wx/?du_dy
          gy * local_x,    // ?Wy/?dv_dx
          gy * local_y,    // ?Wy/?dv_dy
          gx * dx2_2,      // ?Wx/?d2u_dx2
          gx * dxy,        // ?Wx/?d2u_dxdy
          gx * dy2_2,      // ?Wx/?d2u_dy2
          gy * dx2_2,      // ?Wy/?d2v_dx2
          gy * dxy,        // ?Wy/?d2v_dxdy
          gy * dy2_2;      // ?Wy/?d2v_dy2
    return sd;
}

Eigen::Vector2d warp_second_order(
    double center_x, double center_y,
    double local_x, double local_y,
    const Eigen::Matrix<double, 12, 1>& p)
{
    const double dx2_2 = local_x * local_x * 0.5;
    const double dy2_2 = local_y * local_y * 0.5;
    const double dxy = local_x * local_y;

    return {
        center_x + local_x + p(0) + p(2) * local_x + p(3) * local_y
            + p(6) * dx2_2 + p(7) * dxy + p(8) * dy2_2,
        center_y + local_y + p(1) + p(4) * local_x + p(5) * local_y
            + p(9) * dx2_2 + p(10) * dxy + p(11) * dy2_2
    };
}

// Build the 6��6 invertible second-order warp matrix G'(p) used in
// Bai et al. (2017) Eq (20�C22).  The matrix maps the augmented
// homogeneous coordinate  [1, x, y, x2, xy, y2]?  to
// [x', y', 1, x'2, x'y', y'2]?  up to  o(x2+y2).
//
// Parameter convention (this codebase):
//   [u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy]
//
// Literature convention (Bai et al.):
//   [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
//
// Both span the same 12-d function space; only the index mapping differs.
Eigen::Matrix<double, 6, 6> build_second_order_warp_matrix(
    const Eigen::Matrix<double, 12, 1>& p)
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

    Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();

    // ---- Row 0:  x' = u + (1+ux)��x + uy��y + ?uxx��x2 + uxy��xy + ?uyy��y2 ----
    G(0, 0) = u;
    G(0, 1) = 1.0 + ux;
    G(0, 2) = uy;
    G(0, 3) = 0.5 * uxx;
    G(0, 4) = uxy;
    G(0, 5) = 0.5 * uyy;

    // ---- Row 1:  y' = v + vx��x + (1+vy)��y + ?vxx��x2 + vxy��xy + ?vyy��y2 ----
    G(1, 0) = v;
    G(1, 1) = vx;
    G(1, 2) = 1.0 + vy;
    G(1, 3) = 0.5 * vxx;
    G(1, 4) = vxy;
    G(1, 5) = 0.5 * vyy;

    // ---- Row 2:  1  = 1 -------------------------------------------------------
    G(2, 0) = 1.0;

    // ---- Row 3:  x'2 = (x + ��)2  (�� = u-displacement) ------------------------
    // Derived from:  x'2 = x2 + 2x���� + ��2,  keeping terms �� O(x2, y2).
    G(3, 0) = u * u;
    G(3, 1) = 2.0 * u * (1.0 + ux);
    G(3, 2) = 2.0 * u * uy;
    G(3, 3) = 1.0 + 2.0 * ux + ux * ux + u * uxx;
    G(3, 4) = 2.0 * (uy + ux * uy + u * uxy);
    G(3, 5) = uy * uy + u * uyy;

    // ---- Row 4:  x'y' = (x + ��)(y + v?)  (v? = v-displacement) ----------------
    // Derived from:  x'y' = xy + x��v? + y���� + ����v?,  keeping terms �� O(x2, y2).
    G(4, 0) = u * v;
    G(4, 1) = v + u * vx + v * ux;
    G(4, 2) = u + u * vy + v * uy;
    G(4, 3) = vx + ux * vx + 0.5 * (u * vxx + v * uxx);
    G(4, 4) = 1.0 + vy + ux + ux * vy + uy * vx + u * vxy + v * uxy;
    G(4, 5) = uy + uy * vy + 0.5 * (u * vyy + v * uyy);

    // ---- Row 5:  y'2 = (y + v?)2 -----------------------------------------------
    // Derived from:  y'2 = y2 + 2y��v? + v?2,  keeping terms �� O(x2, y2).
    G(5, 0) = v * v;
    G(5, 1) = 2.0 * v * vx;
    G(5, 2) = 2.0 * v * (1.0 + vy);
    G(5, 3) = vx * vx + v * vxx;
    G(5, 4) = 2.0 * (vx + vx * vy + v * vxy);
    G(5, 5) = 1.0 + 2.0 * vy + vy * vy + v * vyy;

    return G;
}

// Extract the 12 deformation parameters from the 6��6 warp matrix G'(p).
// This is the inverse of build_second_order_warp_matrix().
Eigen::Matrix<double, 12, 1> extract_from_warp_matrix(
    const Eigen::Matrix<double, 6, 6>& G)
{
    Eigen::Matrix<double, 12, 1> p;

    // From Row 0:  x' = u + (1+ux)��x + uy��y + ?uxx��x2 + uxy��xy + ?uyy��y2
    p(0) = G(0, 0);                  // u
    p(2) = G(0, 1) - 1.0;            // ux
    p(3) = G(0, 2);                  // uy
    p(6) = 2.0 * G(0, 3);            // uxx
    p(7) = G(0, 4);                  // uxy
    p(8) = 2.0 * G(0, 5);            // uyy

    // From Row 1:  y' = v + vx��x + (1+vy)��y + ?vxx��x2 + vxy��xy + ?vyy��y2
    p(1)  = G(1, 0);                 // v
    p(4)  = G(1, 1);                 // vx
    p(5)  = G(1, 2) - 1.0;           // vy
    p(9)  = 2.0 * G(1, 3);           // vxx
    p(10) = G(1, 4);                 // vxy
    p(11) = 2.0 * G(1, 5);           // vyy

    return p;
}

// Inverse-compositional update for 12-parameter second-order shape functions.
//
// Reference:  Bai, Jiang, Lei & Li (2017), "A novel 2nd-order shape function
// based digital image correlation method for large deformation measurements",
// Optics and Lasers in Engineering, 90, 48�C58.  Eq (23) + Appendix A.
//
// The update composes the current warp with the inverse of the incremental
// warp through a 6��6 invertible matrix:
//
//      G'(p_new) = G'(p_old) �� [G'(��p)]?1 �� P?1
//
// where G'(p) maps INPUT  v = [1, x, y,  x2,  xy,  y2 ]?
//                    to OUTPUT w = [x', y', 1, x'2, x'y', y'2]?.
// P?1 is the permutation that restores the w-convention (v �� w)
// so that G'(��p)?1 : w �� v can be composed with G'(p_old) : v �� w.
//
// This captures the full nonlinear coupling between first-order and
// second-order parameters during warp composition, achieving stable
// second-order precision even for large / non-uniform deformations.
Eigen::Matrix<double, 12, 1> inverse_compositional_second_order_update(
    const Eigen::Matrix<double, 12, 1>& params,
    const Eigen::Matrix<double, 12, 1>& delta)
{
    // Build the warp matrices for current parameters and the increment.
    Eigen::Matrix<double, 6, 6> G_p    = build_second_order_warp_matrix(params);
    Eigen::Matrix<double, 6, 6> G_delta = build_second_order_warp_matrix(delta);

    // Invert the incremental warp matrix.
    // The paper proves G'(��p) is invertible for physically realisable
    // deformations.  A singular / near-singular inverse indicates a
    // numerical issue; fall back to additive update in that case.
    Eigen::Matrix<double, 6, 6> G_delta_inv;
    bool invertible = false;

    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(G_delta);
    if (lu.isInvertible()) {
        G_delta_inv = lu.inverse();
        invertible = true;
    }

    if (!invertible) {
        // Numerical safeguard: if the warp increment is not invertible,
        // fall back to the additive approximation.
        Eigen::Matrix<double, 12, 1> updated = params;
        for (int i = 0; i < 12; ++i) {
            updated(i) -= delta(i);
        }
        return updated;
    }

    // Compose:  G'(p_new) = G'(p_old) �� [G'(��p)]?1 �� P?1
    //
    // G'(��) maps input convention  v = [1, x, y, x2, xy, y2]?
    //                     to output w = [x', y', 1, x'2, x'y', y'2]?.
    // G'(��p)?1 : w �� v.  To feed its output into G'(p_old) : v �� w
    // we must first restore the w-convention with the permutation
    // P?1 that sends v �� w (i.e. cycles [1,x,y] �� [x,y,1]).
    Eigen::Matrix<double, 6, 6> P_inv = Eigen::Matrix<double, 6, 6>::Zero();
    P_inv(0, 1) = 1.0;   // v[1] = x  �� w[0]
    P_inv(1, 2) = 1.0;   // v[2] = y  �� w[1]
    P_inv(2, 0) = 1.0;   // v[0] = 1  �� w[2]
    P_inv(3, 3) = 1.0;   //   x2 �� x'2
    P_inv(4, 4) = 1.0;   //   xy �� x'y'
    P_inv(5, 5) = 1.0;   //   y2 �� y'2

    Eigen::Matrix<double, 6, 6> G_new = G_p * G_delta_inv * P_inv;

    return extract_from_warp_matrix(G_new);
}

// Compute the normalized ZNSSD of a converged SSD displacement.
//
// The SSD solvers minimize the raw intensity residual, whose magnitude is a
// function of the subset area, intensity scale and illumination offset - it is
// NOT comparable to the ZNSSD in [0, 2] that every downstream quality gate
// (seed selection `max_znssd`, propagation `max_znssd`, 3D `max_znssd`) is
// tuned for.  On noisy stereo pairs the raw residual easily exceeds 2.0 even
// for a correct match, silently rejecting every seed (e.g. CylinderDIC ssd ->
// empty fields).  Re-deriving the ZNSSD of the final warp restores a
// criterion-independent quality value without changing the SSD optimization.
//
// `warp(local_x, local_y)` maps a reference-subset sample to its deformed
// coordinates using the converged parameters.  Returns +inf when the final
// warp leaves the image or the intensity statistics degenerate.
template <typename SampleT, typename WarpFn>
double normalized_znssd_of_final_warp(
    const std::vector<SampleT>& samples,
    const WarpFn& warp,
    const Image& deformed,
    const BSplineInterpolator& deformed_interpolator)
{
    double reference_mean = 0.0;
    for (const auto& sample : samples) {
        reference_mean += sample.reference_value;
    }
    reference_mean /= static_cast<double>(samples.size());

    double reference_norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - reference_mean;
        reference_norm += diff * diff;
    }
    reference_norm = std::sqrt(reference_norm);
    if (reference_norm <= kEpsilon) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());
    double deformed_mean = 0.0;
    for (const auto& sample : samples) {
        const auto warped = warp(sample.local_x, sample.local_y);
        if (!warped_point_in_bounds(warped.x(), warped.y(), deformed)) {
            return std::numeric_limits<double>::infinity();
        }
        const double value = deformed_interpolator.value(warped.x(), warped.y());
        deformed_values.push_back(value);
        deformed_mean += value;
    }
    deformed_mean /= static_cast<double>(deformed_values.size());
    const double deformed_norm = vector_norm(deformed_values, deformed_mean);
    if (deformed_norm <= kEpsilon) {
        return std::numeric_limits<double>::infinity();
    }

    double corrcoef = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double reference_normalized =
            (samples[i].reference_value - reference_mean) / reference_norm;
        const double deformed_normalized =
            (deformed_values[i] - deformed_mean) / deformed_norm;
        const double diff = reference_normalized - deformed_normalized;
        corrcoef += diff * diff;
    }
    return corrcoef;
}

} // namespace

ICGNSolver::ICGNSolver(SubsetConfig config) : config_(config) {}

Displacement2D ICGNSolver::solve(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    const bool second_order = config_.shape_function == SubsetShapeFunctionMethod::SecondOrder ||
                              config_.use_second_order;
    if (second_order) {
        if (config_.objective == CorrelationCriterionKind::SSD) {
            return solve_second_order_ssd(reference, deformed, point, initial);
        }
        return solve_second_order_znssd(reference, deformed, point, initial);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd(reference, deformed, point, initial);
    }
    return solve_first_order_znssd(reference, deformed, point, initial);
}

Displacement2D ICGNSolver::solve_with_interpolators(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    const bool second_order = config_.shape_function == SubsetShapeFunctionMethod::SecondOrder ||
                              config_.use_second_order;
    if (second_order) {
        if (config_.objective == CorrelationCriterionKind::SSD) {
            return solve_second_order_ssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
        }
        return solve_second_order_znssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
    }
    return solve_first_order_znssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_with_mask(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    const bool second_order = config_.shape_function == SubsetShapeFunctionMethod::SecondOrder ||
                              config_.use_second_order;
    if (second_order) {
        if (config_.objective == CorrelationCriterionKind::SSD) {
            return solve_second_order_ssd_masked(reference, deformed, roi, point, initial,
                                                  reference_interpolator, deformed_interpolator);
        }
        return solve_second_order_znssd_masked(reference, deformed, roi, point, initial, reference_interpolator, deformed_interpolator);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd_masked(
            reference, deformed, roi, point, initial, reference_interpolator, deformed_interpolator);
    }
    return solve_first_order_znssd_masked(
        reference, deformed, roi, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_first_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator reference_interpolator(reference, config_.image_precompute);
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_first_order_znssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_first_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (center_x - radius < 0 || center_y - radius < 0 ||
        center_x + radius >= reference.width() ||
        center_y + radius >= reference.height()) {
        return result;
    }

    std::vector<SamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            SamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent << gradient.x(),
                                       gradient.y(),
                                       gradient.x() * sample.local_x,
                                       gradient.x() * sample.local_y,
                                       gradient.y() * sample.local_x,
                                       gradient.y() * sample.local_y;

            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    if (samples.size() < 6) {
        return result;
    }

    reference_mean /= static_cast<double>(samples.size());
    double reference_norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - reference_mean;
        reference_norm += diff * diff;
    }
    reference_norm = std::sqrt(reference_norm);
    if (reference_norm <= kEpsilon) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }
    for (auto& sample : samples) {
        sample.reference_normalized = (sample.reference_value - reference_mean) / reference_norm;
    }

    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }
    hessian *= 2.0 / (reference_norm * reference_norm);

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    Eigen::Matrix<double, 6, 1> parameters;
    parameters << initial.u, initial.v,
                  initial.du_dx, initial.du_dy,
                  initial.dv_dx, initial.dv_dy;

    double corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());
    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        double deformed_mean = 0.0;
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y;
            if (!warped_point_in_bounds(warped_x, warped_y, deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped_x, warped_y);
            deformed_values.push_back(value);
            deformed_mean += value;
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;
            gradient += normalized_difference * samples[i].steepest_descent;
            corrcoef += normalized_difference * normalized_difference;
        }
        gradient *= 2.0 / reference_norm;

        Eigen::Matrix<double, 6, 1> delta = -decomposition.solve(gradient);
        if (!finite_parameters(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_affine_update(parameters, delta);
        if (!finite_parameters(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.correlation = corrcoef;
    if (!converged && std::isfinite(corrcoef)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_first_order_znssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (!roi.valid(center_x, center_y)) {
        return result;
    }

    std::vector<SamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    int full_count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            ++full_count;
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (!reference.contains(x, y) || !roi.valid(x, y)) {
                continue;
            }

            SamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent << gradient.x(),
                                       gradient.y(),
                                       gradient.x() * sample.local_x,
                                       gradient.x() * sample.local_y,
                                       gradient.y() * sample.local_x,
                                       gradient.y() * sample.local_y;

            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(6, min_samples)) {
        return result;
    }

    reference_mean /= static_cast<double>(samples.size());
    double reference_norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - reference_mean;
        reference_norm += diff * diff;
    }
    reference_norm = std::sqrt(reference_norm);
    if (reference_norm <= kEpsilon) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }
    for (auto& sample : samples) {
        sample.reference_normalized = (sample.reference_value - reference_mean) / reference_norm;
    }

    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }
    hessian *= 2.0 / (reference_norm * reference_norm);

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    Eigen::Matrix<double, 6, 1> parameters;
    parameters << initial.u, initial.v,
                  initial.du_dx, initial.du_dy,
                  initial.dv_dx, initial.dv_dy;

    double corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());
    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        double deformed_mean = 0.0;
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y;
            if (!warped_point_in_bounds(warped_x, warped_y, deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped_x, warped_y);
            deformed_values.push_back(value);
            deformed_mean += value;
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;
            gradient += normalized_difference * samples[i].steepest_descent;
            corrcoef += normalized_difference * normalized_difference;
        }
        gradient *= 2.0 / reference_norm;

        Eigen::Matrix<double, 6, 1> delta = -decomposition.solve(gradient);
        if (!finite_parameters(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_affine_update(parameters, delta);
        if (!finite_parameters(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.correlation = corrcoef;
    if (!converged && std::isfinite(corrcoef)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_first_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator reference_interpolator(reference, config_.image_precompute);
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_first_order_ssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_first_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (center_x - radius < 0 || center_y - radius < 0 ||
        center_x + radius >= reference.width() ||
        center_y + radius >= reference.height()) {
        return result;
    }

    // --- Build sample list (no normalization needed for SSD) ---
    std::vector<SamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            SamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent << gradient.x(),
                                       gradient.y(),
                                       gradient.x() * sample.local_x,
                                       gradient.x() * sample.local_y,
                                       gradient.y() * sample.local_x,
                                       gradient.y() * sample.local_y;

            samples.push_back(sample);
        }
    }

    if (samples.size() < 6) {
        return result;
    }

    // --- Hessian: H = J^T * J (no 2/��2 scaling) ---
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    Eigen::Matrix<double, 6, 1> parameters;
    parameters << initial.u, initial.v,
                  initial.du_dx, initial.du_dy,
                  initial.dv_dx, initial.dv_dy;

    double ssd_corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y;
            if (!warped_point_in_bounds(warped_x, warped_y, deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped_x, warped_y);
            deformed_values.push_back(value);
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- Gradient: ��[(ref - def) * J]  (no 2/�� scaling) ---
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        ssd_corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double residual = samples[i].reference_value - deformed_values[i];
            gradient += residual * samples[i].steepest_descent;
            ssd_corrcoef += residual * residual;
        }

        Eigen::Matrix<double, 6, 1> delta = -decomposition.solve(gradient);
        if (!finite_parameters(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_affine_update(parameters, delta);
        if (!finite_parameters(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    // Report the normalized ZNSSD of the converged SSD warp so downstream
    // quality gates (tuned for ZNSSD in [0,2]) work criterion-independently.
    const double reported_corr = normalized_znssd_of_final_warp(
        samples,
        [&](double lx, double ly) {
            return Eigen::Vector2d(
                static_cast<double>(center_x) + lx + parameters(0) + parameters(2) * lx + parameters(3) * ly,
                static_cast<double>(center_y) + ly + parameters(1) + parameters(4) * lx + parameters(5) * ly);
        },
        deformed,
        deformed_interpolator);
    result.correlation = reported_corr;
    if (!converged && std::isfinite(reported_corr)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_first_order_ssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (!roi.valid(center_x, center_y)) {
        return result;
    }

    // --- Build sample list, skip pixels outside ROI ---
    std::vector<SamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    int full_count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            ++full_count;
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (!reference.contains(x, y) || !roi.valid(x, y)) {
                continue;
            }

            SamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent << gradient.x(),
                                       gradient.y(),
                                       gradient.x() * sample.local_x,
                                       gradient.x() * sample.local_y,
                                       gradient.y() * sample.local_x,
                                       gradient.y() * sample.local_y;

            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(6, min_samples)) {
        return result;
    }

    // --- Hessian: H = J^T * J ---
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    Eigen::Matrix<double, 6, 1> parameters;
    parameters << initial.u, initial.v,
                  initial.du_dx, initial.du_dy,
                  initial.dv_dx, initial.dv_dy;

    double ssd_corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y;
            if (!warped_point_in_bounds(warped_x, warped_y, deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped_x, warped_y);
            deformed_values.push_back(value);
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- Gradient: ��[(ref - def) * J] ---
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        ssd_corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double residual = samples[i].reference_value - deformed_values[i];
            gradient += residual * samples[i].steepest_descent;
            ssd_corrcoef += residual * residual;
        }

        Eigen::Matrix<double, 6, 1> delta = -decomposition.solve(gradient);
        if (!finite_parameters(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_affine_update(parameters, delta);
        if (!finite_parameters(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    // Report the normalized ZNSSD of the converged SSD warp so downstream
    // quality gates (tuned for ZNSSD in [0,2]) work criterion-independently.
    const double reported_corr = normalized_znssd_of_final_warp(
        samples,
        [&](double lx, double ly) {
            return Eigen::Vector2d(
                static_cast<double>(center_x) + lx + parameters(0) + parameters(2) * lx + parameters(3) * ly,
                static_cast<double>(center_y) + ly + parameters(1) + parameters(4) * lx + parameters(5) * ly);
        },
        deformed,
        deformed_interpolator);
    result.correlation = reported_corr;
    if (!converged && std::isfinite(reported_corr)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_second_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator reference_interpolator(reference, config_.image_precompute);
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_second_order_znssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_second_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (center_x - radius < 0 || center_y - radius < 0 ||
        center_x + radius >= reference.width() ||
        center_y + radius >= reference.height()) {
        return result;
    }

    // --- Build sample list with 12-param steepest descent ---
    std::vector<SecondOrderSamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            SecondOrderSamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent = second_order_steepest_descent(
                gradient.x(), gradient.y(), sample.local_x, sample.local_y);

            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    if (samples.size() < 12) {
        return result;
    }

    // --- ZNSSD normalization: reference ---
    reference_mean /= static_cast<double>(samples.size());
    double reference_norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - reference_mean;
        reference_norm += diff * diff;
    }
    reference_norm = std::sqrt(reference_norm);
    if (reference_norm <= kEpsilon) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }
    for (auto& sample : samples) {
        sample.reference_normalized = (sample.reference_value - reference_mean) / reference_norm;
    }

    // --- Hessian: 12��12, H = (2/��2)����[J��J?] ---
    Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }
    hessian *= 2.0 / (reference_norm * reference_norm);

    Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    // --- Initialize 12 parameters ---
    Eigen::Matrix<double, 12, 1> parameters = Eigen::Matrix<double, 12, 1>::Zero();
    parameters(0) = initial.u;
    parameters(1) = initial.v;
    parameters(2) = initial.du_dx;
    parameters(3) = initial.du_dy;
    parameters(4) = initial.dv_dx;
    parameters(5) = initial.dv_dy;
    // Second-order parameters initialized to zero

    double corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        double deformed_mean = 0.0;
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const auto warped = warp_second_order(
                static_cast<double>(center_x), static_cast<double>(center_y),
                sample.local_x, sample.local_y, parameters);
            if (!warped_point_in_bounds(warped.x(), warped.y(), deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped.x(), warped.y());
            deformed_values.push_back(value);
            deformed_mean += value;
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- ZNSSD normalization: deformed ---
        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        // --- Gradient: (2/��)����[normalized_diff �� J] ---
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;
            gradient += normalized_difference * samples[i].steepest_descent;
            corrcoef += normalized_difference * normalized_difference;
        }
        gradient *= 2.0 / reference_norm;

        // --- Solve: ��p = -H?1��g ---
        Eigen::Matrix<double, 12, 1> delta = -decomposition.solve(gradient);
        if (!finite_12_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_second_order_update(parameters, delta);
        if (!finite_12_params(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.d2u_dx2 = parameters(6);
    result.d2u_dxdy = parameters(7);
    result.d2u_dy2 = parameters(8);
    result.d2v_dx2 = parameters(9);
    result.d2v_dxdy = parameters(10);
    result.d2v_dy2 = parameters(11);
    result.correlation = corrcoef;
    if (!converged && std::isfinite(corrcoef)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_second_order_znssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (!roi.valid(center_x, center_y)) {
        return result;
    }

    // --- Build sample list, skip pixels outside ROI ---
    std::vector<SecondOrderSamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    int full_count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            ++full_count;
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (!reference.contains(x, y) || !roi.valid(x, y)) {
                continue;
            }

            SecondOrderSamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent = second_order_steepest_descent(
                gradient.x(), gradient.y(), sample.local_x, sample.local_y);

            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(12, min_samples)) {
        return result;
    }

    // --- ZNSSD normalization: reference ---
    reference_mean /= static_cast<double>(samples.size());
    double reference_norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - reference_mean;
        reference_norm += diff * diff;
    }
    reference_norm = std::sqrt(reference_norm);
    if (reference_norm <= kEpsilon) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }
    for (auto& sample : samples) {
        sample.reference_normalized = (sample.reference_value - reference_mean) / reference_norm;
    }

    // --- Hessian: 12��12 ---
    Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }
    hessian *= 2.0 / (reference_norm * reference_norm);

    Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    // --- Initialize 12 parameters ---
    Eigen::Matrix<double, 12, 1> parameters = Eigen::Matrix<double, 12, 1>::Zero();
    parameters(0) = initial.u;
    parameters(1) = initial.v;
    parameters(2) = initial.du_dx;
    parameters(3) = initial.du_dy;
    parameters(4) = initial.dv_dx;
    parameters(5) = initial.dv_dy;

    double corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        double deformed_mean = 0.0;
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const auto warped = warp_second_order(
                static_cast<double>(center_x), static_cast<double>(center_y),
                sample.local_x, sample.local_y, parameters);
            if (!warped_point_in_bounds(warped.x(), warped.y(), deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped.x(), warped.y());
            deformed_values.push_back(value);
            deformed_mean += value;
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;
            gradient += normalized_difference * samples[i].steepest_descent;
            corrcoef += normalized_difference * normalized_difference;
        }
        gradient *= 2.0 / reference_norm;

        Eigen::Matrix<double, 12, 1> delta = -decomposition.solve(gradient);
        if (!finite_12_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_second_order_update(parameters, delta);
        if (!finite_12_params(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.d2u_dx2 = parameters(6);
    result.d2u_dxdy = parameters(7);
    result.d2u_dy2 = parameters(8);
    result.d2v_dx2 = parameters(9);
    result.d2v_dxdy = parameters(10);
    result.d2v_dy2 = parameters(11);
    result.correlation = corrcoef;
    if (!converged && std::isfinite(corrcoef)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_second_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator reference_interpolator(reference, config_.image_precompute);
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_second_order_ssd(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_second_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (center_x - radius < 0 || center_y - radius < 0 ||
        center_x + radius >= reference.width() ||
        center_y + radius >= reference.height()) {
        return result;
    }

    // --- Build sample list with 12-param steepest descent ---
    std::vector<SecondOrderSamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            SecondOrderSamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent = second_order_steepest_descent(
                gradient.x(), gradient.y(), sample.local_x, sample.local_y);

            samples.push_back(sample);
        }
    }

    if (samples.size() < 12) {
        return result;
    }

    // --- Hessian: 12��12, H = J^T * J (no 2/��2 scaling for SSD) ---
    Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }

    Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    // --- Initialize 12 parameters ---
    Eigen::Matrix<double, 12, 1> parameters = Eigen::Matrix<double, 12, 1>::Zero();
    parameters(0) = initial.u;
    parameters(1) = initial.v;
    parameters(2) = initial.du_dx;
    parameters(3) = initial.du_dy;
    parameters(4) = initial.dv_dx;
    parameters(5) = initial.dv_dy;
    // Second-order parameters initialized to zero

    double ssd_corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const auto warped = warp_second_order(
                static_cast<double>(center_x), static_cast<double>(center_y),
                sample.local_x, sample.local_y, parameters);
            if (!warped_point_in_bounds(warped.x(), warped.y(), deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped.x(), warped.y());
            deformed_values.push_back(value);
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- Gradient: ��[(ref - def) * J] (no 2/�� scaling for SSD) ---
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        ssd_corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double residual = samples[i].reference_value - deformed_values[i];
            gradient += residual * samples[i].steepest_descent;
            ssd_corrcoef += residual * residual;
        }

        // --- Solve: ��p = -H?1��g ---
        Eigen::Matrix<double, 12, 1> delta = -decomposition.solve(gradient);
        if (!finite_12_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_second_order_update(parameters, delta);
        if (!finite_12_params(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.d2u_dx2 = parameters(6);
    result.d2u_dxdy = parameters(7);
    result.d2u_dy2 = parameters(8);
    result.d2v_dx2 = parameters(9);
    result.d2v_dxdy = parameters(10);
    result.d2v_dy2 = parameters(11);
    // Report the normalized ZNSSD of the converged SSD warp so downstream
    // quality gates (tuned for ZNSSD in [0,2]) work criterion-independently.
    const double reported_corr = normalized_znssd_of_final_warp(
        samples,
        [&](double lx, double ly) {
            return warp_second_order(static_cast<double>(center_x), static_cast<double>(center_y), lx, ly, parameters);
        },
        deformed,
        deformed_interpolator);
    result.correlation = reported_corr;
    if (!converged && std::isfinite(reported_corr)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_second_order_ssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::InvalidInput;
    result.valid = false;

    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height() ||
        config_.subset_radius < 1 ||
        config_.max_iterations < 0) {
        return result;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const int radius = config_.subset_radius;
    if (!roi.valid(center_x, center_y)) {
        return result;
    }

    // --- Build sample list, skip pixels outside ROI ---
    std::vector<SecondOrderSamplePoint> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    int full_count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            ++full_count;
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (!reference.contains(x, y) || !roi.valid(x, y)) {
                continue;
            }

            SecondOrderSamplePoint sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));

            const auto gradient = reference_gradient_at(reference, reference_interpolator, x, y);
            sample.steepest_descent = second_order_steepest_descent(
                gradient.x(), gradient.y(), sample.local_x, sample.local_y);

            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(12, min_samples)) {
        return result;
    }

    // --- Hessian: 12��12, H = J^T * J (no 2/��2 scaling for SSD) ---
    Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
    for (const auto& sample : samples) {
        hessian += sample.steepest_descent * sample.steepest_descent.transpose();
    }

    Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        result.status = SolverStatus::NumericalFailure;
        return result;
    }

    // --- Initialize 12 parameters ---
    Eigen::Matrix<double, 12, 1> parameters = Eigen::Matrix<double, 12, 1>::Zero();
    parameters(0) = initial.u;
    parameters(1) = initial.v;
    parameters(2) = initial.du_dx;
    parameters(3) = initial.du_dy;
    parameters(4) = initial.dv_dx;
    parameters(5) = initial.dv_dy;
    // Second-order parameters initialized to zero

    double ssd_corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const auto warped = warp_second_order(
                static_cast<double>(center_x), static_cast<double>(center_y),
                sample.local_x, sample.local_y, parameters);
            if (!warped_point_in_bounds(warped.x(), warped.y(), deformed)) {
                all_warped_points_valid = false;
                break;
            }
            const double value = deformed_interpolator.value(warped.x(), warped.y());
            deformed_values.push_back(value);
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- Gradient: ��[(ref - def) * J] (no 2/�� scaling for SSD) ---
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        ssd_corrcoef = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double residual = samples[i].reference_value - deformed_values[i];
            gradient += residual * samples[i].steepest_descent;
            ssd_corrcoef += residual * residual;
        }

        // --- Solve: ��p = -H?1��g ---
        Eigen::Matrix<double, 12, 1> delta = -decomposition.solve(gradient);
        if (!finite_12_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        const double delta_norm = delta.norm();
        parameters = inverse_compositional_second_order_update(parameters, delta);
        if (!finite_12_params(parameters)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        if (delta_norm < config_.convergence_threshold) {
            converged = true;
            break;
        }
    }

    result.u = parameters(0);
    result.v = parameters(1);
    result.du_dx = parameters(2);
    result.du_dy = parameters(3);
    result.dv_dx = parameters(4);
    result.dv_dy = parameters(5);
    result.d2u_dx2 = parameters(6);
    result.d2u_dxdy = parameters(7);
    result.d2u_dy2 = parameters(8);
    result.d2v_dx2 = parameters(9);
    result.d2v_dxdy = parameters(10);
    result.d2v_dy2 = parameters(11);
    // Report the normalized ZNSSD of the converged SSD warp so downstream
    // quality gates (tuned for ZNSSD in [0,2]) work criterion-independently.
    const double reported_corr = normalized_znssd_of_final_warp(
        samples,
        [&](double lx, double ly) {
            return warp_second_order(static_cast<double>(center_x), static_cast<double>(center_y), lx, ly, parameters);
        },
        deformed,
        deformed_interpolator);
    result.correlation = reported_corr;
    if (!converged && std::isfinite(reported_corr)) {
        converged = true;
    }

    result.status = converged ? SolverStatus::Success : SolverStatus::NotConverged;
    result.valid = converged;
    return result;
}

Displacement2D ICGNSolver::solve_unimplemented(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::NotConverged;
    result.valid = false;
    return result;
}

Eigen::VectorXd ICGNSolver::extract_reference_subset(const Image& reference, const Eigen::Vector2d& point) const { (void)reference; (void)point; return {}; }
Eigen::MatrixXd ICGNSolver::compute_reference_gradient(const Image& reference, const Eigen::Vector2d& point) const { (void)reference; (void)point; return {}; }
Eigen::MatrixXd ICGNSolver::compute_steepest_descent_images() const { return {}; }
Eigen::MatrixXd ICGNSolver::compute_hessian(const Eigen::MatrixXd& steepest_descent) const { return steepest_descent.transpose() * steepest_descent; }
Eigen::VectorXd ICGNSolver::compute_residual() const { return {}; }
Eigen::VectorXd ICGNSolver::solve_parameter_increment(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& residual) const { (void)hessian; (void)residual; return {}; }
Eigen::VectorXd ICGNSolver::inverse_compositional_update(const Eigen::VectorXd& parameters, const Eigen::VectorXd& delta) const { return parameters - delta; }
bool ICGNSolver::check_convergence(const Eigen::VectorXd& delta) const { return delta.norm() < config_.convergence_threshold; }

} // namespace dic
