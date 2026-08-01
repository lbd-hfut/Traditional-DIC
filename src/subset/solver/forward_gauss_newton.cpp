#include <dic/interpolation/bspline.hpp>
#include <dic/subset/solver/forward_gauss_newton.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace dic {
namespace {

constexpr double kEpsilon = 1e-12;

bool finite_6_params(const Eigen::Matrix<double, 6, 1>& p)
{
    for (int i = 0; i < p.size(); ++i) {
        if (!std::isfinite(p(i))) {
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


double vector_norm(const std::vector<double>& values, double mean)
{
    double sum = 0.0;
    for (const double value : values) {
        const double diff = value - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Forward-compositional 6×1 steepest descent from deformed image gradient at warped position.
// ∇g_at_point comes from BSplineInterpolator::gradient() evaluated at (warped_x, warped_y).
// Chain rule:  SD = ∇g(W(x;p)) · ∂W(z;p)/∂z|_{z=(lx,ly)} · ∂W(x;Δp)/∂Δp|₀
Eigen::Matrix<double, 6, 1> forward_steepest_descent(
    double gx, double gy,
    double local_x, double local_y,
    const Eigen::Matrix<double, 6, 1>& parameters)
{
    // ∂W(z;p)/∂z for the first-order warp: [[1+p2, p3], [p4, 1+p5]]
    const double gx_eff = gx * (1.0 + parameters(2)) + gy * parameters(4);
    const double gy_eff = gx * parameters(3) + gy * (1.0 + parameters(5));
    Eigen::Matrix<double, 6, 1> sd;
    sd << gx_eff,
          gy_eff,
          gx_eff * local_x,
          gx_eff * local_y,
          gy_eff * local_x,
          gy_eff * local_y;
    return sd;
}

// Forward-compositional warp update:  W(x;p_new) = W(x;p) ∘ W(x;Δp).
// Exact for first-order (affine) warps.
Eigen::Matrix<double, 6, 1> compose_warp(
    const Eigen::Matrix<double, 6, 1>& p,
    const Eigen::Matrix<double, 6, 1>& dp)
{
    Eigen::Matrix<double, 6, 1> pn;
    pn(0) = dp(0) + p(0) + p(2) * dp(0) + p(3) * dp(1);
    pn(1) = dp(1) + p(1) + p(4) * dp(0) + p(5) * dp(1);
    pn(2) = dp(2) + p(2) * (1.0 + dp(2)) + p(3) * dp(4);
    pn(3) = dp(3) + p(2) * dp(3) + p(3) * (1.0 + dp(5));
    pn(4) = dp(4) + p(4) * (1.0 + dp(2)) + p(5) * dp(4);
    pn(5) = dp(5) + p(4) * dp(3) + p(5) * (1.0 + dp(5));
    return pn;
}

} // namespace

// ---------------------------------------------------------------------------
//  Public
// ---------------------------------------------------------------------------

ForwardGaussNewtonSolver::ForwardGaussNewtonSolver(SubsetConfig config)
    : config_(config)
{
}

Displacement2D ForwardGaussNewtonSolver::solve(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    // FGN only uses deformed image for warp + gradient.
    // Create only the deformed interpolator (no reference needed).
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    const bool second_order = config_.shape_function == SubsetShapeFunctionMethod::SecondOrder ||
                              config_.use_second_order;
    if (second_order) {
        if (config_.objective == CorrelationCriterionKind::SSD) {
            return solve_second_order_ssd(reference, deformed, point, initial,
                                          BSplineInterpolator(nullptr), deformed_interpolator);
        }
        return solve_second_order_znssd(reference, deformed, point, initial,
                                         BSplineInterpolator(nullptr), deformed_interpolator);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd(reference, deformed, point, initial,
                                      BSplineInterpolator(nullptr), deformed_interpolator);
    }
    return solve_first_order_znssd(reference, deformed, point, initial,
                                    BSplineInterpolator(nullptr), deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_with_interpolators(
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
            return solve_second_order_ssd(reference, deformed, point, initial,
                                          reference_interpolator, deformed_interpolator);
        }
        return solve_second_order_znssd(reference, deformed, point, initial,
                                        reference_interpolator, deformed_interpolator);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd(reference, deformed, point, initial,
                                      reference_interpolator, deformed_interpolator);
    }
    return solve_first_order_znssd(reference, deformed, point, initial,
                                    reference_interpolator, deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_with_mask(
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
        return solve_second_order_znssd_masked(reference, deformed, roi, point, initial,
                                                 reference_interpolator, deformed_interpolator);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd_masked(reference, deformed, roi, point, initial,
                                             reference_interpolator, deformed_interpolator);
    }
    return solve_first_order_znssd_masked(reference, deformed, roi, point, initial,
                                           reference_interpolator, deformed_interpolator);
}

// ---------------------------------------------------------------------------
//  First-order + ZNSSD  (Forward-compositional Gauss-Newton)
// ---------------------------------------------------------------------------

Displacement2D ForwardGaussNewtonSolver::solve_first_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_first_order_znssd(reference, deformed, point, initial,
                                    BSplineInterpolator(nullptr), deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_first_order_znssd(
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

    (void)reference_interpolator;   // FGN uses only deformed interpolator

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

    // --- Build sample list (reference-image coordinates, values, and precomputed graded) ---
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
        double reference_normalized;
    };
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    if (samples.size() < 6) {
        return result;
    }

    // --- ZNSSD normalization: reference (precomputed once) ---
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

    // --- Initialize 6 parameters ---
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

        // --- Warp & interpolate deformed image ---
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

        // --- ZNSSD normalization: deformed (per-iteration) ---
        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        // --- Build Hessian & gradient from deformed image gradient ---
        // Forward-compositional: compute steepest-descent images on-the-fly
        // using ∇g (deformed) instead of ∇f (reference).
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y;

            // Deformed image gradient at the warped position
            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);

            // Steepest descent:  ∇g · ∂W/∂p  (evaluated at current warp)
            const Eigen::Matrix<double, 6, 1> sd = forward_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;

            hessian += sd * sd.transpose();
            gradient += normalized_difference * sd;
            corrcoef += normalized_difference * normalized_difference;
        }
        hessian *= 2.0 / (reference_norm * reference_norm);
        gradient *= 2.0 / reference_norm;

        // --- Solve: Forward-compositional  Δp = H⁻¹·g  (no minus sign) ---
        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 6, 1> delta = decomposition.solve(gradient);
        if (!finite_6_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        double delta_norm = delta.norm();
        // Step cap: ZNSSD provides natural damping, but FC-GN's recomputed
        // Hessian can still produce overshoot steps that warp points outside
        // the image boundary. A moderate cap preserves convergence while
        // preventing InvalidInput early-exits.
        constexpr double kMaxStep = 0.5;
        if (delta_norm > kMaxStep) {
            delta *= kMaxStep / delta_norm;
            delta_norm = kMaxStep;
        }
        // Forward-compositional update  W(x;p_new) = W(x;p) ∘ W(x;Δp)
        parameters = compose_warp(parameters, delta);
        if (!finite_6_params(parameters)) {
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

Displacement2D ForwardGaussNewtonSolver::solve_first_order_znssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    (void)reference_interpolator;   // FGN uses only deformed interpolator

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
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
        double reference_normalized;
    };
    std::vector<Sample> samples;
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

            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
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

        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 6, 1> sd = forward_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;

            hessian += sd * sd.transpose();
            gradient += normalized_difference * sd;
            corrcoef += normalized_difference * normalized_difference;
        }
        hessian *= 2.0 / (reference_norm * reference_norm);
        gradient *= 2.0 / reference_norm;

        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 6, 1> delta = decomposition.solve(gradient);
        if (!finite_6_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        double delta_norm = delta.norm();
        constexpr double kMaxStepMasked = 0.5;
        if (delta_norm > kMaxStepMasked) {
            delta *= kMaxStepMasked / delta_norm;
            delta_norm = kMaxStepMasked;
        }
        parameters = compose_warp(parameters, delta);
        if (!finite_6_params(parameters)) {
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

// ---------------------------------------------------------------------------
//  First-order + SSD  (Forward-compositional Gauss-Newton, no normalisation)
// ---------------------------------------------------------------------------

Displacement2D ForwardGaussNewtonSolver::solve_first_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_first_order_ssd(reference, deformed, point, initial,
                                  BSplineInterpolator(nullptr), deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_first_order_ssd(
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

    (void)reference_interpolator;   // FGN uses only deformed interpolator for gradient

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

    // --- Build sample list (raw reference values, no normalisation) ---
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
    };
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            samples.push_back(sample);
        }
    }

    if (samples.size() < 6) {
        return result;
    }

    // --- Initialize 6 parameters ---
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

        // --- Warp & interpolate deformed image (raw values, no normalisation) ---
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
            deformed_values.push_back(deformed_interpolator.value(warped_x, warped_y));
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        // --- Build Hessian & gradient using raw residuals ---
        // SSD: H = Σ[SD·SDᵀ],  g = Σ[(f - g_raw)·SD]
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 6, 1> sd = forward_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double residual = samples[i].reference_value - deformed_values[i];

            hessian += sd * sd.transpose();
            gradient += residual * sd;
            corrcoef += residual * residual;
        }

        // --- Solve: Forward-compositional  Δp = H⁻¹·g  ---
        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 6, 1> delta = decomposition.solve(gradient);
        if (!finite_6_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        // Cap the increment step to damp GN overshoot.
        // The GN Hessian approximation can produce excessively large
        // increment warps when the deformed-image gradient differs
        // significantly from the reference gradient.
        constexpr double kMaxTranslationStep = 0.35;
        const double delta_norm = delta.norm();
        if (delta_norm > kMaxTranslationStep) {
            delta *= kMaxTranslationStep / delta_norm;
        }
        parameters = compose_warp(parameters, delta);
        if (!finite_6_params(parameters)) {
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

Displacement2D ForwardGaussNewtonSolver::solve_first_order_ssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    (void)reference_interpolator;   // FGN uses only deformed interpolator for gradient

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
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
    };
    std::vector<Sample> samples;
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

            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(6, min_samples)) {
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
            deformed_values.push_back(deformed_interpolator.value(warped_x, warped_y));
        }

        if (!all_warped_points_valid || deformed_values.size() != samples.size()) {
            result.status = SolverStatus::InvalidInput;
            return result;
        }

        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 6, 1> sd = forward_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double residual = samples[i].reference_value - deformed_values[i];

            hessian += sd * sd.transpose();
            gradient += residual * sd;
            corrcoef += residual * residual;
        }

        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 6, 1> delta = decomposition.solve(gradient);
        if (!finite_6_params(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        constexpr double kMaxTranslationStep = 0.35;
        double delta_norm = delta.norm();
        if (delta_norm > kMaxTranslationStep) {
            delta *= kMaxTranslationStep / delta_norm;
            delta_norm = kMaxTranslationStep;
        }
        parameters = compose_warp(parameters, delta);
        if (!finite_6_params(parameters)) {
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

// ---------------------------------------------------------------------------
//  Second-order + ZNSSD  (Forward-compositional Gauss-Newton, 12 parameters)
// ---------------------------------------------------------------------------

namespace {

bool finite_12_params_fgn(const Eigen::Matrix<double, 12, 1>& p)
{
    for (int i = 0; i < 12; ++i) {
        if (!std::isfinite(p(i))) {
            return false;
        }
    }
    return true;
}

Eigen::Matrix<double, 12, 1> forward_second_order_steepest_descent(
    double gx, double gy, double local_x, double local_y,
    const Eigen::Matrix<double, 12, 1>& parameters)
{
    const double dx2_2 = local_x * local_x * 0.5;
    const double dy2_2 = local_y * local_y * 0.5;
    const double dxy = local_x * local_y;

    // ∂W(z;p)/∂z at z=(lx,ly) for the second-order warp (position-dependent)
    const double a00 = 1.0 + parameters(2) + parameters(6) * local_x + parameters(7) * local_y;
    const double a01 = parameters(3) + parameters(7) * local_x + parameters(8) * local_y;
    const double a10 = parameters(4) + parameters(9) * local_x + parameters(10) * local_y;
    const double a11 = 1.0 + parameters(5) + parameters(10) * local_x + parameters(11) * local_y;

    const double gx_eff = gx * a00 + gy * a10;
    const double gy_eff = gx * a01 + gy * a11;

    Eigen::Matrix<double, 12, 1> sd;
    // Parameters: [u, v, du_dx, du_dy, dv_dx, dv_dy,
    //              d2u_dx2, d2u_dxdy, d2u_dy2, d2v_dx2, d2v_dxdy, d2v_dy2]
    sd << gx_eff,
          gy_eff,
          gx_eff * local_x,
          gx_eff * local_y,
          gy_eff * local_x,
          gy_eff * local_y,
          gx_eff * dx2_2,
          gx_eff * dxy,
          gx_eff * dy2_2,
          gy_eff * dx2_2,
          gy_eff * dxy,
          gy_eff * dy2_2;
    return sd;
}

// Forward-compositional warp update for the second-order shape function:
// W(x;p_new) = W(x;p) ∘ W(x;Δp), expanded up to second order in (lx,ly) and
// first order in Δp (products Δpᵢ·Δpⱼ dropped), consistent with the GN
// linearisation.
Eigen::Matrix<double, 12, 1> compose_warp(
    const Eigen::Matrix<double, 12, 1>& p,
    const Eigen::Matrix<double, 12, 1>& dp)
{
    Eigen::Matrix<double, 12, 1> pn;
    // Translation
    pn(0) = dp(0) + p(0) + p(2) * dp(0) + p(3) * dp(1);
    pn(1) = dp(1) + p(1) + p(4) * dp(0) + p(5) * dp(1);
    // First-order terms
    pn(2) = dp(2) + p(2) * (1.0 + dp(2)) + p(3) * dp(4) + p(6) * dp(0) + p(7) * dp(1);
    pn(3) = dp(3) + p(2) * dp(3) + p(3) * (1.0 + dp(5)) + p(7) * dp(0) + p(8) * dp(1);
    pn(4) = dp(4) + p(4) * (1.0 + dp(2)) + p(5) * dp(4) + p(9) * dp(0) + p(10) * dp(1);
    pn(5) = dp(5) + p(4) * dp(3) + p(5) * (1.0 + dp(5)) + p(10) * dp(0) + p(11) * dp(1);
    // Second-order u terms
    pn(6) = dp(6) + p(2) * dp(6) + p(3) * dp(9) +
            p(6) * (1.0 + 2.0 * dp(2)) + 2.0 * p(7) * dp(4);
    pn(7) = dp(7) + p(2) * dp(7) + p(3) * dp(10) +
            p(6) * dp(3) + p(7) * (1.0 + dp(2) + dp(5)) + p(8) * dp(4);
    pn(8) = dp(8) + p(2) * dp(8) + p(3) * dp(11) +
            p(7) * dp(3) + p(8) * (1.0 + 2.0 * dp(5));
    // Second-order v terms
    pn(9) = dp(9) + p(4) * dp(6) + p(5) * dp(9) +
            p(9) * (1.0 + 2.0 * dp(2)) + 2.0 * p(10) * dp(4);
    pn(10) = dp(10) + p(4) * dp(7) + p(5) * dp(10) +
             p(9) * dp(3) + p(10) * (1.0 + dp(2) + dp(5)) + p(11) * dp(4);
    pn(11) = dp(11) + p(4) * dp(8) + p(5) * dp(11) +
             p(10) * dp(3) + p(11) * (1.0 + 2.0 * dp(5));
    return pn;
}

} // namespace

Displacement2D ForwardGaussNewtonSolver::solve_second_order_znssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_second_order_znssd(reference, deformed, point, initial,
                                     BSplineInterpolator(nullptr), deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_second_order_znssd(
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

    (void)reference_interpolator;   // FGN uses only deformed interpolator

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

    // --- Build sample list with ZNSSD normalization ---
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
        double reference_normalized;
    };
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    double reference_mean = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    if (samples.size() < 12) {
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

    // --- Initialize 12 parameters ---
    Eigen::Matrix<double, 12, 1> parameters = Eigen::Matrix<double, 12, 1>::Zero();
    parameters(0) = initial.u;
    parameters(1) = initial.v;
    parameters(2) = initial.du_dx;
    parameters(3) = initial.du_dy;
    parameters(4) = initial.dv_dx;
    parameters(5) = initial.dv_dy;
    // parameters(6..11) = 0 (second-order terms start at zero)

    double corrcoef = std::numeric_limits<double>::infinity();
    bool converged = false;
    std::vector<double> deformed_values;
    deformed_values.reserve(samples.size());

    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
        deformed_values.clear();

        // --- Second-order warp & interpolate deformed image ---
        double deformed_mean = 0.0;
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double dx2_2 = sample.local_x * sample.local_x * 0.5;
            const double dy2_2 = sample.local_y * sample.local_y * 0.5;
            const double dxy = sample.local_x * sample.local_y;

            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;
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

        // --- ZNSSD normalization: deformed (per-iteration) ---
        deformed_mean /= static_cast<double>(deformed_values.size());
        const double deformed_norm = vector_norm(deformed_values, deformed_mean);
        if (deformed_norm <= kEpsilon) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        // --- Build 12×12 Hessian & 12×1 gradient ---
        Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double dx2_2 = samples[i].local_x * samples[i].local_x * 0.5;
            const double dy2_2 = samples[i].local_y * samples[i].local_y * 0.5;
            const double dxy = samples[i].local_x * samples[i].local_y;

            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 12, 1> sd = forward_second_order_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;

            hessian += sd * sd.transpose();
            gradient += normalized_difference * sd;
            corrcoef += normalized_difference * normalized_difference;
        }
        hessian *= 2.0 / (reference_norm * reference_norm);
        gradient *= 2.0 / reference_norm;

        // LM regularization for 12×12 conditioning
        const double lambda = 1e-4 * hessian.diagonal().maxCoeff();
        for (int i = 0; i < 12; ++i) {
            hessian(i, i) += lambda;
        }

        // --- Solve: Forward-compositional  Δp = H⁻¹·g ---
        Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 12, 1> delta = decomposition.solve(gradient);
        if (!finite_12_params_fgn(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        double delta_norm = delta.norm();
        constexpr double kMaxStepSecondOrder = 0.5;
        if (delta_norm > kMaxStepSecondOrder) {
            delta *= kMaxStepSecondOrder / delta_norm;
            delta_norm = kMaxStepSecondOrder;
        }
        parameters = compose_warp(parameters, delta);
        if (!finite_12_params_fgn(parameters)) {
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

Displacement2D ForwardGaussNewtonSolver::solve_second_order_znssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    (void)reference_interpolator;   // FGN uses only deformed interpolator

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
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
        double reference_normalized;
    };
    std::vector<Sample> samples;
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

            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            reference_mean += sample.reference_value;
            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(12, min_samples)) {
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
            const double dx2_2 = sample.local_x * sample.local_x * 0.5;
            const double dy2_2 = sample.local_y * sample.local_y * 0.5;
            const double dxy = sample.local_x * sample.local_y;

            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;
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

        Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double dx2_2 = samples[i].local_x * samples[i].local_x * 0.5;
            const double dy2_2 = samples[i].local_y * samples[i].local_y * 0.5;
            const double dxy = samples[i].local_x * samples[i].local_y;

            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 12, 1> sd = forward_second_order_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double normalized_difference =
                samples[i].reference_normalized - (deformed_values[i] - deformed_mean) / deformed_norm;

            hessian += sd * sd.transpose();
            gradient += normalized_difference * sd;
            corrcoef += normalized_difference * normalized_difference;
        }
        hessian *= 2.0 / (reference_norm * reference_norm);
        gradient *= 2.0 / reference_norm;

        Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 12, 1> delta = decomposition.solve(gradient);
        if (!finite_12_params_fgn(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        constexpr double kMaxStepMasked = 0.3;
        const double delta_norm = delta.norm();
        if (delta_norm > kMaxStepMasked) {
            delta *= kMaxStepMasked / delta_norm;
        }
        parameters = compose_warp(parameters, delta);
        if (!finite_12_params_fgn(parameters)) {
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

// ---------------------------------------------------------------------------
//  Second-order + SSD  (Forward-compositional Gauss-Newton, 12 parameters)
// ---------------------------------------------------------------------------

Displacement2D ForwardGaussNewtonSolver::solve_second_order_ssd(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_second_order_ssd(reference, deformed, point, initial,
                                   BSplineInterpolator(nullptr), deformed_interpolator);
}

Displacement2D ForwardGaussNewtonSolver::solve_second_order_ssd(
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

    (void)reference_interpolator;   // FGN uses only deformed interpolator

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

    // --- Build sample list (no ZNSSD normalization for SSD) ---
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
    };
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const int x = center_x + dx;
            const int y = center_y + dy;
            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            samples.push_back(sample);
        }
    }

    if (samples.size() < 12) {
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

        // --- Second-order warp & interpolate deformed image ---
        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double dx2_2 = sample.local_x * sample.local_x * 0.5;
            const double dy2_2 = sample.local_y * sample.local_y * 0.5;
            const double dxy = sample.local_x * sample.local_y;

            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;
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

        // --- Build 12×12 Hessian & 12×1 gradient (SSD: no normalization) ---
        Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double dx2_2 = samples[i].local_x * samples[i].local_x * 0.5;
            const double dy2_2 = samples[i].local_y * samples[i].local_y * 0.5;
            const double dxy = samples[i].local_x * samples[i].local_y;

            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 12, 1> sd = forward_second_order_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double residual = samples[i].reference_value - deformed_values[i];

            // SSD: H = Σ[J·Jᵀ], g = Σ[(ref-def)·J]
            hessian += sd * sd.transpose();
            gradient += residual * sd;
            corrcoef += residual * residual;
        }

        // LM regularization for 12×12 conditioning
        const double lambda = 1e-4 * hessian.diagonal().maxCoeff();
        for (int i = 0; i < 12; ++i) {
            hessian(i, i) += lambda;
        }

        // --- Solve: Forward-compositional  Δp = H⁻¹·g ---
        Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 12, 1> delta = decomposition.solve(gradient);
        if (!finite_12_params_fgn(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        // Step cap: SSD + 12-param forward GN oscillates without damping
        constexpr double kMaxStep = 0.35;
        double delta_norm = delta.norm();
        if (delta_norm > kMaxStep) {
            delta *= kMaxStep / delta_norm;
            delta_norm = kMaxStep;
        }
        parameters = compose_warp(parameters, delta);

        if (!finite_12_params_fgn(parameters)) {
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

Displacement2D ForwardGaussNewtonSolver::solve_second_order_ssd_masked(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    (void)reference_interpolator;   // FGN uses only deformed interpolator

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

    // --- Build sample list (SSD: no ZNSSD normalization), skip pixels outside ROI ---
    struct Sample {
        int x, y;
        double local_x, local_y;
        double reference_value;
    };
    std::vector<Sample> samples;
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

            Sample sample;
            sample.x = x;
            sample.y = y;
            sample.local_x = static_cast<double>(dx);
            sample.local_y = static_cast<double>(dy);
            sample.reference_value = static_cast<double>(reference.at(x, y));
            samples.push_back(sample);
        }
    }

    const int min_samples = std::max(config_.min_valid_samples,
        static_cast<int>(std::ceil(config_.min_valid_sample_ratio * static_cast<double>(full_count))));
    if (static_cast<int>(samples.size()) < std::max(12, min_samples)) {
        return result;
    }

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

        bool all_warped_points_valid = true;
        for (const auto& sample : samples) {
            const double dx2_2 = sample.local_x * sample.local_x * 0.5;
            const double dy2_2 = sample.local_y * sample.local_y * 0.5;
            const double dxy = sample.local_x * sample.local_y;

            const double warped_x = static_cast<double>(center_x) + sample.local_x +
                parameters(0) + parameters(2) * sample.local_x + parameters(3) * sample.local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + sample.local_y +
                parameters(1) + parameters(4) * sample.local_x + parameters(5) * sample.local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;
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

        Eigen::Matrix<double, 12, 12> hessian = Eigen::Matrix<double, 12, 12>::Zero();
        Eigen::Matrix<double, 12, 1> gradient = Eigen::Matrix<double, 12, 1>::Zero();
        corrcoef = 0.0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double dx2_2 = samples[i].local_x * samples[i].local_x * 0.5;
            const double dy2_2 = samples[i].local_y * samples[i].local_y * 0.5;
            const double dxy = samples[i].local_x * samples[i].local_y;

            const double warped_x = static_cast<double>(center_x) + samples[i].local_x +
                parameters(0) + parameters(2) * samples[i].local_x + parameters(3) * samples[i].local_y +
                parameters(6) * dx2_2 + parameters(7) * dxy + parameters(8) * dy2_2;
            const double warped_y = static_cast<double>(center_y) + samples[i].local_y +
                parameters(1) + parameters(4) * samples[i].local_x + parameters(5) * samples[i].local_y +
                parameters(9) * dx2_2 + parameters(10) * dxy + parameters(11) * dy2_2;

            const auto def_gradient = deformed_interpolator.gradient(warped_x, warped_y);
            const Eigen::Matrix<double, 12, 1> sd = forward_second_order_steepest_descent(
                def_gradient.x(), def_gradient.y(),
                samples[i].local_x, samples[i].local_y, parameters);

            const double residual = samples[i].reference_value - deformed_values[i];

            hessian += sd * sd.transpose();
            gradient += residual * sd;
            corrcoef += residual * residual;
        }

        const double lambda = 1e-4 * hessian.diagonal().maxCoeff();
        for (int i = 0; i < 12; ++i) {
            hessian(i, i) += lambda;
        }

        Eigen::LDLT<Eigen::Matrix<double, 12, 12>> decomposition(hessian);
        if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }
        Eigen::Matrix<double, 12, 1> delta = decomposition.solve(gradient);
        if (!finite_12_params_fgn(delta)) {
            result.status = SolverStatus::NumericalFailure;
            return result;
        }

        constexpr double kMaxStep = 0.35;
        double delta_norm = delta.norm();
        if (delta_norm > kMaxStep) {
            delta *= kMaxStep / delta_norm;
            delta_norm = kMaxStep;
        }
        parameters = compose_warp(parameters, delta);

        if (!finite_12_params_fgn(parameters)) {
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

} // namespace dic
