#include <dic/interpolation/bspline.hpp>
#include <dic/subset/shape/first_order.hpp>
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

} // namespace

ICGNSolver::ICGNSolver(SubsetConfig config) : config_(config) {}

Displacement2D ICGNSolver::solve(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    if (config_.shape_function == SubsetShapeFunctionMethod::SecondOrder || config_.use_second_order) {
        return solve_second_order_placeholder(point, initial);
    }
    return solve_first_order(reference, deformed, point, initial);
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
    if (config_.shape_function == SubsetShapeFunctionMethod::SecondOrder || config_.use_second_order) {
        return solve_second_order_placeholder(point, initial);
    }
    return solve_first_order(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
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
    if (config_.shape_function == SubsetShapeFunctionMethod::SecondOrder || config_.use_second_order) {
        return solve_second_order_placeholder(point, initial);
    }
    return solve_first_order_masked(
        reference, deformed, roi, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_first_order(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    BSplineInterpolator reference_interpolator(reference, config_.image_precompute);
    BSplineInterpolator deformed_interpolator(deformed, config_.image_precompute);
    return solve_first_order(reference, deformed, point, initial, reference_interpolator, deformed_interpolator);
}

Displacement2D ICGNSolver::solve_first_order(
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

Displacement2D ICGNSolver::solve_first_order_masked(
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

Displacement2D ICGNSolver::solve_second_order_placeholder(
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
