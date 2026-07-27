/**
 * @file first_order.cpp
 * @brief Minimal implementation placeholder for first_order.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/subset/shape/first_order.hpp>
#include <stdexcept>

namespace dic {

int FirstOrderShapeFunction::parameter_count() const { return 6; }
Eigen::Vector2d FirstOrderShapeFunction::warp(const Eigen::Vector2d& local_point, const Eigen::VectorXd& parameters) const
{
    if (parameters.size() < 6) {
        throw std::invalid_argument("Insufficient shape parameters.");
    }

    const double x = local_point.x();
    const double y = local_point.y();
    return {
        x + parameters(0) + parameters(2) * x + parameters(3) * y,
        y + parameters(1) + parameters(4) * x + parameters(5) * y
    };
}

Eigen::MatrixXd FirstOrderShapeFunction::jacobian(const Eigen::Vector2d& local_point) const
{
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(2, 6);
    jacobian(0, 0) = 1.0;
    jacobian(0, 2) = local_point.x();
    jacobian(0, 3) = local_point.y();
    jacobian(1, 1) = 1.0;
    jacobian(1, 4) = local_point.x();
    jacobian(1, 5) = local_point.y();
    return jacobian;
}

} // namespace dic
