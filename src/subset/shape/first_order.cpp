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
Eigen::Vector2d FirstOrderShapeFunction::warp(const Eigen::Vector2d& local_point, const Eigen::VectorXd& parameters) const { if (parameters.size() < 6) throw std::invalid_argument("Insufficient shape parameters."); return local_point + parameters.head<2>(); }
Eigen::MatrixXd FirstOrderShapeFunction::jacobian(const Eigen::Vector2d& local_point) const { (void)local_point; return Eigen::MatrixXd::Zero(2, 6); }

} // namespace dic
