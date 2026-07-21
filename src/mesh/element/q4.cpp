/**
 * @file q4.cpp
 * @brief Minimal implementation placeholder for q4.
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

#include <dic/mesh/element/q4.hpp>

namespace dic {

int Q4Element::node_count() const { return 4; }
Eigen::VectorXd Q4Element::shape_functions(double xi, double eta) const { (void)xi; (void)eta; return Eigen::VectorXd::Zero(4); }
Eigen::MatrixXd Q4Element::shape_function_derivatives(double xi, double eta) const { (void)xi; (void)eta; return Eigen::MatrixXd::Zero(4, 2); }

} // namespace dic
