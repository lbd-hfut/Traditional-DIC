/**
 * @file t3.cpp
 * @brief Minimal implementation placeholder for t3.
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

#include <dic/mesh/element/t3.hpp>

namespace dic {

int T3Element::node_count() const { return 3; }
Eigen::VectorXd T3Element::shape_functions(double xi, double eta) const { (void)xi; (void)eta; return Eigen::VectorXd::Zero(3); }
Eigen::MatrixXd T3Element::shape_function_derivatives(double xi, double eta) const { (void)xi; (void)eta; return Eigen::MatrixXd::Zero(3, 2); }

} // namespace dic
