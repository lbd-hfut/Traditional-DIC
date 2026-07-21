/**
 * @file znssd.cpp
 * @brief Minimal implementation placeholder for znssd.
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

#include <dic/correlation/znssd.hpp>
#include <stdexcept>

namespace dic {

double ZNSSDCorrelation::evaluate(const Eigen::VectorXd& reference, const Eigen::VectorXd& deformed) const { (void)reference; (void)deformed; throw std::runtime_error("Not implemented yet."); }

} // namespace dic
