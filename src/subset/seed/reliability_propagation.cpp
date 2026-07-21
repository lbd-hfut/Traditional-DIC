/**
 * @file reliability_propagation.cpp
 * @brief Minimal implementation placeholder for reliability propagation.
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

#include <dic/subset/seed/reliability_propagation.hpp>
#include <algorithm>

namespace dic {

std::vector<PropagationNode> ReliabilityPropagation::order(std::vector<PropagationNode> nodes) const { std::sort(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) { return a.reliability > b.reliability; }); return nodes; }

} // namespace dic
