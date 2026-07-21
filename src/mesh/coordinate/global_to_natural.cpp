/**
 * @file global_to_natural.cpp
 * @brief Minimal implementation placeholder for global-to-natural.
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

#include <dic/mesh/coordinate/global_to_natural.hpp>

namespace dic {

NaturalCoordinate global_to_natural(const Element& element, const std::vector<Node>& nodes, const Eigen::Vector2d& global_point) { (void)element; (void)nodes; (void)global_point; return {}; }

} // namespace dic
