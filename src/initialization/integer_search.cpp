/**
 * @file integer_search.cpp
 * @brief Minimal implementation placeholder for integer_search.
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

#include <dic/initialization/integer_search.hpp>

namespace dic {

IntegerSearchInitializer::IntegerSearchInitializer(int search_radius) : search_radius_(search_radius) {}
InitialDisplacement IntegerSearchInitializer::estimate(const Image& reference, const Image& deformed, const Eigen::Vector2d& point) const { (void)reference; (void)deformed; (void)point; return {0.0, 0.0, 0.0, false}; }

} // namespace dic
