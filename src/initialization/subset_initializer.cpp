/**
 * @file subset_initializer.cpp
 * @brief Minimal implementation placeholder for subset_initializer.
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

#include <dic/initialization/subset_initializer.hpp>

namespace dic {

SubsetInitializer::SubsetInitializer(int search_radius) : search_radius_(search_radius) {}
InitialDisplacement SubsetInitializer::estimate(const Image& reference, const Image& deformed, const Eigen::Vector2d& point) const { (void)reference; (void)deformed; (void)point; return {0.0, 0.0, 0.0, false}; }

} // namespace dic
