/**
 * @file seed_selector.cpp
 * @brief Minimal implementation placeholder for seed selector.
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

#include <dic/subset/seed/seed_selector.hpp>

namespace dic {

std::vector<Eigen::Vector2d> SeedSelector::select(const Image& reference, const Image& deformed) const { (void)reference; (void)deformed; return {}; }

} // namespace dic
