/**
 * @file subset_dic.cpp
 * @brief Minimal implementation placeholder for Subset-DIC controller.
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

#include <dic/subset/subset_dic.hpp>

namespace dic {

SubsetDIC::SubsetDIC(SubsetConfig config) : config_(config) {}
std::vector<Displacement2D> SubsetDIC::compute(const Image& reference, const Image& deformed) const { (void)reference; (void)deformed; (void)config_; return {}; }

} // namespace dic
