/**
 * @file feature_matcher.cpp
 * @brief Minimal implementation placeholder for feature matcher.
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

#include <dic/initialization/feature_matcher.hpp>

namespace dic {

std::vector<FeatureMatch> FeatureMatcher::match(const Image& reference, const Image& deformed) const { (void)reference; (void)deformed; return {}; }

} // namespace dic
