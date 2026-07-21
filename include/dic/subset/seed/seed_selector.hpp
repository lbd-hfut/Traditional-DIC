/**
 * @file seed_selector.hpp
 * @brief Seed candidate selection for reliability-guided Subset-DIC.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP

#include <dic/core/image.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

class SeedSelector { public: std::vector<Eigen::Vector2d> select(const Image& reference, const Image& deformed) const; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP
