/**
 * @file reliability_propagation.hpp
 * @brief Reliability-guided POI propagation skeleton.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP

#include <dic/initialization/initializer.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

struct PropagationNode { Eigen::Vector2d point{0.0, 0.0}; InitialDisplacement initial; double reliability{0.0}; };
class ReliabilityPropagation { public: std::vector<PropagationNode> order(std::vector<PropagationNode> nodes) const; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP
