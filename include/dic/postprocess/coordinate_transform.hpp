/**
 * @file coordinate_transform.hpp
 * @brief Coordinate transform placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP

#include <Eigen/Dense>

namespace dic {

Eigen::Vector3d camera_to_world(const Eigen::Vector3d& point);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP
