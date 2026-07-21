/**
 * @file types.hpp
 * @brief Common numerical types and solver status definitions.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_TYPES_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_TYPES_HPP

#include <Eigen/Dense>

namespace dic {

using PixelCoordinate = Eigen::Vector2d;
using WorldCoordinate = Eigen::Vector3d;

enum class SolverStatus {
    Success,
    NotConverged,
    InvalidInput,
    NumericalFailure
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_TYPES_HPP
