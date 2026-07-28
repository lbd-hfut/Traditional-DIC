/**
 * @file projection.hpp
 * @brief Project 3D world points into image coordinates.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_PROJECTION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_PROJECTION_HPP

#include <Eigen/Dense>
#include <dic/calibration/camera_model.hpp>

namespace dic {

Eigen::Vector3d world_to_camera(const Eigen::Vector3d& point, const CameraModel& camera);
Eigen::Vector2d distort_normalized_point(const Eigen::Vector2d& normalized,
                                         const std::vector<double>& distortion);
Eigen::Vector2d project_point(const Eigen::Vector3d& point, const CameraModel& camera);
double reprojection_error(const Eigen::Vector3d& point,
                          const Eigen::Vector2d& observation,
                          const CameraModel& camera);
double camera_depth(const Eigen::Vector3d& point, const CameraModel& camera);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_PROJECTION_HPP
