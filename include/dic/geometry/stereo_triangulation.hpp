/**
 * @file stereo_triangulation.hpp
 * @brief Stereo triangulation placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_STEREO_TRIANGULATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_STEREO_TRIANGULATION_HPP

#include <Eigen/Dense>
#include <dic/calibration/camera_model.hpp>
#include <dic/geometry/multiview_triangulation.hpp>
#include <vector>

namespace dic {

Eigen::Vector3d triangulate_stereo(const Eigen::Vector2d& point_left, const Eigen::Vector2d& point_right, const CameraModel& left, const CameraModel& right);
TriangulationResult triangulate_stereo_checked(const Eigen::Vector2d& point_left,
                                               const Eigen::Vector2d& point_right,
                                               const CameraModel& left,
                                               const CameraModel& right,
                                               const TriangulationOptions& options = {});

struct StereoReconstructionResult {
    std::vector<Eigen::Vector3d> points;
    std::vector<double> mean_reprojection_errors;
    std::vector<unsigned char> valid_mask;
    double mean_reprojection_error = 0.0;
};

StereoReconstructionResult reconstruct_stereo_points(const std::vector<Eigen::Vector2d>& left_points,
                                                     const std::vector<Eigen::Vector2d>& right_points,
                                                     const CameraModel& left,
                                                     const CameraModel& right,
                                                     const TriangulationOptions& options = {});

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_STEREO_TRIANGULATION_HPP
