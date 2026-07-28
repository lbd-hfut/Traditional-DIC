/**
 * @file multiview_triangulation.hpp
 * @brief Multiview triangulation placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_MULTIVIEW_TRIANGULATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_MULTIVIEW_TRIANGULATION_HPP

#include <Eigen/Dense>
#include <dic/calibration/camera_model.hpp>
#include <vector>

namespace dic {

struct TriangulationOptions {
    double max_reprojection_error = 2.0;
    bool require_positive_depth = true;
};

struct TriangulationResult {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double mean_reprojection_error = 0.0;
    double max_reprojection_error = 0.0;
    int observations_used = 0;
    bool valid = false;
};

Eigen::Vector3d triangulate_multiview(const std::vector<Eigen::Vector2d>& observations, const std::vector<CameraModel>& cameras);
TriangulationResult triangulate_multiview_checked(const std::vector<Eigen::Vector2d>& observations,
                                                  const std::vector<CameraModel>& cameras,
                                                  const TriangulationOptions& options = {});

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_GEOMETRY_MULTIVIEW_TRIANGULATION_HPP
