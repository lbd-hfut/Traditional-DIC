/**
 * @file stereo_triangulation.cpp
 * @brief Minimal implementation placeholder for stereo triangulation.
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

#include <dic/geometry/stereo_triangulation.hpp>

#include <stdexcept>

namespace dic {

TriangulationResult triangulate_stereo_checked(const Eigen::Vector2d& point_left,
                                               const Eigen::Vector2d& point_right,
                                               const CameraModel& left,
                                               const CameraModel& right,
                                               const TriangulationOptions& options)
{
    return triangulate_multiview_checked({point_left, point_right}, {left, right}, options);
}

Eigen::Vector3d triangulate_stereo(const Eigen::Vector2d& point_left,
                                   const Eigen::Vector2d& point_right,
                                   const CameraModel& left,
                                   const CameraModel& right)
{
    const TriangulationResult result = triangulate_stereo_checked(point_left, point_right, left, right);
    if (!result.valid) {
        throw std::runtime_error("Stereo triangulation failed quality checks.");
    }
    return result.point;
}

StereoReconstructionResult reconstruct_stereo_points(const std::vector<Eigen::Vector2d>& left_points,
                                                     const std::vector<Eigen::Vector2d>& right_points,
                                                     const CameraModel& left,
                                                     const CameraModel& right,
                                                     const TriangulationOptions& options)
{
    if (left_points.size() != right_points.size()) {
        throw std::invalid_argument("reconstruct_stereo_points requires matching left/right point counts.");
    }

    StereoReconstructionResult output;
    output.points.reserve(left_points.size());
    output.mean_reprojection_errors.reserve(left_points.size());
    output.valid_mask.reserve(left_points.size());

    double error_sum = 0.0;
    int valid_count = 0;
    for (std::size_t i = 0; i < left_points.size(); ++i) {
        const TriangulationResult point =
            triangulate_stereo_checked(left_points[i], right_points[i], left, right, options);
        output.points.push_back(point.point);
        output.mean_reprojection_errors.push_back(point.mean_reprojection_error);
        output.valid_mask.push_back(point.valid ? 1 : 0);
        if (point.valid) {
            error_sum += point.mean_reprojection_error;
            ++valid_count;
        }
    }
    output.mean_reprojection_error = valid_count > 0 ? error_sum / static_cast<double>(valid_count) : 0.0;
    return output;
}

} // namespace dic
