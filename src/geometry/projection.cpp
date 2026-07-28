/**
 * @file projection.cpp
 * @brief Minimal implementation placeholder for projection.
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

#include <dic/geometry/projection.hpp>

#include <cmath>
#include <limits>

namespace dic {

Eigen::Vector3d world_to_camera(const Eigen::Vector3d& point, const CameraModel& camera)
{
    return camera.R * point + camera.t;
}

Eigen::Vector2d distort_normalized_point(const Eigen::Vector2d& normalized,
                                         const std::vector<double>& distortion)
{
    const double x = normalized.x();
    const double y = normalized.y();
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double k1 = distortion.size() > 0 ? distortion[0] : 0.0;
    const double k2 = distortion.size() > 1 ? distortion[1] : 0.0;
    const double p1 = distortion.size() > 2 ? distortion[2] : 0.0;
    const double p2 = distortion.size() > 3 ? distortion[3] : 0.0;
    const double k3 = distortion.size() > 4 ? distortion[4] : 0.0;
    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    return {x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y};
}

Eigen::Vector2d project_point(const Eigen::Vector3d& point, const CameraModel& camera)
{
    const Eigen::Vector3d pc = world_to_camera(point, camera);
    if (std::abs(pc.z()) < 1e-12) {
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    }
    const Eigen::Vector2d normalized(pc.x() / pc.z(), pc.y() / pc.z());
    const Eigen::Vector2d distorted = distort_normalized_point(normalized, camera.distortion);
    const Eigen::Vector3d pixel = camera.K * Eigen::Vector3d(distorted.x(), distorted.y(), 1.0);
    return {pixel.x() / pixel.z(), pixel.y() / pixel.z()};
}

double reprojection_error(const Eigen::Vector3d& point,
                          const Eigen::Vector2d& observation,
                          const CameraModel& camera)
{
    const Eigen::Vector2d projected = project_point(point, camera);
    if (!std::isfinite(projected.x()) || !std::isfinite(projected.y())) {
        return std::numeric_limits<double>::infinity();
    }
    return (projected - observation).norm();
}

double camera_depth(const Eigen::Vector3d& point, const CameraModel& camera)
{
    return world_to_camera(point, camera).z();
}

} // namespace dic
