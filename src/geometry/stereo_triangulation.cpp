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

Eigen::Vector3d triangulate_stereo(const Eigen::Vector2d& point_left, const Eigen::Vector2d& point_right, const CameraModel& left, const CameraModel& right) { (void)point_left; (void)point_right; (void)left; (void)right; throw std::runtime_error("Not implemented yet."); }

} // namespace dic
