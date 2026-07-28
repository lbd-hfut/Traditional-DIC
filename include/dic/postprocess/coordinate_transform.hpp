/**
 * @file coordinate_transform.hpp
 * @brief Coordinate transform utilities: camera-to-world, scaling, and alignment.
 *
 * Responsibilities:
 * - Scale 3D points from SfM coordinate frame to world units.
 * - Transform points from one coordinate system to another via rotation + translation.
 * - Convert between different coordinate representations.
 *
 * Dependencies:
 * - Eigen for numerical types.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

/// --- Transform utilities ---

/// Scale a 3D point (or point cloud) by a uniform factor.
Eigen::Vector3d camera_to_world(const Eigen::Vector3d& point, double scale = 1.0);

/// Apply scale to a batch of points.
void scale_points(std::vector<Eigen::Vector3d>& points, double scale);

/// Transform a point by rotation and translation: out = R * point + t.
Eigen::Vector3d transform_point(const Eigen::Vector3d& point,
                                 const Eigen::Matrix3d& rotation,
                                 const Eigen::Vector3d& translation);

/// Transform a batch of points.
void transform_points(std::vector<Eigen::Vector3d>& points,
                       const Eigen::Matrix3d& rotation,
                       const Eigen::Vector3d& translation);

/// Compute a rotation that aligns direction `from` to direction `to`.
/// Uses Rodrigues' rotation formula via cross/axis-angle.
Eigen::Matrix3d align_vector(const Eigen::Vector3d& from,
                               const Eigen::Vector3d& to);

/// Compute the centroid of a point cloud (optionally masked).
Eigen::Vector3d compute_centroid(const std::vector<Eigen::Vector3d>& points,
                                  const std::vector<bool>& mask = {});

/// De-mean a point cloud: subtract centroid. Returns the centroid.
Eigen::Vector3d center_points(std::vector<Eigen::Vector3d>& points,
                               const std::vector<bool>& mask = {});

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_COORDINATE_TRANSFORM_HPP
