/**
 * @file displacement_3d.hpp
 * @brief 3D displacement computation and rigid-body-motion removal utilities.
 *
 * Responsibilities:
 * - Compute displacement vectors between reference and deformed 3D point clouds.
 * - Remove rigid body motion via SVD-based optimal rotation alignment.
 * - Compute displacement norms for statistics.
 *
 * Dependencies:
 * - Eigen for numerical types.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_DISPLACEMENT_3D_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_DISPLACEMENT_3D_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

/// --- RBM removal result ---

struct RigidBodyTransform {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    bool valid = false;
};

/// --- Core functions ---

/// Compute displacement: deformed - reference.
Eigen::Vector3d compute_displacement(const Eigen::Vector3d& reference_point,
                                      const Eigen::Vector3d& deformed_point);

/// Compute displacement for a batch of points.
/// Returns (n x 3) displacements (row-major in output vector).
void compute_displacements(const std::vector<Eigen::Vector3d>& reference,
                            const std::vector<Eigen::Vector3d>& deformed,
                            std::vector<Eigen::Vector3d>& displacement_out);

/// Compute Euclidean norm of each displacement.
void compute_displacement_norms(const std::vector<Eigen::Vector3d>& displacements,
                                 std::vector<double>& norms_out);

/// Remove rigid body motion: find optimal SVD rotation + translation
/// to align `from` points to `to` points, using only points where
/// `valid_mask[i] == true`.
RigidBodyTransform find_rigid_body_transform(
    const std::vector<Eigen::Vector3d>& from,
    const std::vector<Eigen::Vector3d>& to,
    const std::vector<bool>& valid_mask);

/// Apply rigid body transform to a point cloud.
void apply_rigid_body_transform(
    std::vector<Eigen::Vector3d>& points,
    const RigidBodyTransform& transform);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_DISPLACEMENT_3D_HPP
