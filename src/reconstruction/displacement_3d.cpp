/**
 * @file displacement_3d.cpp
 * @brief 3D displacement and RBM removal implementation.
 */

#include <dic/reconstruction/displacement_3d.hpp>

#include <algorithm>
#include <cmath>

namespace dic {

// ---------------------------------------------------------------------------
// Single displacement
// ---------------------------------------------------------------------------

Eigen::Vector3d compute_displacement(const Eigen::Vector3d& reference_point,
                                      const Eigen::Vector3d& deformed_point)
{
    return deformed_point - reference_point;
}

// ---------------------------------------------------------------------------
// Batch displacements
// ---------------------------------------------------------------------------

void compute_displacements(const std::vector<Eigen::Vector3d>& reference,
                            const std::vector<Eigen::Vector3d>& deformed,
                            std::vector<Eigen::Vector3d>& displacement_out)
{
    if (reference.size() != deformed.size()) return;
    displacement_out.resize(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i)
        displacement_out[i] = deformed[i] - reference[i];
}

// ---------------------------------------------------------------------------
// Displacement norms
// ---------------------------------------------------------------------------

void compute_displacement_norms(const std::vector<Eigen::Vector3d>& displacements,
                                 std::vector<double>& norms_out)
{
    norms_out.resize(displacements.size());
    for (std::size_t i = 0; i < displacements.size(); ++i)
        norms_out[i] = displacements[i].norm();
}

// ---------------------------------------------------------------------------
// RBM: find optimal rigid body transform (SVD-based)
// ---------------------------------------------------------------------------

RigidBodyTransform find_rigid_body_transform(
    const std::vector<Eigen::Vector3d>& from,
    const std::vector<Eigen::Vector3d>& to,
    const std::vector<bool>& valid_mask)
{
    RigidBodyTransform result;
    if (from.size() != to.size() || from.size() != valid_mask.size())
        return result;

    // Compute centroids of valid points
    Eigen::Vector3d centroid_from = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid_to   = Eigen::Vector3d::Zero();
    std::size_t n = 0;
    for (std::size_t i = 0; i < from.size(); ++i) {
        if (valid_mask[i] && from[i].allFinite() && to[i].allFinite()) {
            centroid_from += from[i];
            centroid_to   += to[i];
            ++n;
        }
    }
    if (n < 3) return result;

    centroid_from /= static_cast<double>(n);
    centroid_to   /= static_cast<double>(n);

    // Build cross-covariance H = sum(d_from_i * d_to_i^T)
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < from.size(); ++i) {
        if (valid_mask[i] && from[i].allFinite() && to[i].allFinite()) {
            Eigen::Vector3d da = from[i] - centroid_from;
            Eigen::Vector3d db = to[i]   - centroid_to;
            H += da * db.transpose();
        }
    }

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();

    // Ensure det(R) = +1 (proper rotation)
    if (R.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }

    result.rotation = R;
    result.translation = centroid_to - R * centroid_from;
    result.valid = true;
    return result;
}

// ---------------------------------------------------------------------------
// Apply RBM transform to point cloud
// ---------------------------------------------------------------------------

void apply_rigid_body_transform(
    std::vector<Eigen::Vector3d>& points,
    const RigidBodyTransform& transform)
{
    if (!transform.valid) return;
    for (auto& pt : points) {
        pt = transform.rotation * pt + transform.translation;
    }
}

} // namespace dic
