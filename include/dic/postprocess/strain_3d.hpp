/**
 * @file strain_3d.hpp
 * @brief 3D surface strain computation from triangle face point pairs.
 *
 * Responsibilities:
 * - Compute deformation gradient F and right Cauchy-Green C from reference
 *   and deformed point coordinates of each triangular face.
 * - Derive Green-Lagrange E and Euler-Almansi e strain tensors per face.
 * - Compute principal strains, equivalent (von Mises) strain, max shear,
 *   and stretch ratios from the planar projection of each tensor.
 *
 * Algorithm:
 *   For each face (X1,X2,X3) -> (x1,x2,x3):
 *     D1 = X2-X1, D2 = X3-X1   (reference edges)
 *     d1 = x2-x1, d2 = x3-x1   (deformed edges)
 *     D3 = (D1 x D2)/|D1 x D2| (reference unit normal)
 *     d3 = (d1 x d2)/|d1 x d2| (deformed unit normal)
 *     F = d1*Drec1^T + d2*Drec2^T + d3*D3^T   where Drec1, Drec2 are
 *         reciprocal basis vectors to (D1, D2, D3)
 *     C = F^T * F
 *     E = 0.5 * (C - I)   (Green-Lagrange)
 *     B = F * F^T
 *     e = 0.5 * (I - B^{-1})  (Euler-Almansi)
 *
 * Dependencies:
 * - Eigen for numerical types.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP

#include <dic/postprocess/strain.hpp>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cstdint>

namespace dic {

/// --- Per-face strain result ---

struct SurfaceStrain3D {
    // Deformation gradient (3x3, row-major)
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    // Right Cauchy-Green tensor
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    // Determinant of F (volume ratio)
    double J = 0.0;
    // Green-Lagrange strain tensor
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    // Euler-Almansi strain tensor
    Eigen::Matrix3d e = Eigen::Matrix3d::Zero();
    // Frobenius norm of E
    double Emgn = 0.0;
    // Frobenius norm of e
    double emgn = 0.0;
    // Green-Lagrange principal strains (planar, sorted)
    double Epc1 = 0.0;
    double Epc2 = 0.0;
    // Euler-Almansi principal strains (planar, sorted)
    double epc1 = 0.0;
    double epc2 = 0.0;
    // Max shear from Green-Lagrange
    double EShearMax = 0.0;
    // Max shear from Euler-Almansi
    double eShearMax = 0.0;
    // Equivalent (von Mises) strain from Green-Lagrange
    double Eeq = 0.0;
    // Equivalent strain from Euler-Almansi
    double eeq = 0.0;
    // Reference face area
    double area = 0.0;
    // Deformed face normal (unit vector)
    Eigen::Vector3d d3 = Eigen::Vector3d::Zero();
    // Stretch ratios (planar, sorted)
    double Lambda1 = 0.0;
    double Lambda2 = 0.0;
    bool valid = false;
};

/// --- Compute surface strain for a batch of faces ---

/// Compute per-face 3D strain tensors from triangular faces.
///
/// @param faces       (n_faces x 3) vertex indices (0-based) into point arrays.
/// @param points_ref  (n_points x 3) reference point coordinates.
/// @param points_def  (n_points x 3) deformed point coordinates.
/// @param valid_faces (n_faces) mask: if false, the face is skipped.
/// @return Vector of per-face strain results.
std::vector<SurfaceStrain3D> compute_surface_strain(
    const std::vector<std::array<int, 3>>& faces,
    const std::vector<Eigen::Vector3d>& points_ref,
    const std::vector<Eigen::Vector3d>& points_def,
    const std::vector<bool>& valid_faces);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP
