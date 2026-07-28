/**
 * @file strain_2d.hpp
 * @brief 2D strain computation from displacement gradient fields.
 *
 * Responsibilities:
 * - Compute infinitesimal strain (exx, eyy, exy) from displacement gradient.
 * - Compute Green-Lagrange strain for large deformations.
 * - Support both direct gradient input and node-based gradient computation
 *   (delegates to mesh/postprocess/strain for mesh-based DIC results).
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - dic/mesh/postprocess/strain.hpp for mesh-based path.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_2D_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_2D_HPP

#include <dic/postprocess/strain.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

/// --- Built-in direct 2D strain ---

/// Compute infinitesimal strain from displacement gradient.
/// grad_u = [[du/dx, du/dy], [dv/dx, dv/dy]]
Strain2D compute_strain_from_gradient(const Eigen::Matrix2d& grad_u);

/// Compute Green-Lagrange strain from displacement gradient.
/// E = 0.5 * (grad_u + grad_u^T + grad_u^T * grad_u)
Strain2D compute_green_lagrange_strain(const Eigen::Matrix2d& grad_u);

/// Compute finite strain from deformation gradient F.
/// F = I + grad_u; E = 0.5 * (F^T * F - I)
Strain2D compute_strain_from_deformation_gradient(const Eigen::Matrix2d& F);

/// Batch computation from displacement gradients at multiple points.
/// grad_u[i] = [[du/dx, du/dy], [dv/dx, dv/dy]] at point i.
std::vector<Strain2D> compute_strain_2d_batch(
    const std::vector<Eigen::Matrix2d>& displacement_gradients);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_2D_HPP
