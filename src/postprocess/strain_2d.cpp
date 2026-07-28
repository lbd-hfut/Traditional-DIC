/**
 * @file strain_2d.cpp
 * @brief 2D strain computation implementation.
 */

#include <dic/postprocess/strain_2d.hpp>

#include <cmath>

namespace dic {

// ---------------------------------------------------------------------------
// Infinitesimal strain: ε = 0.5 * (grad_u + grad_u^T)
// ---------------------------------------------------------------------------

Strain2D compute_strain_from_gradient(const Eigen::Matrix2d& grad_u)
{
    Strain2D result;
    if (!grad_u.allFinite()) return result;

    result.exx = grad_u(0, 0);
    result.eyy = grad_u(1, 1);
    result.exy = 0.5 * (grad_u(0, 1) + grad_u(1, 0));
    result.valid = true;
    return result;
}

// ---------------------------------------------------------------------------
// Green-Lagrange strain: E = 0.5 * (grad_u + grad_u^T + grad_u^T * grad_u)
// ---------------------------------------------------------------------------

Strain2D compute_green_lagrange_strain(const Eigen::Matrix2d& grad_u)
{
    Strain2D result;
    if (!grad_u.allFinite()) return result;

    Eigen::Matrix2d E = 0.5 * (grad_u + grad_u.transpose() +
                                grad_u.transpose() * grad_u);
    result.exx = E(0, 0);
    result.eyy = E(1, 1);
    result.exy = E(0, 1);
    result.valid = true;
    return result;
}

// ---------------------------------------------------------------------------
// Strain from deformation gradient: F = I + grad_u, E = 0.5*(F^T*F - I)
// ---------------------------------------------------------------------------

Strain2D compute_strain_from_deformation_gradient(const Eigen::Matrix2d& F)
{
    Strain2D result;
    if (!F.allFinite()) return result;

    Eigen::Matrix2d C = F.transpose() * F;
    Eigen::Matrix2d E = 0.5 * (C - Eigen::Matrix2d::Identity());
    result.exx = E(0, 0);
    result.eyy = E(1, 1);
    result.exy = E(0, 1);
    result.valid = true;
    return result;
}

// ---------------------------------------------------------------------------
// Batch computation
// ---------------------------------------------------------------------------

std::vector<Strain2D> compute_strain_2d_batch(
    const std::vector<Eigen::Matrix2d>& displacement_gradients)
{
    std::vector<Strain2D> result;
    result.reserve(displacement_gradients.size());
    for (const auto& grad_u : displacement_gradients)
        result.push_back(compute_strain_from_gradient(grad_u));
    return result;
}

} // namespace dic
