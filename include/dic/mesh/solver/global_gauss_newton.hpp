/**
 * @file global_gauss_newton.hpp
 * @brief Forward-additive global Gauss-Newton Mesh-DIC solver skeleton.
 *
 * Responsibilities:
 * - Define the forward-additive global Gauss-Newton solver interface.
 * - Keep the reassembled-Hessian path separate from GlobalICGN.
 *
 * Inputs:
 * - Reference/deformed images, Mesh topology, nodal displacement state, and solver settings.
 *
 * Outputs:
 * - Updated nodal displacement state and convergence flag.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement per-iteration Hessian/Jacobian assembly using deformed image gradients.
 * - Add tests comparing forward GN and GlobalICGN convergence behavior.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP

#include <dic/core/image.hpp>
#include <dic/mesh/mesh.hpp>
#include <Eigen/Dense>

namespace dic {

class GlobalGaussNewton {
public:
    bool solve(const Image& reference, const Image& deformed, Mesh& mesh);
private:
    // TODO: Mesh initialization -> iterate elements -> sampling points ->
    // natural coordinates -> shape functions -> displacement interpolation ->
    // image warp -> deformed image interpolation -> deformed image gradient ->
    // residual -> per-iteration local Jacobian/Hessian -> global sparse
    // assembly -> sparse solve -> nodal displacement update -> convergence.
    void initialize_mesh_displacement(Mesh& mesh);
    void assemble_system(const Image& reference, const Image& deformed, const Mesh& mesh);
    Eigen::VectorXd solve_increment();
    void update_nodal_displacement(Mesh& mesh, const Eigen::VectorXd& increment);
    bool check_convergence(const Eigen::VectorXd& increment) const;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP
