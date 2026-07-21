/**
 * @file global_gauss_newton.hpp
 * @brief Forward global Gauss-Newton Mesh-DIC skeleton.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP

#include <dic/core/image.hpp>
#include <dic/mesh/mesh.hpp>

namespace dic {

class GlobalGaussNewton {
public:
    bool solve(const Image& reference, const Image& deformed, Mesh& mesh);
private:
    // TODO: Mesh initialization -> iterate elements -> sampling points ->
    // natural coordinates -> shape functions -> displacement interpolation ->
    // image warp -> deformed image interpolation -> image gradient -> residual
    // -> local Jacobian/Hessian -> global sparse assembly -> sparse solve ->
    // nodal displacement update -> convergence. Default path is forward
    // Gauss-Newton, not Mesh IC-GN.
    void initialize_mesh_displacement(Mesh& mesh);
    void assemble_system(const Image& reference, const Image& deformed, const Mesh& mesh);
    Eigen::VectorXd solve_increment();
    void update_nodal_displacement(Mesh& mesh, const Eigen::VectorXd& increment);
    bool check_convergence(const Eigen::VectorXd& increment) const;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_GAUSS_NEWTON_HPP
