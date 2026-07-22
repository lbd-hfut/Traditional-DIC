/**
 * @file global_icgn.hpp
 * @brief Inverse-compositional global IC-GN solver skeleton for 2D Mesh-DIC.
 *
 * Responsibilities:
 * - Define the FE-based global inverse-compositional Gauss-Newton solver interface.
 * - Separate Mesh-DIC IC-GN orchestration from sparse assembly and linear solving.
 * - Reserve constant-Hessian/reference-gradient precomputation for Q4/T3/Q8 elements.
 *
 * Inputs:
 * - Reference/deformed grayscale images, a 2D finite-element mesh, and solver settings.
 *
 * Outputs:
 * - Updated nodal displacement stored on Mesh nodes and a convergence flag.
 *
 * Dependencies:
 * - Eigen for dense/sparse numerical data.
 * - dic::Image and dic::Mesh for image and finite-element data.
 * - GlobalAssembler and LinearSolver for sparse system assembly/solution.
 *
 * TODO:
 * - Implement FE Global IC-GN following the fixed reference-gradient workflow:
 *   reference gradient -> shape functions -> constant Hessian/stiffness ->
 *   deformed image warp -> residual/right-hand side -> sparse solve -> nodal update.
 * - Add optional displacement regularization terms such as Laplacian or elasticity.
 * - Validate against synthetic Q4/T3/Q8 displacement fields and published FE-DIC examples.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_ICGN_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_ICGN_HPP

#include <dic/core/image.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <dic/mesh/solver/assembler.hpp>
#include <dic/mesh/solver/linear_solver.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace dic {

class GlobalICGN {
public:
    explicit GlobalICGN(MeshConfig config = {});

    bool solve(
        const Image& reference,
        const Image& deformed,
        Mesh& mesh
    );

private:
    struct ReferencePrecompute {
        Eigen::MatrixXd gradient_x;
        Eigen::MatrixXd gradient_y;
        Eigen::SparseMatrix<double> constant_hessian;
        bool valid{false};
    };

    ReferencePrecompute precompute_reference_terms(
        const Image& reference,
        const Mesh& mesh
    ) const;

    void assemble_constant_hessian(
        const Image& reference,
        const Mesh& mesh,
        ReferencePrecompute& precompute
    ) const;

    Eigen::VectorXd assemble_residual_rhs(
        const Image& reference,
        const Image& deformed,
        const Mesh& mesh,
        const ReferencePrecompute& precompute
    ) const;

    Eigen::VectorXd solve_increment(
        const Eigen::SparseMatrix<double>& constant_hessian,
        const Eigen::VectorXd& rhs
    ) const;

    void update_nodal_displacement(
        Mesh& mesh,
        const Eigen::VectorXd& increment
    ) const;

    bool check_convergence(
        const Eigen::VectorXd& increment,
        int iteration
    ) const;

    MeshConfig config_;
    LinearSolver linear_solver_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_GLOBAL_ICGN_HPP
