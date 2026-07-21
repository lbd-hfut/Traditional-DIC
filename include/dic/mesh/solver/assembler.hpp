/**
 * @file assembler.hpp
 * @brief Sparse global Hessian/gradient assembler.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_ASSEMBLER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_ASSEMBLER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace dic {

class GlobalAssembler {
public:
    void reset(std::size_t dof_count);
    void add_element_contribution(const std::vector<int>& global_dofs, const Eigen::MatrixXd& local_hessian, const Eigen::VectorXd& local_gradient);
    Eigen::SparseMatrix<double> hessian() const;
    Eigen::VectorXd gradient() const;
private:
    std::size_t dof_count_{0};
    std::vector<Eigen::Triplet<double>> triplets_;
    Eigen::VectorXd gradient_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_SOLVER_ASSEMBLER_HPP
