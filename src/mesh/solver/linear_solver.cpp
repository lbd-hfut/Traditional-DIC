/**
 * @file linear_solver.cpp
 * @brief Minimal implementation placeholder for linear solver.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/mesh/solver/linear_solver.hpp>
#include <Eigen/SparseCholesky>
#include <stdexcept>

namespace dic {

Eigen::VectorXd LinearSolver::solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) const { Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(A); if (solver.info() != Eigen::Success) { throw std::runtime_error("Sparse factorization failed."); } return solver.solve(b); }

} // namespace dic
