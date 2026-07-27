#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_LINEAR_ALGEBRA_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_LINEAR_ALGEBRA_HPP

#include <Eigen/Dense>

namespace dic {

bool cholesky_in_place(Eigen::MatrixXd& matrix, double lambda = 1e-10);
void forward_substitution_in_place(Eigen::VectorXd& vector, const Eigen::MatrixXd& lower);
void backward_substitution_in_place(Eigen::VectorXd& vector, const Eigen::MatrixXd& lower);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_LINEAR_ALGEBRA_HPP
