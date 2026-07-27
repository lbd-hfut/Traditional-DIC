#include <dic/subset/solver/linear_algebra.hpp>
#include <gtest/gtest.h>

TEST(SubsetLinearAlgebra, CholeskyForwardBackwardSolve)
{
    Eigen::MatrixXd matrix(2, 2);
    matrix << 4.0, 2.0,
              2.0, 3.0;
    Eigen::VectorXd rhs(2);
    rhs << 6.0, 5.0;

    ASSERT_TRUE(dic::cholesky_in_place(matrix));
    dic::forward_substitution_in_place(rhs, matrix);
    dic::backward_substitution_in_place(rhs, matrix);

    EXPECT_NEAR(rhs(0), 1.0, 1e-12);
    EXPECT_NEAR(rhs(1), 1.0, 1e-12);
}
