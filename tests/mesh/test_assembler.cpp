#include <gtest/gtest.h>

// Verify that the FEM assembler module links correctly.
// Full integration tests for stiffness assembly will come with the
// solver tests (Step 5-6) where a real G2L mapping and mesh are set up.

#include <dic/mesh/solver/assembler.hpp>

TEST(FEMAssembler, GlobalAssemblerWorks) {
    // The existing GlobalAssembler (triplet-based) should still work
    dic::GlobalAssembler ga;
    ga.reset(4);

    std::vector<int> dofs = {0, 1, 2, 3};
    Eigen::MatrixXd H(4, 4);
    H << 2, 0, 0, 0,
         0, 2, 0, 0,
         0, 0, 2, 0,
         0, 0, 0, 2;
    Eigen::VectorXd g(4);
    g << 1, 2, 3, 4;

    ga.add_element_contribution(dofs, H, g);

    auto A = ga.hessian();
    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);

    auto grad = ga.gradient();
    EXPECT_NEAR(grad[0], 1.0, 1e-12);
    EXPECT_NEAR(grad[1], 2.0, 1e-12);
    EXPECT_NEAR(grad[2], 3.0, 1e-12);
    EXPECT_NEAR(grad[3], 4.0, 1e-12);
}
