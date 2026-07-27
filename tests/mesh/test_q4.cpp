#include <gtest/gtest.h>

#include <dic/mesh/element/q4.hpp>

#include <cmath>

namespace {

// Tolerance for floating-point comparisons
constexpr double kEps = 1e-12;

} // namespace

// ---------------------------------------------------------------------------
// Partition of unity: sum of shape functions must equal 1
// ---------------------------------------------------------------------------
TEST(Q4Element, PartitionOfUnity)
{
    dic::Q4Element elem;
    const double test_points[][2] = {
        { 0.0,  0.0},   // center
        {-1.0, -1.0},   // corner 0
        { 1.0, -1.0},   // corner 1
        { 1.0,  1.0},   // corner 2
        {-1.0,  1.0},   // corner 3
        { 0.0, -1.0},   // mid-edge bottom
        { 1.0,  0.0},   // mid-edge right
        { 0.0,  1.0},   // mid-edge top
        {-1.0,  0.0},   // mid-edge left
        { 0.5,  0.3},   // arbitrary interior
        {-0.7, -0.2},   // arbitrary interior
    };

    for (const auto& pt : test_points) {
        Eigen::VectorXd N = elem.shape_functions(pt[0], pt[1]);
        double sum = N.sum();
        EXPECT_NEAR(sum, 1.0, kEps)
            << "Partition of unity failed at (xi=" << pt[0] << ", eta=" << pt[1] << ")";
        ASSERT_EQ(N.size(), 4);
    }
}

// ---------------------------------------------------------------------------
// Sum of derivative rows should be zero
// ---------------------------------------------------------------------------
TEST(Q4Element, DerivativeSumZero)
{
    dic::Q4Element elem;
    const double test_points[][2] = {
        { 0.0,  0.0},
        { 0.3, -0.5},
        {-0.8,  0.2},
    };

    for (const auto& pt : test_points) {
        Eigen::MatrixXd dN = elem.shape_function_derivatives(pt[0], pt[1]);
        ASSERT_EQ(dN.rows(), 4);
        ASSERT_EQ(dN.cols(), 2);

        double sum_xi = 0.0, sum_eta = 0.0;
        for (int i = 0; i < 4; ++i) {
            sum_xi  += dN(i, 0);
            sum_eta += dN(i, 1);
        }
        EXPECT_NEAR(sum_xi,  0.0, kEps);
        EXPECT_NEAR(sum_eta, 0.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// Kronecker delta at corner nodes
// ---------------------------------------------------------------------------
TEST(Q4Element, KroneckerDeltaAtCorners)
{
    dic::Q4Element elem;

    // Corner 0: (-1, -1) -> N0=1, others=0
    {
        Eigen::VectorXd N = elem.shape_functions(-1.0, -1.0);
        EXPECT_NEAR(N[0], 1.0, kEps);
        EXPECT_NEAR(N[1], 0.0, kEps);
        EXPECT_NEAR(N[2], 0.0, kEps);
        EXPECT_NEAR(N[3], 0.0, kEps);
    }
    // Corner 1: (1, -1)
    {
        Eigen::VectorXd N = elem.shape_functions(1.0, -1.0);
        EXPECT_NEAR(N[0], 0.0, kEps);
        EXPECT_NEAR(N[1], 1.0, kEps);
        EXPECT_NEAR(N[2], 0.0, kEps);
        EXPECT_NEAR(N[3], 0.0, kEps);
    }
    // Corner 2: (1, 1)
    {
        Eigen::VectorXd N = elem.shape_functions(1.0, 1.0);
        EXPECT_NEAR(N[0], 0.0, kEps);
        EXPECT_NEAR(N[1], 0.0, kEps);
        EXPECT_NEAR(N[2], 1.0, kEps);
        EXPECT_NEAR(N[3], 0.0, kEps);
    }
    // Corner 3: (-1, 1)
    {
        Eigen::VectorXd N = elem.shape_functions(-1.0, 1.0);
        EXPECT_NEAR(N[0], 0.0, kEps);
        EXPECT_NEAR(N[1], 0.0, kEps);
        EXPECT_NEAR(N[2], 0.0, kEps);
        EXPECT_NEAR(N[3], 1.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// Node count
// ---------------------------------------------------------------------------
TEST(Q4Element, NodeCount)
{
    dic::Q4Element elem;
    EXPECT_EQ(elem.node_count(), 4);
}
