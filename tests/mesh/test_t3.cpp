#include <gtest/gtest.h>

#include <dic/mesh/element/t3.hpp>

#include <cmath>

namespace {

constexpr double kEps = 1e-12;

} // namespace

// ---------------------------------------------------------------------------
// Partition of unity
// ---------------------------------------------------------------------------
TEST(T3Element, PartitionOfUnity)
{
    dic::T3Element elem;
    const double test_points[][2] = {
        {0.0, 0.0},        // node 0
        {1.0, 0.0},        // node 1
        {0.0, 1.0},        // node 2
        {1.0/3.0, 1.0/3.0},// centroid
        {0.5, 0.2},        // arbitrary
        {0.1, 0.8},        // arbitrary
    };

    for (const auto& pt : test_points) {
        Eigen::VectorXd N = elem.shape_functions(pt[0], pt[1]);
        double sum = N.sum();
        EXPECT_NEAR(sum, 1.0, kEps)
            << "Partition of unity failed at (xi=" << pt[0] << ", eta=" << pt[1] << ")";
        ASSERT_EQ(N.size(), 3);
    }
}

// ---------------------------------------------------------------------------
// Derivative sum zero
// ---------------------------------------------------------------------------
TEST(T3Element, DerivativeSumZero)
{
    dic::T3Element elem;
    const double test_points[][2] = {
        {1.0/3.0, 1.0/3.0},
        {0.5, 0.2},
        {0.1, 0.8},
    };

    for (const auto& pt : test_points) {
        Eigen::MatrixXd dN = elem.shape_function_derivatives(pt[0], pt[1]);
        ASSERT_EQ(dN.rows(), 3);
        ASSERT_EQ(dN.cols(), 2);

        double sum_xi = 0.0, sum_eta = 0.0;
        for (int i = 0; i < 3; ++i) {
            sum_xi  += dN(i, 0);
            sum_eta += dN(i, 1);
        }
        EXPECT_NEAR(sum_xi,  0.0, kEps);
        EXPECT_NEAR(sum_eta, 0.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// Kronecker delta at nodes
// ---------------------------------------------------------------------------
TEST(T3Element, KroneckerDeltaAtNodes)
{
    dic::T3Element elem;

    // Node 0: (0, 0)
    {
        Eigen::VectorXd N = elem.shape_functions(0.0, 0.0);
        EXPECT_NEAR(N[0], 1.0, kEps);
        EXPECT_NEAR(N[1], 0.0, kEps);
        EXPECT_NEAR(N[2], 0.0, kEps);
    }
    // Node 1: (1, 0)
    {
        Eigen::VectorXd N = elem.shape_functions(1.0, 0.0);
        EXPECT_NEAR(N[0], 0.0, kEps);
        EXPECT_NEAR(N[1], 1.0, kEps);
        EXPECT_NEAR(N[2], 0.0, kEps);
    }
    // Node 2: (0, 1)
    {
        Eigen::VectorXd N = elem.shape_functions(0.0, 1.0);
        EXPECT_NEAR(N[0], 0.0, kEps);
        EXPECT_NEAR(N[1], 0.0, kEps);
        EXPECT_NEAR(N[2], 1.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// Centroid values: all shape functions should equal 1/3
// ---------------------------------------------------------------------------
TEST(T3Element, CentroidValues)
{
    dic::T3Element elem;
    Eigen::VectorXd N = elem.shape_functions(1.0/3.0, 1.0/3.0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(N[i], 1.0/3.0, kEps);
    }
}

// ---------------------------------------------------------------------------
// Node count
// ---------------------------------------------------------------------------
TEST(T3Element, NodeCount)
{
    dic::T3Element elem;
    EXPECT_EQ(elem.node_count(), 3);
}
