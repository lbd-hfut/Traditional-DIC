#include <gtest/gtest.h>

#include <dic/mesh/element/q8.hpp>

#include <cmath>

namespace {

constexpr double kEps = 1e-12;

} // namespace

// ---------------------------------------------------------------------------
// Partition of unity
// ---------------------------------------------------------------------------
TEST(Q8Element, PartitionOfUnity)
{
    dic::Q8Element elem;
    const double test_points[][2] = {
        { 0.0,  0.0},   // center
        {-1.0, -1.0},   // corner 0
        { 1.0, -1.0},   // corner 1
        { 1.0,  1.0},   // corner 2
        {-1.0,  1.0},   // corner 3
        { 0.0, -1.0},   // mid-side 4
        { 1.0,  0.0},   // mid-side 5
        { 0.0,  1.0},   // mid-side 6
        {-1.0,  0.0},   // mid-side 7
        { 0.3,  0.7},   // arbitrary
        {-0.5, -0.2},   // arbitrary
    };

    for (const auto& pt : test_points) {
        Eigen::VectorXd N = elem.shape_functions(pt[0], pt[1]);
        double sum = N.sum();
        EXPECT_NEAR(sum, 1.0, kEps)
            << "Partition of unity failed at (xi=" << pt[0] << ", eta=" << pt[1] << ")";
        ASSERT_EQ(N.size(), 8);
    }
}

// ---------------------------------------------------------------------------
// Derivative sum zero
// ---------------------------------------------------------------------------
TEST(Q8Element, DerivativeSumZero)
{
    dic::Q8Element elem;
    const double test_points[][2] = {
        { 0.0,  0.0},
        { 0.3, -0.5},
        {-0.8,  0.2},
    };

    for (const auto& pt : test_points) {
        Eigen::MatrixXd dN = elem.shape_function_derivatives(pt[0], pt[1]);
        ASSERT_EQ(dN.rows(), 8);
        ASSERT_EQ(dN.cols(), 2);

        double sum_xi = 0.0, sum_eta = 0.0;
        for (int i = 0; i < 8; ++i) {
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
TEST(Q8Element, KroneckerDeltaAtCorners)
{
    dic::Q8Element elem;

    auto check_corner = [&](double xi, double eta, int expected_idx) {
        Eigen::VectorXd N = elem.shape_functions(xi, eta);
        for (int i = 0; i < 8; ++i) {
            if (i == expected_idx) {
                EXPECT_NEAR(N[i], 1.0, kEps);
            } else {
                EXPECT_NEAR(N[i], 0.0, kEps);
            }
        }
    };

    check_corner(-1.0, -1.0, 0);
    check_corner( 1.0, -1.0, 1);
    check_corner( 1.0,  1.0, 2);
    check_corner(-1.0,  1.0, 3);
}

// ---------------------------------------------------------------------------
// Kronecker delta at mid-side nodes
// ---------------------------------------------------------------------------
TEST(Q8Element, KroneckerDeltaAtMidSides)
{
    dic::Q8Element elem;

    auto check_mid = [&](double xi, double eta, int expected_idx) {
        Eigen::VectorXd N = elem.shape_functions(xi, eta);
        for (int i = 0; i < 8; ++i) {
            if (i == expected_idx) {
                EXPECT_NEAR(N[i], 1.0, kEps);
            } else {
                EXPECT_NEAR(N[i], 0.0, kEps);
            }
        }
    };

    check_mid( 0.0, -1.0, 4);   // bottom mid-side
    check_mid( 1.0,  0.0, 5);   // right mid-side
    check_mid( 0.0,  1.0, 6);   // top mid-side
    check_mid(-1.0,  0.0, 7);   // left mid-side
}

// ---------------------------------------------------------------------------
// Node count
// ---------------------------------------------------------------------------
TEST(Q8Element, NodeCount)
{
    dic::Q8Element elem;
    EXPECT_EQ(elem.node_count(), 8);
}
