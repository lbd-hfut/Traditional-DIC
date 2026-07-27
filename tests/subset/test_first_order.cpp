#include <dic/subset/shape/first_order.hpp>
#include <gtest/gtest.h>

TEST(FirstOrder, WarpsWithAffineParameters)
{
    const dic::FirstOrderShapeFunction shape;
    Eigen::VectorXd parameters(6);
    parameters << 2.0, -1.0, 0.1, 0.2, -0.3, 0.4;

    const auto warped = shape.warp(Eigen::Vector2d(3.0, 5.0), parameters);

    EXPECT_DOUBLE_EQ(warped.x(), 3.0 + 2.0 + 0.1 * 3.0 + 0.2 * 5.0);
    EXPECT_DOUBLE_EQ(warped.y(), 5.0 - 1.0 - 0.3 * 3.0 + 0.4 * 5.0);
}

TEST(FirstOrder, ProvidesSubsetDICParameterJacobian)
{
    const dic::FirstOrderShapeFunction shape;
    const auto jacobian = shape.jacobian(Eigen::Vector2d(3.0, 5.0));

    ASSERT_EQ(jacobian.rows(), 2);
    ASSERT_EQ(jacobian.cols(), 6);
    EXPECT_DOUBLE_EQ(jacobian(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(jacobian(0, 3), 5.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 4), 3.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 5), 5.0);
}
