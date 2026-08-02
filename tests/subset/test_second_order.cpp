#include <dic/subset/shape/second_order.hpp>
#include <gtest/gtest.h>

#include <Eigen/Dense>

TEST(SecondOrder, ParameterCount)
{
    const dic::SecondOrderShapeFunction shape;
    EXPECT_EQ(shape.parameter_count(), 12);
}

TEST(SecondOrder, JacobianShape)
{
    const dic::SecondOrderShapeFunction shape;
    const Eigen::Vector2d point(3.0, -2.0);
    const auto J = shape.jacobian(point);
    EXPECT_EQ(J.rows(), 2);
    EXPECT_EQ(J.cols(), 12);
}

TEST(SecondOrder, ZeroParametersGivesIdentity)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p = Eigen::VectorXd::Zero(12);
    const Eigen::Vector2d point(5.0, 7.0);
    const auto warped = shape.warp(point, p);
    EXPECT_DOUBLE_EQ(warped.x(), 5.0);
    EXPECT_DOUBLE_EQ(warped.y(), 7.0);
}

TEST(SecondOrder, PureTranslation)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p = Eigen::VectorXd::Zero(12);
    p(0) = 1.5;  // u
    p(1) = -0.7; // v

    const Eigen::Vector2d point(4.0, 3.0);
    const auto warped = shape.warp(point, p);
    EXPECT_DOUBLE_EQ(warped.x(), 5.5);   // 4 + 1.5
    EXPECT_DOUBLE_EQ(warped.y(), 2.3);   // 3 + (-0.7)
}

TEST(SecondOrder, FirstOrderAffine)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p = Eigen::VectorXd::Zero(12);
    p(0) = 0.5;  // u
    p(1) = 0.2;  // v
    p(2) = 0.1;  // du_dx
    p(3) = 0.05; // du_dy
    p(4) = -0.03;// dv_dx
    p(5) = 0.08; // dv_dy

    const Eigen::Vector2d point(10.0, 5.0);
    const auto warped = shape.warp(point, p);
    // x + u + du_dx*x + du_dy*y = 10 + 0.5 + 0.1*10 + 0.05*5 = 10 + 0.5 + 1.0 + 0.25 = 11.75
    EXPECT_DOUBLE_EQ(warped.x(), 11.75);
    // y + v + dv_dx*x + dv_dy*y = 5 + 0.2 + (-0.03)*10 + 0.08*5 = 5 + 0.2 - 0.3 + 0.4 = 5.3
    EXPECT_DOUBLE_EQ(warped.y(), 5.3);
}

TEST(SecondOrder, PureSecondOrderTerms)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p = Eigen::VectorXd::Zero(12);
    p(6) = 0.2;   // d2u_dx2
    p(7) = 0.1;   // d2u_dxdy
    p(8) = -0.05; // d2u_dy2
    p(9) = -0.1;  // d2v_dx2
    p(10) = 0.08; // d2v_dxdy
    p(11) = 0.04; // d2v_dy2

    const double x = 3.0;
    const double y = 2.0;
    const Eigen::Vector2d point(x, y);
    const auto warped = shape.warp(point, p);

    // warp_x = x + d2u_dx2 * x2/2 + d2u_dxdy * x*y + d2u_dy2 * y2/2
    //        = 3 + 0.2*9/2 + 0.1*6 + (-0.05)*4/2 = 3 + 0.9 + 0.6 - 0.1 = 4.4
    EXPECT_DOUBLE_EQ(warped.x(), 4.4);
    // warp_y = y + d2v_dx2 * x2/2 + d2v_dxdy * x*y + d2v_dy2 * y2/2
    //        = 2 + (-0.1)*9/2 + 0.08*6 + 0.04*4/2 = 2 - 0.45 + 0.48 + 0.08 = 2.11
    EXPECT_DOUBLE_EQ(warped.y(), 2.11);
}

TEST(SecondOrder, JacobianEntriesAtTestPoint)
{
    const dic::SecondOrderShapeFunction shape;
    const double x = 2.0;
    const double y = 3.0;
    const Eigen::Vector2d point(x, y);
    const auto J = shape.jacobian(point);

    // Row 0 entries
    EXPECT_DOUBLE_EQ(J(0, 0), 1.0);     // dwarp_x / du
    EXPECT_DOUBLE_EQ(J(0, 1), 0.0);     // dwarp_x / dv
    EXPECT_DOUBLE_EQ(J(0, 2), x);       // dwarp_x / du_dx = 2.0
    EXPECT_DOUBLE_EQ(J(0, 3), y);       // dwarp_x / du_dy = 3.0
    EXPECT_DOUBLE_EQ(J(0, 4), 0.0);     // dwarp_x / dv_dx
    EXPECT_DOUBLE_EQ(J(0, 5), 0.0);     // dwarp_x / dv_dy
    EXPECT_DOUBLE_EQ(J(0, 6), x * x * 0.5);   // dwarp_x / d2u_dx2 = 2.0
    EXPECT_DOUBLE_EQ(J(0, 7), x * y);         // dwarp_x / d2u_dxdy = 6.0
    EXPECT_DOUBLE_EQ(J(0, 8), y * y * 0.5);   // dwarp_x / d2u_dy2 = 4.5
    EXPECT_DOUBLE_EQ(J(0, 9), 0.0);     // dwarp_x / d2v_dx2
    EXPECT_DOUBLE_EQ(J(0, 10), 0.0);    // dwarp_x / d2v_dxdy
    EXPECT_DOUBLE_EQ(J(0, 11), 0.0);    // dwarp_x / d2v_dy2

    // Row 1 entries
    EXPECT_DOUBLE_EQ(J(1, 0), 0.0);     // dwarp_y / du
    EXPECT_DOUBLE_EQ(J(1, 1), 1.0);     // dwarp_y / dv
    EXPECT_DOUBLE_EQ(J(1, 2), 0.0);     // dwarp_y / du_dx
    EXPECT_DOUBLE_EQ(J(1, 3), 0.0);     // dwarp_y / du_dy
    EXPECT_DOUBLE_EQ(J(1, 4), x);       // dwarp_y / dv_dx = 2.0
    EXPECT_DOUBLE_EQ(J(1, 5), y);       // dwarp_y / dv_dy = 3.0
    EXPECT_DOUBLE_EQ(J(1, 6), 0.0);     // dwarp_y / d2u_dx2
    EXPECT_DOUBLE_EQ(J(1, 7), 0.0);     // dwarp_y / d2u_dxdy
    EXPECT_DOUBLE_EQ(J(1, 8), 0.0);     // dwarp_y / d2u_dy2
    EXPECT_DOUBLE_EQ(J(1, 9), x * x * 0.5);   // dwarp_y / d2v_dx2 = 2.0
    EXPECT_DOUBLE_EQ(J(1, 10), x * y);        // dwarp_y / d2v_dxdy = 6.0
    EXPECT_DOUBLE_EQ(J(1, 11), y * y * 0.5);  // dwarp_y / d2v_dy2 = 4.5
}

TEST(SecondOrder, FullParametersWarp)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p(12);
    p << 0.1, -0.2,          // u, v
         0.01, 0.02,         // du_dx, du_dy
         0.03, 0.04,         // dv_dx, dv_dy
         0.001, 0.002,       // d2u_dx2, d2u_dxdy
         -0.001,             // d2u_dy2
         0.003, -0.002,      // d2v_dx2, d2v_dxdy
         0.004;              // d2v_dy2

    const double x = 4.0;
    const double y = 5.0;
    const Eigen::Vector2d point(x, y);
    const auto warped = shape.warp(point, p);

    const double expected_x = x + p(0) + p(2) * x + p(3) * y
                              + p(6) * x * x * 0.5 + p(7) * x * y + p(8) * y * y * 0.5;
    const double expected_y = y + p(1) + p(4) * x + p(5) * y
                              + p(9) * x * x * 0.5 + p(10) * x * y + p(11) * y * y * 0.5;

    EXPECT_DOUBLE_EQ(warped.x(), expected_x);
    EXPECT_DOUBLE_EQ(warped.y(), expected_y);
}

TEST(SecondOrder, InsufficientParametersThrows)
{
    const dic::SecondOrderShapeFunction shape;
    Eigen::VectorXd p(6); // only 6 params, need 12
    const Eigen::Vector2d point(0.0, 0.0);
    EXPECT_THROW(shape.warp(point, p), std::invalid_argument);
}
