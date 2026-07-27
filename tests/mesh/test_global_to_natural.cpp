#include <gtest/gtest.h>

#include <dic/mesh/coordinate/global_to_natural.hpp>
#include <dic/mesh/element/q4.hpp>
#include <dic/mesh/element/q8.hpp>
#include <dic/mesh/element/t3.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr double kEps = 1e-9;

// 辅助：用形函数 + 节点坐标计算物理坐标，用于正向映射
Eigen::Vector2d forward_map(const dic::Element& elem,
                            const std::vector<dic::Node>& nodes,
                            double xi, double eta) {
    Eigen::VectorXd N = elem.shape_functions(xi, eta);
    double x = 0.0, y = 0.0;
    int nn = elem.node_count();
    for (int i = 0; i < nn; ++i) {
        x += N[i] * nodes[static_cast<std::size_t>(i)].coordinate.x();
        y += N[i] * nodes[static_cast<std::size_t>(i)].coordinate.y();
    }
    return Eigen::Vector2d(x, y);
}

// ---- T3 ----

std::vector<dic::Node> make_t3_nodes(
    double x1, double y1, double x2, double y2, double x3, double y3) {
    std::vector<dic::Node> nodes(3);
    nodes[0].coordinate = Eigen::Vector2d(x1, y1);
    nodes[1].coordinate = Eigen::Vector2d(x2, y2);
    nodes[2].coordinate = Eigen::Vector2d(x3, y3);
    return nodes;
}

TEST(G2L_T3, Node0MapsToOrigin) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(10.0, 20.0, 30.0, 20.0, 10.0, 50.0);
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(10.0, 20.0));
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  0.0, kEps);
    EXPECT_NEAR(result.eta, 0.0, kEps);
}

TEST(G2L_T3, Node1MapsTo10) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(10.0, 20.0, 30.0, 20.0, 10.0, 50.0);
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(30.0, 20.0));
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  1.0, kEps);
    EXPECT_NEAR(result.eta, 0.0, kEps);
}

TEST(G2L_T3, Node2MapsTo01) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(10.0, 20.0, 30.0, 20.0, 10.0, 50.0);
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(10.0, 50.0));
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  0.0, kEps);
    EXPECT_NEAR(result.eta, 1.0, kEps);
}

TEST(G2L_T3, CentroidRoundTrip) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(0.0, 0.0, 6.0, 0.0, 0.0, 6.0);
    double xi_in = 1.0 / 3.0, eta_in = 1.0 / 3.0;
    Eigen::Vector2d phys = forward_map(elem, nodes, xi_in, eta_in);
    auto result = dic::global_to_natural(elem, nodes, phys);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  xi_in,  kEps);
    EXPECT_NEAR(result.eta, eta_in, kEps);
}

TEST(G2L_T3, ArbitraryPointRoundTrip) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(10.0, 5.0, 50.0, 5.0, 10.0, 45.0);
    const double test_pts[][2] = {
        {0.1, 0.1}, {0.5, 0.3}, {0.7, 0.1}, {0.3, 0.6}, {0.05, 0.9}
    };
    for (const auto& pt : test_pts) {
        Eigen::Vector2d phys = forward_map(elem, nodes, pt[0], pt[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.xi,  pt[0], kEps) << "ξ at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.eta, pt[1], kEps) << "η at (" << pt[0] << ", " << pt[1] << ")";
    }
}

TEST(G2L_T3, OutsidePointFails) {
    dic::T3Element elem;
    auto nodes = make_t3_nodes(0.0, 0.0, 10.0, 0.0, 0.0, 10.0);
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(20.0, 5.0));
    EXPECT_FALSE(result.converged);
}

// ---- Q4 ----

std::vector<dic::Node> make_q4_nodes(
    double x0, double y0, double x1, double y1,
    double x2, double y2, double x3, double y3) {
    std::vector<dic::Node> nodes(4);
    nodes[0].coordinate = Eigen::Vector2d(x0, y0);
    nodes[1].coordinate = Eigen::Vector2d(x1, y1);
    nodes[2].coordinate = Eigen::Vector2d(x2, y2);
    nodes[3].coordinate = Eigen::Vector2d(x3, y3);
    return nodes;
}

TEST(G2L_Q4, CornersRoundTrip) {
    dic::Q4Element elem;
    // 规则矩形
    auto nodes = make_q4_nodes(0.0, 0.0, 10.0, 0.0, 10.0, 8.0, 0.0, 8.0);
    const double corners[][2] = {
        {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}
    };
    for (const auto& c : corners) {
        Eigen::Vector2d phys = forward_map(elem, nodes, c[0], c[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.xi,  c[0], kEps);
        EXPECT_NEAR(result.eta, c[1], kEps);
    }
}

TEST(G2L_Q4, CenterRoundTrip) {
    dic::Q4Element elem;
    auto nodes = make_q4_nodes(0.0, 0.0, 10.0, 0.0, 10.0, 8.0, 0.0, 8.0);
    Eigen::Vector2d phys = forward_map(elem, nodes, 0.0, 0.0);
    auto result = dic::global_to_natural(elem, nodes, phys);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  0.0, kEps);
    EXPECT_NEAR(result.eta, 0.0, kEps);
}

TEST(G2L_Q4, ArbitraryPointRoundTrip) {
    dic::Q4Element elem;
    auto nodes = make_q4_nodes(0.0, 0.0, 10.0, 0.0, 10.0, 8.0, 0.0, 8.0);
    const double test_pts[][2] = {
        {0.2, -0.7}, {-0.5, 0.3}, {0.7, 0.5}, {-0.3, -0.9}, {0.0, 0.8}
    };
    for (const auto& pt : test_pts) {
        Eigen::Vector2d phys = forward_map(elem, nodes, pt[0], pt[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.xi,  pt[0], kEps) << "ξ";
        EXPECT_NEAR(result.eta, pt[1], kEps) << "η";
    }
}

TEST(G2L_Q4, DistortedQuadRoundTrip) {
    dic::Q4Element elem;
    // 畸变四边形（非平行四边形）
    auto nodes = make_q4_nodes(2.0, 1.0, 12.0, 3.0, 10.0, 10.0, 0.0, 9.0);
    const double test_pts[][2] = {
        {0.0, 0.0}, {0.3, -0.5}, {-0.7, 0.4}, {0.6, 0.6}, {-0.2, -0.8}
    };
    for (const auto& pt : test_pts) {
        Eigen::Vector2d phys = forward_map(elem, nodes, pt[0], pt[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.xi,  pt[0], kEps) << "xi";
        EXPECT_NEAR(result.eta, pt[1], kEps) << "eta";
    }
}

TEST(G2L_Q4, OutsidePointFails) {
    dic::Q4Element elem;
    auto nodes = make_q4_nodes(0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0);
    // 远离单元的点
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(100.0, 5.0));
    EXPECT_FALSE(result.converged);
}

// ---- Q8 ----

std::vector<dic::Node> make_q8_nodes(
    double x0, double y0, double x1, double y1,
    double x2, double y2, double x3, double y3,
    double x4, double y4, double x5, double y5,
    double x6, double y6, double x7, double y7) {
    std::vector<dic::Node> nodes(8);
    nodes[0].coordinate = Eigen::Vector2d(x0, y0);
    nodes[1].coordinate = Eigen::Vector2d(x1, y1);
    nodes[2].coordinate = Eigen::Vector2d(x2, y2);
    nodes[3].coordinate = Eigen::Vector2d(x3, y3);
    nodes[4].coordinate = Eigen::Vector2d(x4, y4);
    nodes[5].coordinate = Eigen::Vector2d(x5, y5);
    nodes[6].coordinate = Eigen::Vector2d(x6, y6);
    nodes[7].coordinate = Eigen::Vector2d(x7, y7);
    return nodes;
}

TEST(G2L_Q8, CornersRoundTrip) {
    dic::Q8Element elem;
    // 规则矩形 Q8，边中点在边的正中间
    auto nodes = make_q8_nodes(
        0.0, 0.0,   10.0, 0.0,   10.0, 10.0,   0.0, 10.0,
        5.0, 0.0,   10.0, 5.0,   5.0,  10.0,   0.0, 5.0);
    const double corners[][2] = {
        {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}
    };
    for (const auto& c : corners) {
        Eigen::Vector2d phys = forward_map(elem, nodes, c[0], c[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "corner (" << c[0] << ", " << c[1] << ")";
        EXPECT_NEAR(result.xi,  c[0], kEps);
        EXPECT_NEAR(result.eta, c[1], kEps);
    }
}

TEST(G2L_Q8, MidSidesRoundTrip) {
    dic::Q8Element elem;
    auto nodes = make_q8_nodes(
        0.0, 0.0,   10.0, 0.0,   10.0, 10.0,   0.0, 10.0,
        5.0, 0.0,   10.0, 5.0,   5.0,  10.0,   0.0, 5.0);
    const double mids[][2] = {
        {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}
    };
    for (const auto& m : mids) {
        Eigen::Vector2d phys = forward_map(elem, nodes, m[0], m[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "mid-side (" << m[0] << ", " << m[1] << ")";
        EXPECT_NEAR(result.xi,  m[0], kEps);
        EXPECT_NEAR(result.eta, m[1], kEps);
    }
}

TEST(G2L_Q8, CenterRoundTrip) {
    dic::Q8Element elem;
    auto nodes = make_q8_nodes(
        0.0, 0.0,   10.0, 0.0,   10.0, 10.0,   0.0, 10.0,
        5.0, 0.0,   10.0, 5.0,   5.0,  10.0,   0.0, 5.0);
    Eigen::Vector2d phys = forward_map(elem, nodes, 0.0, 0.0);
    auto result = dic::global_to_natural(elem, nodes, phys);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.xi,  0.0, kEps);
    EXPECT_NEAR(result.eta, 0.0, kEps);
}

TEST(G2L_Q8, ArbitraryPointRoundTrip) {
    dic::Q8Element elem;
    auto nodes = make_q8_nodes(
        0.0, 0.0,   10.0, 0.0,   10.0, 10.0,   0.0, 10.0,
        5.0, 0.0,   10.0, 5.0,   5.0,  10.0,   0.0, 5.0);
    const double test_pts[][2] = {
        {0.2, -0.3}, {-0.5, 0.4}, {0.7, 0.6}, {-0.1, -0.8},
        {0.0, 0.5},  {0.6, -0.9}, {-0.8, -0.3}
    };
    for (const auto& pt : test_pts) {
        Eigen::Vector2d phys = forward_map(elem, nodes, pt[0], pt[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.xi,  pt[0], 1e-6) << "ξ";
        EXPECT_NEAR(result.eta, pt[1], 1e-6) << "η";
    }
}

TEST(G2L_Q8, DistortedQuadRoundTrip) {
    dic::Q8Element elem;
    // 畸变 Q8：边中点在曲线边上偏移
    auto nodes = make_q8_nodes(
        1.0, 2.0,    11.0, 3.0,    10.0, 11.0,   0.0, 10.0,
        6.0, 2.5,    10.5, 7.0,    5.0, 10.5,    0.5, 6.0);
    const double test_pts[][2] = {
        {0.0, 0.0}, {0.3, -0.2}, {-0.6, 0.3}, {0.5, 0.7}, {-0.2, -0.7}
    };
    for (const auto& pt : test_pts) {
        Eigen::Vector2d phys = forward_map(elem, nodes, pt[0], pt[1]);
        auto result = dic::global_to_natural(elem, nodes, phys);
        EXPECT_TRUE(result.converged) << "at (" << pt[0] << ", " << pt[1] << ")";
        EXPECT_NEAR(result.xi,  pt[0], 1e-5) << "ξ";
        EXPECT_NEAR(result.eta, pt[1], 1e-5) << "η";
    }
}

TEST(G2L_Q8, OutsidePointFails) {
    dic::Q8Element elem;
    auto nodes = make_q8_nodes(
        0.0, 0.0,   10.0, 0.0,   10.0, 10.0,   0.0, 10.0,
        5.0, 0.0,   10.0, 5.0,   5.0,  10.0,   0.0, 5.0);
    // Point outside element: solver may still converge to a value outside [-1,1]
    // but the resulting natural coordinate should be clearly outside the element domain.
    auto result = dic::global_to_natural(elem, nodes, Eigen::Vector2d(-10.0, 5.0));
    bool inside_element = (std::abs(result.xi) <= 1.1 && std::abs(result.eta) <= 1.1);
    EXPECT_FALSE(inside_element) << "xi=" << result.xi << " eta=" << result.eta;
}

} // namespace
