#include <gtest/gtest.h>

#include "generation/inform_builder.hpp"
#include "coordinate/g2l_internal.hpp"

#include <dic/mesh/mesh_generation_config.hpp>

#include <vector>
#include <Eigen/Dense>

namespace {

bool has_candidate(const std::vector<double>& inform, double x, double y)
{
    for (std::size_t i = 0; i + 2 < inform.size(); i += 3) {
        if (inform[i] == x && inform[i + 1] == y) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(InformBuilder, Q8KeepsBoundingBoxCandidatesForCurvedBoundaryG2L)
{
    const std::vector<double> nodes = {
        2.0, 2.0,
        8.0, 2.0,
        8.0, 8.0,
        2.0, 8.0,
        5.0, 0.0,
        10.0, 5.0,
        5.0, 10.0,
        0.0, 5.0,
    };
    const std::vector<int> elements = {1, 2, 3, 4, 5, 6, 7, 8, 0};

    const auto inform = dic::mesh::build_inform(
        nodes.data(),
        8,
        elements.data(),
        1,
        dic::mesh::MeshElementType::Q8,
        12,
        12);

    EXPECT_TRUE(has_candidate(inform, 1.0, 1.0));
}

TEST(InformBuilder, Q8GlobalToNaturalKeepsOneSamplePerElementOwner)
{
    const std::vector<double> nodes = {
        2.0, 2.0, 8.0, 2.0, 8.0, 8.0, 2.0, 8.0,
        5.0, 2.0, 8.0, 5.0, 5.0, 8.0, 2.0, 5.0,
        2.0, 2.0, 8.0, 2.0, 8.0, 8.0, 2.0, 8.0,
        5.0, 2.0, 8.0, 5.0, 5.0, 8.0, 2.0, 5.0,
    };
    const std::vector<int> elements = {
        1, 2, 3, 4, 5, 6, 7, 8, 0,
        9, 10, 11, 12, 13, 14, 15, 16, 0,
    };
    const auto inform = dic::mesh::build_inform(
        nodes.data(), 16, elements.data(), 2,
        dic::mesh::MeshElementType::Q8, 12, 12);
    const auto g2l = dic::mesh::internal::compute_global_to_local(
        inform.data(), static_cast<int>(inform.size() / 3),
        nodes.data(), 16, elements.data(), 2, 12, 12,
        dic::mesh::MeshElementType::Q8);

    int sample_count = 0;
    for (const auto& sample : g2l.element_samples) {
        if (sample.pixel_index == 5 * 12 + 5) ++sample_count;
    }
    EXPECT_EQ(sample_count, 2);
}

TEST(InformBuilder, FEDICQ4KeepsSharedBoundaryPixelForEachElement)
{
    const std::vector<double> nodes = {
        1.0, 1.0, 3.0, 1.0, 5.0, 1.0,
        1.0, 3.0, 3.0, 3.0, 5.0, 3.0,
    };
    const std::vector<int> elements = {
        1, 2, 5, 4,
        2, 3, 6, 5,
    };
    const auto inform = dic::mesh::build_fedic_inform(
        nodes.data(), 6, elements.data(), 2,
        dic::mesh::MeshElementType::Q4, 8, 8);

    int shared_count = 0;
    for (std::size_t i = 0; i < inform.size(); i += 3) {
        if (inform[i] == 3.0 && inform[i + 1] == 2.0) ++shared_count;
    }
    EXPECT_EQ(shared_count, 2);

    const auto g2l = dic::mesh::internal::compute_global_to_local(
        inform.data(), static_cast<int>(inform.size() / 3),
        nodes.data(), 6, elements.data(), 2, 8, 8,
        dic::mesh::MeshElementType::Q4);
    int sample_count = 0;
    for (const auto& sample : g2l.element_samples) {
        if (sample.pixel_index == 2 * 8 + 3) ++sample_count;
    }
    EXPECT_EQ(sample_count, 2);
}

TEST(InformBuilder, FEDICQ4GlobalToNaturalMatchesPolynomialInverse)
{
    const double nodes[] = {
        2.0, 3.0,
        9.0, 2.0,
        11.0, 10.0,
        1.0, 12.0,
    };
    const double gx = 6.0;
    const double gy = 7.0;

    Eigen::Matrix4d matrix;
    for (int i = 0; i < 4; ++i) {
        const double x = nodes[2 * i];
        const double y = nodes[2 * i + 1];
        matrix.row(i) << x * y, x, y, 1.0;
    }
    const Eigen::Vector4d l = matrix.fullPivLu().solve(
        (Eigen::Vector4d() << -1.0, 1.0, 1.0, -1.0).finished());
    const Eigen::Vector4d m = matrix.fullPivLu().solve(
        (Eigen::Vector4d() << -1.0, -1.0, 1.0, 1.0).finished());
    const double expected_xi = l(0) * gx * gy + l(1) * gx + l(2) * gy + l(3);
    const double expected_eta = m(0) * gx * gy + m(1) * gx + m(2) * gy + m(3);

    double xi = 0.0, eta = 0.0, j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
    ASSERT_TRUE(dic::mesh::internal::solve_point_q4_fedic(
        gx, gy, nodes, xi, eta, j11, j12, j21, j22));
    EXPECT_NEAR(xi, expected_xi, 1e-12);
    EXPECT_NEAR(eta, expected_eta, 1e-12);
}
