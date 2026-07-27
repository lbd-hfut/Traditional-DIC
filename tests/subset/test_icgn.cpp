#include <dic/core/image.hpp>
#include <dic/initialization/initializer.hpp>
#include <dic/subset/solver/icgn.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double texture(double x, double y)
{
    return 0.45 +
           0.20 * std::sin(0.17 * x + 0.11 * y) +
           0.15 * std::cos(0.07 * x - 0.19 * y) +
           0.10 * std::sin(0.31 * x) * std::cos(0.23 * y);
}

dic::Image make_shifted_image(int width, int height, double shift_x, double shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(texture(static_cast<double>(x) - shift_x,
                                           static_cast<double>(y) - shift_y));
        }
    }
    return dic::Image(width, height, std::move(data));
}

} // namespace

TEST(Icgn, RefinesSubpixelTranslation)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);

    dic::SubsetConfig config;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.15);
    EXPECT_NEAR(result.v, -0.65, 0.15);
}

TEST(Icgn, SecondOrderShapeFunctionIsAStablePlaceholder)
{
    const auto reference = make_shifted_image(32, 32, 0.0, 0.0);
    const auto deformed = make_shifted_image(32, 32, 1.0, 0.0);

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(16.0, 16.0), initial);

    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::NotConverged);
    EXPECT_DOUBLE_EQ(result.u, 1.0);
    EXPECT_DOUBLE_EQ(result.v, 0.0);
}
