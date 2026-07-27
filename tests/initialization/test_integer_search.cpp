#include <dic/core/image.hpp>
#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/subset_initializer.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double search_texture(double x, double y)
{
    return 0.5 +
           0.22 * std::sin(0.15 * x + 0.13 * y) +
           0.18 * std::cos(0.21 * x - 0.08 * y) +
           0.08 * std::sin(0.37 * x) * std::sin(0.29 * y);
}

dic::Image make_search_image(int width, int height, double shift_x, double shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(search_texture(static_cast<double>(x) - shift_x,
                                                  static_cast<double>(y) - shift_y));
        }
    }
    return dic::Image(width, height, std::move(data));
}

} // namespace

TEST(IntegerSearch, FindsIntegerShiftOnly)
{
    const auto reference = make_search_image(72, 72, 0.0, 0.0);
    const auto deformed = make_search_image(72, 72, 2.25, -1.45);

    const dic::IntegerSearchInitializer initializer(5, 9);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(36.0, 36.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 2.0);
    EXPECT_DOUBLE_EQ(initial.v, -1.0);
}

TEST(IntegerSearch, CanDisableSubpixelRefinementFromSeedConfig)
{
    const auto reference = make_search_image(72, 72, 0.0, 0.0);
    const auto deformed = make_search_image(72, 72, 2.25, -1.45);

    dic::SeedInitializationConfig config;
    config.integer_search.subset_radius = 9;
    config.integer_search.search_radius = 5;
    config.subpixel.enabled = false;

    const dic::IntegerSearchInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(36.0, 36.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 2.0);
    EXPECT_DOUBLE_EQ(initial.v, -1.0);
}

TEST(IntegerSearch, IgnoresSubpixelRefinementFromSeedConfig)
{
    const auto reference = make_search_image(72, 72, 0.0, 0.0);
    const auto deformed = make_search_image(72, 72, 2.25, -1.45);

    dic::SeedInitializationConfig config;
    config.integer_search.subset_radius = 9;
    config.integer_search.search_radius = 5;
    config.subpixel.enabled = true;
    config.subpixel.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.subpixel.optimizer = dic::SubsetOptimizationMethod::ForwardGaussNewton;

    const dic::IntegerSearchInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(36.0, 36.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 2.0);
    EXPECT_DOUBLE_EQ(initial.v, -1.0);
}

TEST(IntegerSearch, PyramidSearchFindsLargeIntegerShift)
{
    const auto reference = make_search_image(128, 128, 0.0, 0.0);
    const auto deformed = make_search_image(128, 128, 18.0, -14.0);

    dic::SeedInitializationConfig config;
    config.integer_search.subset_radius = 10;
    config.integer_search.search_radius = 24;
    config.integer_search.pyramid_enabled = true;
    config.integer_search.pyramid_scale = 4;
    config.integer_search.pyramid_refinement_radius = 4;
    config.subpixel.enabled = false;

    const dic::IntegerSearchInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(64.0, 64.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 18.0);
    EXPECT_DOUBLE_EQ(initial.v, -14.0);
}

TEST(SubsetInitializer, RunsIntegerSearchAndSubpixelRefinement)
{
    const auto reference = make_search_image(72, 72, 0.0, 0.0);
    const auto deformed = make_search_image(72, 72, 2.25, -1.45);

    dic::SubsetConfig config;
    config.seed_initialization.integer_search.subset_radius = 9;
    config.seed_initialization.integer_search.search_radius = 5;
    config.seed_initialization.subpixel.enabled = true;
    config.seed_initialization.subpixel.shape_function = dic::SubsetShapeFunctionMethod::FirstOrder;
    config.seed_initialization.subpixel.optimizer = dic::SubsetOptimizationMethod::ICGN;
    config.seed_initialization.subpixel.subset_radius = 9;
    config.seed_initialization.subpixel.max_iterations = 30;
    config.seed_initialization.subpixel.convergence_threshold = 1e-4;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::SubsetInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(36.0, 36.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_NEAR(initial.u, 2.25, 0.35);
    EXPECT_NEAR(initial.v, -1.45, 0.35);
}

TEST(SubsetInitializer, CanSelectForwardGaussNewtonSubpixelPlaceholder)
{
    const auto reference = make_search_image(72, 72, 0.0, 0.0);
    const auto deformed = make_search_image(72, 72, 2.25, -1.45);

    dic::SubsetConfig config;
    config.seed_initialization.integer_search.subset_radius = 9;
    config.seed_initialization.integer_search.search_radius = 5;
    config.seed_initialization.subpixel.enabled = true;
    config.seed_initialization.subpixel.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.seed_initialization.subpixel.optimizer = dic::SubsetOptimizationMethod::ForwardGaussNewton;

    const dic::SubsetInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(36.0, 36.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 2.0);
    EXPECT_DOUBLE_EQ(initial.v, -1.0);
}
