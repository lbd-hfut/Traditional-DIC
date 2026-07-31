#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/subset_initializer.hpp>
#include <dic/subset/padding.hpp>
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

TEST(IntegerSearch, NcorrStyleFullImageCoarseSearchFindsShiftOutsideLocalRadius)
{
    const auto reference = make_search_image(160, 128, 0.0, 0.0);
    const auto deformed = make_search_image(160, 128, 54.0, -11.0);

    dic::SeedInitializationConfig config;
    config.integer_search.subset_radius = 10;
    config.integer_search.search_radius = 8;
    config.integer_search.pyramid_enabled = true;
    config.integer_search.pyramid_scale = 4;
    config.integer_search.pyramid_refinement_radius = 4;
    config.subpixel.enabled = false;

    const dic::IntegerSearchInitializer initializer(config);
    const auto initial = initializer.estimate(reference, deformed, Eigen::Vector2d(72.0, 64.0));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 54.0);
    EXPECT_DOUBLE_EQ(initial.v, -11.0);
    EXPECT_GT(initial.zncc, 0.9);
    EXPECT_LT(initial.znssd, 0.2);
}

TEST(IntegerSearch, RefinesAroundExternalDisplacementPrior)
{
    const auto reference = make_search_image(160, 128, 0.0, 0.0);
    const auto deformed = make_search_image(160, 128, 54.0, -11.0);

    dic::SeedInitializationConfig config;
    config.integer_search.subset_radius = 10;
    config.integer_search.search_radius = 6;
    config.integer_search.pyramid_enabled = false;
    config.subpixel.enabled = false;

    const dic::IntegerSearchInitializer initializer(config);
    const auto initial = initializer.estimate_around_displacement(
        reference, deformed, Eigen::Vector2d(72.0, 64.0), 51.0, -9.0);

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 54.0);
    EXPECT_DOUBLE_EQ(initial.v, -11.0);
    EXPECT_GT(initial.zncc, 0.9);
}

TEST(IntegerSearch, UsesRoiTruncatedSamplesNearPaddedImageBoundary)
{
    const auto reference = make_search_image(48, 48, 0.0, 0.0);
    const auto deformed = make_search_image(48, 48, 2.0, -1.0);
    dic::Mask roi(reference.width(), reference.height());
    roi.fill(true);

    dic::SubsetConfig subset_config;
    subset_config.truncate_roi_subsets = true;
    subset_config.subset_radius = 9;
    subset_config.seed_initialization.integer_search.subset_radius = 9;
    subset_config.seed_initialization.integer_search.search_radius = 5;
    subset_config.seed_initialization.subpixel.enabled = false;

    const int pad = dic::recommended_subset_padding(subset_config);
    const auto padded_reference = dic::mirror_pad_image(reference, pad);
    const auto padded_deformed = dic::mirror_pad_image(deformed, pad);
    const auto padded_roi = dic::zero_pad_mask(roi, pad);

    const dic::IntegerSearchInitializer initializer(subset_config.seed_initialization);
    const auto initial = initializer.estimate_with_mask(
        padded_reference, padded_deformed, padded_roi, Eigen::Vector2d(2.0 + pad, 24.0 + pad));

    EXPECT_TRUE(initial.valid);
    EXPECT_DOUBLE_EQ(initial.u, 2.0);
    EXPECT_DOUBLE_EQ(initial.v, -1.0);
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
