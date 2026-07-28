#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/subset/seed/reliability_propagation.hpp>
#include <dic/subset/subset_dic.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double subset_dic_texture(double x, double y)
{
    return 0.5 +
           0.23 * std::sin(0.16 * x + 0.11 * y) +
           0.19 * std::cos(0.19 * x - 0.07 * y) +
           0.07 * std::sin(0.31 * x) * std::cos(0.27 * y);
}

dic::Image make_subset_dic_image(int width, int height, double shift_x, double shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(subset_dic_texture(static_cast<double>(x) - shift_x,
                                                      static_cast<double>(y) - shift_y));
        }
    }
    return dic::Image(width, height, std::move(data));
}

dic::SubsetConfig make_subset_dic_config()
{
    dic::SubsetConfig config;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;
    config.seed_initialization.integer_search.subset_radius = 7;
    config.seed_initialization.integer_search.search_radius = 5;
    config.seed_initialization.integer_search.pyramid_enabled = false;
    config.seed_initialization.subpixel.enabled = true;
    config.seed_initialization.subpixel.shape_function = dic::SubsetShapeFunctionMethod::FirstOrder;
    config.seed_initialization.subpixel.optimizer = dic::SubsetOptimizationMethod::ICGN;
    config.seed_initialization.subpixel.subset_radius = 7;
    config.seed_initialization.subpixel.max_iterations = 30;
    config.seed_initialization.subpixel.convergence_threshold = 1e-4;
    config.seed_selection.seed_count = 9;
    config.seed_selection.quality_metric = dic::SeedQualityMetric::ZNSSD;
    config.seed_selection.max_znssd = 0.5;
    config.seed_selection.min_texture_std = 0.0;
    config.propagation_spacing = 8;
    config.propagation_max_znssd = 0.5;
    return config;
}

} // namespace

TEST(SubsetDIC, PropagatesReliablePointsFromAutomaticSeed)
{
    const auto reference = make_subset_dic_image(96, 96, 0.0, 0.0);
    const auto deformed = make_subset_dic_image(96, 96, 2.2, -1.4);
    dic::Mask roi(96, 96);
    roi.fill(true);

    const dic::SubsetDIC solver(make_subset_dic_config());
    const auto results = solver.compute(reference, deformed, roi);

    ASSERT_GT(results.size(), 5U);
    int valid_found = 0;
    for (const auto& result : results) {
        if (!result.valid) continue;
        ++valid_found;
        EXPECT_EQ(result.status, dic::SolverStatus::Success);
        EXPECT_NEAR(result.u, 2.2, 0.6);
        EXPECT_NEAR(result.v, -1.4, 0.6);
    }
    EXPECT_GT(valid_found, 5);
}

TEST(ReliabilityPropagation, UsesNcorrSpacingAsReducedGridStride)
{
    const auto reference = make_subset_dic_image(11, 10, 0.0, 0.0);
    const auto deformed = make_subset_dic_image(11, 10, 0.0, 0.0);
    dic::Mask roi(11, 10);
    roi.fill(true);

    dic::SubsetConfig config;
    config.subset_radius = 1;
    config.max_iterations = 1;
    config.propagation_spacing = 4;

    dic::PropagationSeed seed;
    seed.point = Eigen::Vector2d(5.0, 5.0);
    seed.displacement.x = 5.0;
    seed.displacement.y = 5.0;
    seed.displacement.valid = true;
    seed.displacement.status = dic::SolverStatus::Success;

    const dic::ReliabilityPropagation propagation(config);
    const auto result = propagation.propagate(reference, deformed, roi, {seed});

    EXPECT_EQ(result.spacing, 5);
    EXPECT_EQ(result.grid_width, 3);
    EXPECT_EQ(result.grid_height, 2);
    ASSERT_EQ(result.points.size(), 6U);
    EXPECT_DOUBLE_EQ(result.points[0].x, 0.0);
    EXPECT_DOUBLE_EQ(result.points[1].x, 5.0);
    EXPECT_DOUBLE_EQ(result.points[2].x, 10.0);
    EXPECT_DOUBLE_EQ(result.points[3].y, 5.0);
}

TEST(ReliabilityPropagation, SnapsOffGridSeedToNearestReducedGridPoint)
{
    const auto reference = make_subset_dic_image(32, 32, 0.0, 0.0);
    const auto deformed = make_subset_dic_image(32, 32, 0.0, 0.0);
    dic::Mask roi(32, 32);
    roi.fill(true);

    dic::SubsetConfig config;
    config.subset_radius = 4;
    config.max_iterations = 10;
    config.convergence_threshold = 1e-6;
    config.propagation_spacing = 3;

    dic::PropagationSeed seed;
    seed.point = Eigen::Vector2d(10.0, 9.0);
    seed.displacement.x = 10.0;
    seed.displacement.y = 9.0;
    seed.displacement.valid = true;
    seed.displacement.status = dic::SolverStatus::Success;

    const dic::ReliabilityPropagation propagation(config);
    const auto result = propagation.propagate(reference, deformed, roi, {seed});

    ASSERT_EQ(result.grid_width, 8);
    ASSERT_EQ(result.grid_height, 8);
    const auto& snapped = result.points[2 + 2 * result.grid_width];
    EXPECT_TRUE(snapped.valid);
    EXPECT_DOUBLE_EQ(snapped.x, 8.0);
    EXPECT_DOUBLE_EQ(snapped.y, 8.0);
}
