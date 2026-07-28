#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
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
