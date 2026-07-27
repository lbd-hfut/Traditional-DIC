#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/subset/seed/seed_selector.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double seed_texture(double x, double y)
{
    return 0.5 +
           0.22 * std::sin(0.15 * x + 0.13 * y) +
           0.18 * std::cos(0.21 * x - 0.08 * y) +
           0.08 * std::sin(0.37 * x) * std::sin(0.29 * y);
}

dic::Image make_seed_image(int width, int height, double shift_x, double shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(seed_texture(static_cast<double>(x) - shift_x,
                                                static_cast<double>(y) - shift_y));
        }
    }
    return dic::Image(width, height, std::move(data));
}

dic::Mask make_full_roi(int width, int height)
{
    dic::Mask roi(width, height);
    roi.fill(true);
    return roi;
}

dic::SubsetConfig make_seed_selector_config(int seed_count)
{
    dic::SubsetConfig config;
    config.seed_selection.seed_count = seed_count;
    config.seed_selection.threads = 1;
    config.seed_selection.quality_metric = dic::SeedQualityMetric::ZNSSD;
    config.seed_selection.max_znssd = 0.4;
    config.seed_selection.min_displacement_norm = 0.0;
    config.seed_initialization.integer_search.subset_radius = 7;
    config.seed_initialization.integer_search.search_radius = 4;
    config.seed_initialization.subpixel.enabled = true;
    config.seed_initialization.subpixel.shape_function = dic::SubsetShapeFunctionMethod::FirstOrder;
    config.seed_initialization.subpixel.optimizer = dic::SubsetOptimizationMethod::ICGN;
    config.seed_initialization.subpixel.subset_radius = 9;
    config.seed_initialization.subpixel.max_iterations = 30;
    config.seed_initialization.subpixel.convergence_threshold = 1e-4;
    return config;
}

} // namespace

TEST(SeedSelector, GeneratesUniformCandidatesInsideRoi)
{
    const auto reference = make_seed_image(80, 80, 0.0, 0.0);
    const auto roi = make_full_roi(80, 80);
    const dic::SeedSelector selector(make_seed_selector_config(16));

    const auto candidates = selector.generate_candidates(reference, roi);

    EXPECT_EQ(candidates.size(), 16);
    for (const auto& point : candidates) {
        EXPECT_TRUE(roi.valid(static_cast<int>(point.x()), static_cast<int>(point.y())));
    }
}

TEST(SeedSelector, GeneratesKmeansCandidatesAcrossHollowRoi)
{
    const auto reference = make_seed_image(120, 120, 0.0, 0.0);
    dic::Mask roi(120, 120);
    roi.fill(false);
    for (int y = 0; y < roi.height(); ++y) {
        for (int x = 0; x < roi.width(); ++x) {
            const int dx = x - 60;
            const int dy = y - 60;
            const int r2 = dx * dx + dy * dy;
            if (r2 >= 35 * 35 && r2 <= 45 * 45) {
                roi.set(x, y, true);
            }
        }
    }

    auto config = make_seed_selector_config(25);
    config.seed_initialization.integer_search.subset_radius = 2;
    config.seed_initialization.integer_search.search_radius = 1;
    config.seed_initialization.subpixel.subset_radius = 2;
    config.seed_selection.min_texture_std = 0.0;
    const dic::SeedSelector selector(config);

    const auto candidates = selector.generate_candidates(reference, roi);

    EXPECT_EQ(candidates.size(), 25);
    for (const auto& point : candidates) {
        EXPECT_TRUE(roi.valid(static_cast<int>(point.x()), static_cast<int>(point.y())));
    }
}

TEST(SeedSelector, SelectsQualityPassingCandidateWithLargestDisplacement)
{
    const auto reference = make_seed_image(80, 80, 0.0, 0.0);
    const auto deformed = make_seed_image(80, 80, 2.35, -1.45);
    const auto roi = make_full_roi(80, 80);
    const dic::SeedSelector selector(make_seed_selector_config(16));

    const auto result = selector.select_best_seed(reference, deformed, roi);

    ASSERT_TRUE(result.found);
    ASSERT_FALSE(result.candidates.empty());
    EXPECT_TRUE(result.best_seed.valid);
    EXPECT_TRUE(result.best_seed.quality_passed);
    EXPECT_NEAR(result.best_seed.displacement.u, 2.35, 0.5);
    EXPECT_NEAR(result.best_seed.displacement.v, -1.45, 0.5);

    for (const auto& candidate : result.candidates) {
        if (candidate.valid && candidate.quality_passed) {
            EXPECT_LE(candidate.displacement_norm, result.best_seed.displacement_norm + 1e-12);
        }
    }
}
