#include <gtest/gtest.h>

#include <dic/initialization/sift_initializer.hpp>

TEST(SIFTInitializer, InterpolatesPointFromRobustFeatureMatches)
{
    dic::SIFTInitializerConfig config;
    config.interpolation_neighbors = 4;
    config.interpolation_radius = 20.0;
    const dic::SIFTInitializer initializer(config);

    std::vector<dic::FeatureMatch> matches;
    for (const auto& point : {
             Eigen::Vector2d{10.0, 10.0},
             Eigen::Vector2d{20.0, 10.0},
             Eigen::Vector2d{10.0, 20.0},
             Eigen::Vector2d{20.0, 20.0}}) {
        dic::FeatureMatch match;
        match.reference_point = point;
        match.deformed_point = point + Eigen::Vector2d{2.0, -1.0};
        match.displacement = {2.0, -1.0};
        match.confidence = 0.9;
        match.valid = true;
        match.robust_inlier = true;
        matches.push_back(match);
    }

    dic::FeatureMatch outlier;
    outlier.reference_point = {15.0, 15.0};
    outlier.deformed_point = {115.0, 115.0};
    outlier.displacement = {100.0, 100.0};
    outlier.confidence = 0.1;
    outlier.valid = false;
    outlier.robust_inlier = false;
    matches.push_back(outlier);

    const auto initial = initializer.estimate_from_matches(matches, {15.0, 15.0});
    ASSERT_TRUE(initial.valid);
    EXPECT_NEAR(initial.u, 2.0, 1e-12);
    EXPECT_NEAR(initial.v, -1.0, 1e-12);
}

TEST(SIFTInitializer, ReturnsInvalidWithoutNearbyInliers)
{
    dic::SIFTInitializerConfig config;
    config.interpolation_radius = 5.0;
    const dic::SIFTInitializer initializer(config);

    dic::FeatureMatch match;
    match.reference_point = {100.0, 100.0};
    match.displacement = {2.0, -1.0};
    match.valid = true;
    match.robust_inlier = true;

    const auto initial = initializer.estimate_from_matches({match}, {0.0, 0.0});
    EXPECT_FALSE(initial.valid);
}
