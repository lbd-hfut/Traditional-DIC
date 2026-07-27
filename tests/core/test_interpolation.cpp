#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <dic/subset/subset_config.hpp>

#include <gtest/gtest.h>

#include <cmath>

TEST(BSplineInterpolation, PrecomputeShapes)
{
    dic::Image image(3, 2, {
        0.0F, 1.0F, 2.0F,
        3.0F, 4.0F, 5.0F
    });

    for (const auto degree : {dic::BSplineDegree::Linear, dic::BSplineDegree::Cubic, dic::BSplineDegree::Quintic}) {
        dic::BSplineImagePreprocessor preprocessor({degree, 3, false});
        const auto precomputed = preprocessor.compute(image);

        EXPECT_EQ(precomputed.width, 3);
        EXPECT_EQ(precomputed.height, 2);
        EXPECT_EQ(precomputed.local_polynomial_blocks.size(), 6U);
        EXPECT_EQ(precomputed.gradient_x.rows(), 2);
        EXPECT_EQ(precomputed.gradient_x.cols(), 3);
        EXPECT_EQ(precomputed.qk.rows(), static_cast<int>(degree) + 1);
        EXPECT_EQ(precomputed.qk.cols(), static_cast<int>(degree) + 1);
    }
}

TEST(BSplinePrecomputeConfig, DefaultsUseExactPrefilterAndMinimumBorder)
{
    dic::Image image(4, 4, {
        0.0F, 1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F, 7.0F,
        8.0F, 9.0F, 10.0F, 11.0F,
        12.0F, 13.0F, 14.0F, 15.0F
    });

    const dic::BSplinePrecomputeConfig default_config;
    EXPECT_EQ(default_config.degree, dic::BSplineDegree::Quintic);
    EXPECT_EQ(default_config.border, 3);
    EXPECT_TRUE(default_config.use_exact_prefilter);

    dic::BSplineImagePreprocessor preprocessor({dic::BSplineDegree::Cubic, 1, true});
    const auto precomputed = preprocessor.compute(image);

    EXPECT_EQ(precomputed.config.border, 3);
    EXPECT_TRUE(precomputed.config.use_exact_prefilter);
    EXPECT_EQ(precomputed.coefficients.rows(), image.height() + 6);
    EXPECT_EQ(precomputed.coefficients.cols(), image.width() + 6);
}

TEST(BSplineInterpolation, LinearIntegerValueAndGradient)
{
    dic::Image image(2, 2, {
        1.0F, 3.0F,
        5.0F, 7.0F
    });

    dic::BSplineInterpolator interpolator(
        image,
        dic::BSplinePrecomputeConfig{dic::BSplineDegree::Linear, 1, false}
    );

    EXPECT_NEAR(interpolator.value(0.0, 0.0), 1.0, 1e-12);
    EXPECT_NEAR(interpolator.value(1.0, 0.0), 3.0, 1e-12);
    EXPECT_NEAR(interpolator.value(0.0, 1.0), 5.0, 1e-12);

    const auto gradient = interpolator.gradient(0.0, 0.0);
    EXPECT_NEAR(gradient.x(), 2.0, 1e-12);
    EXPECT_NEAR(gradient.y(), 4.0, 1e-12);
}

TEST(BSplineInterpolation, CubicSplineSubpixelQuery)
{
    dic::Image image(2, 2, {
        1.0F, 2.0F,
        3.0F, 4.0F
    });

    dic::BSplineInterpolator interpolator(
        image,
        dic::BSplinePrecomputeConfig{dic::BSplineDegree::Cubic, 3, false}
    );

    EXPECT_NO_THROW({
        (void)interpolator.value(0.25, 0.25);
        (void)interpolator.gradient(0.25, 0.25);
    });
}

TEST(BSplineInterpolation, PrecomputedGradientsMatchInterpolatorAtPixelCenters)
{
    dic::Image image(5, 4, {
        0.0F, 1.0F, 3.0F, 6.0F, 10.0F,
        2.0F, 4.0F, 7.0F, 11.0F, 16.0F,
        5.0F, 8.0F, 12.0F, 17.0F, 23.0F,
        9.0F, 13.0F, 18.0F, 24.0F, 31.0F
    });

    for (const auto degree : {dic::BSplineDegree::Linear, dic::BSplineDegree::Cubic, dic::BSplineDegree::Quintic}) {
        dic::BSplinePrecomputeConfig config{degree, 3, false};
        dic::BSplineImagePreprocessor preprocessor(config);
        const auto precomputed = preprocessor.compute(image);
        dic::BSplineInterpolator interpolator(precomputed);

        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                const auto gradient = interpolator.gradient(
                    static_cast<double>(x) + 0.5,
                    static_cast<double>(y) + 0.5
                );
                EXPECT_NEAR(precomputed.gradient_x(y, x), gradient.x(), 1e-12);
                EXPECT_NEAR(precomputed.gradient_y(y, x), gradient.y(), 1e-12);
            }
        }
    }
}

TEST(BSplineInterpolation, AnalyticGradientMatchesFiniteDifference)
{
    dic::Image image(6, 6, {
        0.0F, 1.0F, 4.0F, 9.0F, 16.0F, 25.0F,
        1.0F, 3.0F, 7.0F, 13.0F, 21.0F, 31.0F,
        4.0F, 7.0F, 12.0F, 19.0F, 28.0F, 39.0F,
        9.0F, 13.0F, 19.0F, 27.0F, 37.0F, 49.0F,
        16.0F, 21.0F, 28.0F, 37.0F, 48.0F, 61.0F,
        25.0F, 31.0F, 39.0F, 49.0F, 61.0F, 75.0F
    });

    for (const auto degree : {dic::BSplineDegree::Linear, dic::BSplineDegree::Cubic, dic::BSplineDegree::Quintic}) {
        dic::BSplineInterpolator interpolator(
            image,
            dic::BSplinePrecomputeConfig{degree, 3, false}
        );

        const double x = 2.35;
        const double y = 2.65;
        const double h = 1e-5;
        const auto gradient = interpolator.gradient(x, y);
        const double fd_x = (interpolator.value(x + h, y) - interpolator.value(x - h, y)) / (2.0 * h);
        const double fd_y = (interpolator.value(x, y + h) - interpolator.value(x, y - h)) / (2.0 * h);

        EXPECT_NEAR(gradient.x(), fd_x, 1e-6);
        EXPECT_NEAR(gradient.y(), fd_y, 1e-6);
    }
}

TEST(BSplinePrecomputeConfig, SharedBySubsetAndMesh)
{
    dic::SubsetConfig subset_config;
    dic::MeshConfig mesh_config;

    subset_config.image_precompute.degree = dic::BSplineDegree::Linear;
    mesh_config.image_precompute.degree = dic::BSplineDegree::Cubic;

    EXPECT_EQ(static_cast<int>(subset_config.image_precompute.degree), 1);
    EXPECT_EQ(static_cast<int>(mesh_config.image_precompute.degree), 3);

    subset_config.image_precompute.degree = dic::BSplineDegree::Quintic;
    mesh_config.image_precompute = subset_config.image_precompute;

    EXPECT_EQ(static_cast<int>(mesh_config.image_precompute.degree), 5);
}
