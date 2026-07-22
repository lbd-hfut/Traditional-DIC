#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>

#include <gtest/gtest.h>

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
