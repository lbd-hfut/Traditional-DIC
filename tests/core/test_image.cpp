#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

TEST(Image, MemoryConstruction)
{
    dic::Image image(2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

    EXPECT_EQ(image.width(), 2);
    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.size(), 4U);
    EXPECT_TRUE(image.contains(1, 1));
    EXPECT_FALSE(image.contains(2, 1));
    EXPECT_FLOAT_EQ(image.at(1, 0), 2.0F);

    image.set(0, 1, 5.0F);
    EXPECT_FLOAT_EQ(image.at(0, 1), 5.0F);

    // TODO: Add OpenCV-backed file loading tests when test fixtures exist.
}

TEST(Image, DefaultFileLoadOptionsScaleSpeckleImagesToUnitRange)
{
    dic::ImageLoadOptions options;

    EXPECT_EQ(options.color_mode, dic::ImageColorMode::Grayscale);
    EXPECT_EQ(options.intensity_scale, dic::ImageIntensityScale::Unit);
}

TEST(ImagePreprocessing, GlobalMeanStdNormalization)
{
    dic::Image image(2, 2, {1.0F, 2.0F, 3.0F, 4.0F});

    const auto normalized = dic::normalize_global_mean_std(image);

    EXPECT_EQ(normalized.width(), 2);
    EXPECT_EQ(normalized.height(), 2);
    EXPECT_NEAR(normalized.at(0, 0), -1.3416408F, 1e-6F);
    EXPECT_NEAR(normalized.at(1, 1), 1.3416408F, 1e-6F);
}

TEST(ImagePreprocessing, MaxIntensityNormalizationUsesFullImageMaximum)
{
    dic::Image image(2, 2, {0.0F, 64.0F, 128.0F, 255.0F});

    const auto normalized = dic::normalize_max_intensity(image);

    EXPECT_FLOAT_EQ(normalized.at(0, 0), 0.0F);
    EXPECT_NEAR(normalized.at(1, 0), 64.0F / 255.0F, 1e-6F);
    EXPECT_NEAR(normalized.at(0, 1), 128.0F / 255.0F, 1e-6F);
    EXPECT_FLOAT_EQ(normalized.at(1, 1), 1.0F);
}

TEST(ImagePreprocessing, RoiMeanStdNormalizationUsesOnlyValidPixelsForStats)
{
    dic::Image image(2, 2, {1.0F, 2.0F, 10.0F, 20.0F});
    dic::Mask roi(2, 2, {true, true, false, false});

    const auto normalized = dic::normalize_roi_mean_std(image, roi);

    EXPECT_FLOAT_EQ(normalized.at(0, 0), -1.0F);
    EXPECT_FLOAT_EQ(normalized.at(1, 0), 1.0F);
    EXPECT_FLOAT_EQ(normalized.at(0, 1), 17.0F);
    EXPECT_FLOAT_EQ(normalized.at(1, 1), 37.0F);
}

TEST(ImagePreprocessing, ConstantImageNormalizesToZero)
{
    dic::Image image(2, 2, {5.0F, 5.0F, 5.0F, 5.0F});

    const auto normalized = dic::normalize_image(image, dic::ImageNormalization::GlobalMeanStd);

    for (float value : normalized.data()) {
        EXPECT_FLOAT_EQ(value, 0.0F);
    }
}

TEST(ImagePreprocessing, RoiNormalizationRequiresMatchingMask)
{
    dic::Image image(2, 2, {1.0F, 2.0F, 3.0F, 4.0F});
    dic::Mask roi(1, 1);

    EXPECT_THROW((void)dic::normalize_roi_mean_std(image, roi), std::invalid_argument);
    EXPECT_THROW((void)dic::normalize_image(image, dic::ImageNormalization::RoiMeanStd), std::invalid_argument);
}
