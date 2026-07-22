#include <dic/core/mask.hpp>
#include <dic/core/roi.hpp>

#include <gtest/gtest.h>

TEST(Mask, ValidityQueries)
{
    dic::Mask mask(3, 2);
    mask.set(1, 1, false);

    EXPECT_TRUE(mask.valid(0, 0));
    EXPECT_FALSE(mask.valid(1, 1));
    EXPECT_FALSE(mask.valid(-1, 0));

    // TODO: Add user-supplied ROI mask image loading tests.
}

TEST(ROI, MaskInput)
{
    dic::Mask mask(2, 2);
    mask.set(0, 1, false);
    dic::ROI roi(mask);

    EXPECT_TRUE(roi.contains(1.0, 1.0));
    EXPECT_FALSE(roi.contains(0.0, 1.0));

    // TODO: Add rectangle, polygon, and ROI image dimension validation tests.
}
