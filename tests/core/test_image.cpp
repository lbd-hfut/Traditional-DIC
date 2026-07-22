#include <dic/core/image.hpp>

#include <gtest/gtest.h>

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
