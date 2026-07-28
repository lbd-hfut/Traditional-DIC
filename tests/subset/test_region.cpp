#include <dic/subset/region.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

std::vector<std::uint8_t> make_column_major_mask(int width, int height)
{
    std::vector<std::uint8_t> mask(width * height, 0);
    for (int x = 1; x <= 3; ++x) {
        for (int y = 1; y <= 3; ++y) {
            mask[y + x * height] = 1;
        }
    }
    return mask;
}

} // namespace

TEST(SubsetRegion, FormsConnectedColumnMajorRegion)
{
    bool removed = false;
    const auto mask = make_column_major_mask(5, 5);
    const auto regions = dic::form_subset_regions(mask, 5, 5, 0, &removed);

    ASSERT_EQ(regions.size(), 1);
    EXPECT_FALSE(removed);
    EXPECT_EQ(regions[0].leftbound, 1);
    EXPECT_EQ(regions[0].rightbound, 3);
    EXPECT_EQ(regions[0].upperbound, 1);
    EXPECT_EQ(regions[0].lowerbound, 3);
    EXPECT_EQ(regions[0].totalpoints, 9);
    EXPECT_TRUE(regions[0].contains(2, 2));
    EXPECT_FALSE(regions[0].contains(0, 0));
}

TEST(SubsetRegion, ExtractsCircularSubsetWithinRegion)
{
    const auto mask = make_column_major_mask(5, 5);
    const auto regions = dic::form_subset_regions(mask, 5, 5);
    ASSERT_EQ(regions.size(), 1);

    const auto subset = dic::extract_circular_subset(regions[0], 2, 2, 1, false);

    EXPECT_EQ(subset.radius, 1);
    EXPECT_EQ(subset.x, 2);
    EXPECT_EQ(subset.y, 2);
    EXPECT_EQ(subset.region.totalpoints, 5);
    EXPECT_TRUE(subset.region.contains(2, 2));
    EXPECT_EQ(subset.mask[1 + 1 * 3], 1);
}

TEST(SubsetRegion, TruncationFlagControlsRoiClipping)
{
    const auto mask = make_column_major_mask(5, 5);
    const auto regions = dic::form_subset_regions(mask, 5, 5);
    ASSERT_EQ(regions.size(), 1);

    const auto full_circle = dic::extract_circular_subset(regions[0], 1, 2, 1, false);
    const auto truncated = dic::extract_circular_subset(regions[0], 1, 2, 1, true);

    EXPECT_EQ(full_circle.region.totalpoints, 5);
    EXPECT_EQ(truncated.region.totalpoints, 4);
    EXPECT_TRUE(full_circle.region.contains(0, 2));
    EXPECT_FALSE(truncated.region.contains(0, 2));
}
