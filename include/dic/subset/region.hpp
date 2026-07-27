#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_REGION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_REGION_HPP

#include <cstdint>
#include <vector>

namespace dic {

struct SubsetRegion {
    std::vector<int32_t> nodelist;
    std::vector<int32_t> noderange;
    int height_nodelist{0};
    int width_nodelist{0};
    int upperbound{0};
    int lowerbound{0};
    int leftbound{0};
    int rightbound{0};
    int totalpoints{0};

    bool contains(int x, int y) const;
};

struct CircularSubsetRegion {
    SubsetRegion region;
    std::vector<std::uint8_t> mask;
    int radius{0};
    int x{0};
    int y{0};
};

std::vector<SubsetRegion> form_subset_regions(const std::vector<std::uint8_t>& mask,
                                              int width,
                                              int height,
                                              int cutoff = 0,
                                              bool* removed = nullptr);

CircularSubsetRegion extract_circular_subset(const SubsetRegion& region,
                                             int x,
                                             int y,
                                             int radius,
                                             bool truncate_subset);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_REGION_HPP
