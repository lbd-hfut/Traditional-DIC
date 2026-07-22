/**
 * @file mask.hpp
 * @brief Binary image validity mask.
 *
 * Responsibilities:
 * - Store a binary validity mask for image-domain computations.
 * - Support ROI-mask images supplied by users without any GUI dependency.
 *
 * Inputs:
 * - Width/height, boolean validity values, or optional mask image path.
 *
 * Outputs:
 * - Pixel validity queries for ROI, Subset-DIC, and Mesh Generation.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add morphology/cleanup utilities only if downstream modules need them.
 * - Add robust OpenCV mask loading tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace dic {

class Mask {
public:
    Mask() = default;
    Mask(int width, int height);
    Mask(int width, int height, std::vector<bool> data);
    explicit Mask(const std::string& path);

    int width() const;
    int height() const;
    bool empty() const;
    bool contains(int x, int y) const;
    std::size_t size() const;

    bool valid(int x, int y) const;
    void set(int x, int y, bool valid);
    void fill(bool valid);

private:
    int width_{0};
    int height_{0};
    std::vector<bool> data_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP
