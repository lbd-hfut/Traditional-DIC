/**
 * @file image.hpp
 * @brief Grayscale floating-point image container.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP

#include <string>
#include <vector>

namespace dic {

class Image {
public:
    Image();
    explicit Image(const std::string& path);
    int width() const;
    int height() const;
    bool empty() const;
    float at(int x, int y) const;
private:
    int width_{0};
    int height_{0};
    std::vector<float> data_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP
