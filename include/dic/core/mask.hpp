/**
 * @file mask.hpp
 * @brief Binary image validity mask.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP

#include <vector>

namespace dic {

class Mask {
public:
    Mask() = default;
    Mask(int width, int height);
    bool valid(int x, int y) const;
private:
    int width_{0};
    int height_{0};
    std::vector<bool> data_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_MASK_HPP
