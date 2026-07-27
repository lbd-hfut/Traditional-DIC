/**
 * @file subset_dic.hpp
 * @brief High-level 2D Subset-DIC controller.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_DIC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_DIC_HPP

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/result.hpp>
#include <dic/subset/subset_config.hpp>
#include <vector>

namespace dic {

class SubsetDIC {
public:
    explicit SubsetDIC(SubsetConfig config = {});

    std::vector<Displacement2D> compute(const Image& reference,
                                        const Image& deformed) const;
    std::vector<Displacement2D> compute(const Image& reference,
                                        const Image& deformed,
                                        const Mask& roi) const;

private:
    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_DIC_HPP
