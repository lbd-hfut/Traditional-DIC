/**
 * @file sift_initializer.hpp
 * @brief SIFTInitializer placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SIFT_INITIALIZER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SIFT_INITIALIZER_HPP

#include <dic/initialization/feature_matcher.hpp>
#include <dic/initialization/initializer.hpp>

namespace dic {

struct SIFTInitializerConfig {
    FeatureMatcherConfig matcher{};
    int interpolation_neighbors{8};
    double interpolation_radius{180.0};
};

class SIFTInitializer : public Initializer {
public:
    explicit SIFTInitializer(int search_radius = 20);
    explicit SIFTInitializer(SIFTInitializerConfig config);
    InitialDisplacement estimate(const Image& reference, const Image& deformed, const Eigen::Vector2d& point) const override;
    InitialDisplacement estimate_from_matches(const std::vector<FeatureMatch>& matches,
                                              const Eigen::Vector2d& point) const;
private:
    SIFTInitializerConfig config_{};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SIFT_INITIALIZER_HPP
