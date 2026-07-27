/**
 * @file feature_matcher.hpp
 * @brief Feature matching interface for SIFT-ready initialization.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_FEATURE_MATCHER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_FEATURE_MATCHER_HPP

#include <dic/core/image.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

struct FeatureMatcherConfig {
    int max_features{4000};
    double ratio_threshold{0.75};
    double robust_mad_factor{5.0};
};

struct FeatureMatch {
    Eigen::Vector2d reference_point{0.0, 0.0};
    Eigen::Vector2d deformed_point{0.0, 0.0};
    Eigen::Vector2d displacement{0.0, 0.0};
    double confidence{0.0};
    bool valid{false};
    bool ratio_passed{false};
    bool mutual{false};
    bool robust_inlier{false};
};

class FeatureMatcher {
public:
    explicit FeatureMatcher(FeatureMatcherConfig config = {});
    std::vector<FeatureMatch> match(const Image& reference, const Image& deformed) const;

private:
    FeatureMatcherConfig config_{};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_FEATURE_MATCHER_HPP
