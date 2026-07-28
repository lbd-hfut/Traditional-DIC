/**
 * @file integer_search.hpp
 * @brief IntegerSearchInitializer placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INTEGER_SEARCH_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INTEGER_SEARCH_HPP

#include <dic/initialization/initializer.hpp>
#include <dic/initialization/seed_config.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/core/mask.hpp>

namespace dic {

class IntegerSearchInitializer : public Initializer {
public:
    explicit IntegerSearchInitializer(int search_radius = 20);
    IntegerSearchInitializer(int search_radius, int subset_radius);
    explicit IntegerSearchInitializer(SeedInitializationConfig config);
    IntegerSearchInitializer(SeedInitializationConfig config, BSplinePrecomputeConfig image_precompute);
    InitialDisplacement estimate(const Image& reference, const Image& deformed, const Eigen::Vector2d& point) const override;
    InitialDisplacement estimate_with_interpolators(const Image& reference,
                                                    const Image& deformed,
                                                    const Eigen::Vector2d& point,
                                                    const BSplineInterpolator& reference_interpolator,
                                                    const BSplineInterpolator& deformed_interpolator) const;
    InitialDisplacement estimate_with_mask(const Image& reference,
                                           const Image& deformed,
                                           const Mask& roi,
                                           const Eigen::Vector2d& point) const;
    InitialDisplacement estimate_with_mask_interpolators(const Image& reference,
                                                        const Image& deformed,
                                                        const Mask& roi,
                                                        const Eigen::Vector2d& point,
                                                        const BSplineInterpolator& reference_interpolator,
                                                        const BSplineInterpolator& deformed_interpolator) const;
private:
    SeedInitializationConfig config_{};
    BSplinePrecomputeConfig image_precompute_{};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INTEGER_SEARCH_HPP
