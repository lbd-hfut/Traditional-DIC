/**
 * @file subset_config.hpp
 * @brief Configuration for 2D Subset-DIC.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_CONFIG_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_CONFIG_HPP

#include <dic/initialization/seed_config.hpp>
#include <dic/interpolation/bspline.hpp>

namespace dic {

struct SubsetConfig {
    int subset_radius{15};
    int max_iterations{30};
    double convergence_threshold{1e-3};
    bool use_second_order{false};
    int search_radius{20};
    int propagation_spacing{5};
    double propagation_max_znssd{0.5};
    bool truncate_roi_subsets{false};
    double min_valid_sample_ratio{0.5};
    int min_valid_samples{12};
    BSplinePrecomputeConfig image_precompute{};
    SeedInitializationConfig seed_initialization{};
    SeedSelectionConfig seed_selection{};
    SubsetShapeFunctionMethod shape_function{SubsetShapeFunctionMethod::FirstOrder};
    SubsetOptimizationMethod optimizer{SubsetOptimizationMethod::ICGN};
    CorrelationCriterionKind objective{CorrelationCriterionKind::ZNSSD};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SUBSET_CONFIG_HPP
