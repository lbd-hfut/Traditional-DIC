/**
 * @file correlation.hpp
 * @brief Abstract correlation criterion interface for DIC intensity vectors.
 *
 * Responsibilities:
 * - Define a low-coupling interface for comparing reference and deformed subsets.
 * - Standardize input validation expectations for all correlation criteria.
 * - Support ROI/mask boundary subsets through per-sample weights.
 *
 * Inputs:
 * - Equal-length Eigen vectors containing sampled grayscale intensities.
 * - Optional equal-length nonnegative weights; weight 0 excludes non-ROI samples.
 *
 * Outputs:
 * - A scalar criterion value whose interpretation is defined by each subclass.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add explicit valid-mask adapters for ROI and Mask modules.
 * - Add SIMD/OpenMP paths for large batch subset evaluation.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_CORRELATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_CORRELATION_HPP

#include <Eigen/Dense>

namespace dic {

class CorrelationCriterion {
public:
    virtual ~CorrelationCriterion() = default;

    virtual double evaluate(
        const Eigen::VectorXd& reference,
        const Eigen::VectorXd& deformed
    ) const = 0;

    virtual double evaluate(
        const Eigen::VectorXd& reference,
        const Eigen::VectorXd& deformed,
        const Eigen::VectorXd& weights
    ) const = 0;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_CORRELATION_HPP
