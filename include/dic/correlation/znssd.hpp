/**
 * @file znssd.hpp
 * @brief Zero-normalized sum of squared differences criterion.
 *
 * Responsibilities:
 * - Compare two intensity vectors after zero-mean, unit-norm normalization.
 * - Exclude subset samples outside ROI through optional weights.
 * - Provide the primary correlation criterion for Subset-DIC and Mesh-DIC residual checks.
 *
 * Inputs:
 * - Reference/deformed subset intensity vectors and optional nonnegative weights.
 *
 * Outputs:
 * - Weighted ZNSSD value; lower is better, 0 means normalized vectors match, and 4 is perfect inverse correlation.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add masked/weighted ZNSSD for ROI boundary samples.
 * - Add numerical regression tests against known DIC benchmark data.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNSSD_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNSSD_HPP

#include <dic/correlation/correlation.hpp>

namespace dic {

class ZNSSDCorrelation : public CorrelationCriterion {
public:
    double evaluate(const Eigen::VectorXd& reference, const Eigen::VectorXd& deformed) const override;
    double evaluate(
        const Eigen::VectorXd& reference,
        const Eigen::VectorXd& deformed,
        const Eigen::VectorXd& weights
    ) const override;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNSSD_HPP
