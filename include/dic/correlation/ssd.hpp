/**
 * @file ssd.hpp
 * @brief Sum of squared differences correlation criterion.
 *
 * Responsibilities:
 * - Compare two equal-length intensity vectors without normalization.
 * - Exclude subset samples outside ROI through optional weights.
 * - Provide a simple baseline criterion for integer search and testing.
 *
 * Inputs:
 * - Reference/deformed subset intensity vectors and optional nonnegative weights.
 *
 * Outputs:
 * - Weighted sum of squared intensity differences; lower is better and 0 means identical vectors.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add optional robust/weighted SSD for masked subset samples.
 * - Add batch evaluation helpers for integer search windows.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_SSD_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_SSD_HPP

#include <dic/correlation/correlation.hpp>

namespace dic {

class SSDCorrelation : public CorrelationCriterion {
public:
    double evaluate(const Eigen::VectorXd& reference, const Eigen::VectorXd& deformed) const override;
    double evaluate(
        const Eigen::VectorXd& reference,
        const Eigen::VectorXd& deformed,
        const Eigen::VectorXd& weights
    ) const override;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_SSD_HPP
