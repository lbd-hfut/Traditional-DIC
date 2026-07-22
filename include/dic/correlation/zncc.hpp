/**
 * @file zncc.hpp
 * @brief Zero-normalized cross-correlation criterion.
 *
 * Responsibilities:
 * - Compare two intensity vectors using normalized cross-correlation.
 * - Exclude subset samples outside ROI through optional weights.
 * - Provide a score-oriented counterpart to ZNSSD for initialization and diagnostics.
 *
 * Inputs:
 * - Reference/deformed subset intensity vectors and optional nonnegative weights.
 *
 * Outputs:
 * - Weighted ZNCC score in [-1, 1]; higher is better and 1 means perfect positive correlation.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add masked/weighted ZNCC for ROI boundary samples.
 * - Add batch evaluation helpers for feature/integer initialization.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNCC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNCC_HPP

#include <dic/correlation/correlation.hpp>

namespace dic {

class ZNCCCorrelation : public CorrelationCriterion {
public:
    double evaluate(const Eigen::VectorXd& reference, const Eigen::VectorXd& deformed) const override;
    double evaluate(
        const Eigen::VectorXd& reference,
        const Eigen::VectorXd& deformed,
        const Eigen::VectorXd& weights
    ) const override;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORRELATION_ZNCC_HPP
