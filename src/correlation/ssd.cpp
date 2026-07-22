/**
 * @file ssd.cpp
 * @brief Sum of squared differences implementation.
 *
 * Responsibilities:
 * - Validate correlation vector inputs.
 * - Evaluate the unnormalized SSD objective.
 * - Ignore non-ROI subset samples by assigning them zero weight.
 *
 * Inputs:
 * - Equal-length reference/deformed intensity vectors and optional sample weights.
 *
 * Outputs:
 * - Weighted sum of squared intensity residuals.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Add weighted/masked SSD variants.
 * - Add batch evaluation for integer-search displacement windows.
 */

#include <dic/correlation/ssd.hpp>

#include <stdexcept>

namespace dic {
namespace {

void validate_vectors(
    const Eigen::VectorXd& reference,
    const Eigen::VectorXd& deformed
)
{
    if (reference.size() == 0) {
        throw std::invalid_argument("Correlation vectors must not be empty.");
    }
    if (reference.size() != deformed.size()) {
        throw std::invalid_argument("Correlation vectors must have the same length.");
    }
    if (!reference.allFinite() || !deformed.allFinite()) {
        throw std::invalid_argument("Correlation vectors must contain only finite values.");
    }
}

void validate_weights(
    const Eigen::VectorXd& values,
    const Eigen::VectorXd& weights
)
{
    if (values.size() != weights.size()) {
        throw std::invalid_argument("Correlation weights must match vector length.");
    }
    if (!weights.allFinite()) {
        throw std::invalid_argument("Correlation weights must contain only finite values.");
    }
    if ((weights.array() < 0.0).any()) {
        throw std::invalid_argument("Correlation weights must be nonnegative.");
    }
    if (weights.sum() <= 0.0) {
        throw std::invalid_argument("Correlation weights must include at least one valid sample.");
    }
}

} // namespace

double SSDCorrelation::evaluate(
    const Eigen::VectorXd& reference,
    const Eigen::VectorXd& deformed
) const
{
    validate_vectors(reference, deformed);
    return (reference - deformed).squaredNorm();
}

double SSDCorrelation::evaluate(
    const Eigen::VectorXd& reference,
    const Eigen::VectorXd& deformed,
    const Eigen::VectorXd& weights
) const
{
    validate_vectors(reference, deformed);
    validate_weights(reference, weights);

    return (weights.array() * (reference - deformed).array().square()).sum();
}

} // namespace dic
