/**
 * @file znssd.cpp
 * @brief Zero-normalized sum of squared differences implementation.
 *
 * Responsibilities:
 * - Validate correlation vector inputs.
 * - Evaluate normalized SSD after removing mean intensity and contrast scale.
 * - Ignore non-ROI subset samples by assigning them zero weight.
 *
 * Inputs:
 * - Equal-length reference/deformed intensity vectors and optional sample weights.
 *
 * Outputs:
 * - Weighted ZNSSD objective value where lower is better.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Add weighted/masked ZNSSD variants for clipped ROI subsets.
 * - Add batch evaluation for integer search and seed selection.
 */

#include <dic/correlation/znssd.hpp>

#include <cmath>
#include <stdexcept>

namespace dic {
namespace {

constexpr double kVarianceEpsilon = 1.0e-12;

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

Eigen::VectorXd zero_mean_unit_norm(
    const Eigen::VectorXd& values,
    const Eigen::VectorXd& weights
)
{
    validate_weights(values, weights);

    const double weight_sum = weights.sum();
    const double mean = weights.dot(values) / weight_sum;
    const auto centered = values.array() - mean;
    const double norm = std::sqrt((weights.array() * centered.square()).sum());
    if (norm <= kVarianceEpsilon) {
        throw std::invalid_argument("ZNSSD requires nonzero subset intensity variance.");
    }
    return centered.matrix() / norm;
}

} // namespace

double ZNSSDCorrelation::evaluate(
    const Eigen::VectorXd& reference,
    const Eigen::VectorXd& deformed
) const
{
    validate_vectors(reference, deformed);

    return evaluate(reference, deformed, Eigen::VectorXd::Ones(reference.size()));
}

double ZNSSDCorrelation::evaluate(
    const Eigen::VectorXd& reference,
    const Eigen::VectorXd& deformed,
    const Eigen::VectorXd& weights
) const
{
    validate_vectors(reference, deformed);

    const auto normalized_reference = zero_mean_unit_norm(reference, weights);
    const auto normalized_deformed = zero_mean_unit_norm(deformed, weights);
    return (weights.array() * (normalized_reference - normalized_deformed).array().square()).sum();
}

} // namespace dic
