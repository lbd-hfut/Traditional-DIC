/**
 * @file reliability_propagation.hpp
 * @brief Reliability-guided DIC propagation (RG-DIC).
 *
 * Responsibilities:
 * - Starting from pre-computed seed points, propagate outward using a
 *   priority queue ordered by ZNSSD correlation coefficient (lower = more reliable).
 * - For each computed point, extrapolate initial displacement to 4-connected
 *   neighbors using the first-order affine warp parameters.
 * - Run ICGN at each neighbor; on failure, fall back to integer search + ICGN.
 *
 * Dependencies:
 * - ICGNSolver, IntegerSearchInitializer, BSplineInterpolator, SubsetInitializer.
 * - Core Image, Mask, and Result containers.
 *
 * Reference:
 * - Ncorr RG-DIC algorithm (ncorr_alg_rgdic.cpp / ncorr_alg_dicanalysis.m).
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/result.hpp>
#include <dic/subset/subset_config.hpp>

#include <Eigen/Dense>
#include <vector>

namespace dic {

/// A pre-computed seed point with full affine initial guess.
struct PropagationSeed {
    Eigen::Vector2d point{0.0, 0.0};
    Displacement2D displacement{};
};

/// Result of a reliability-guided propagation.
struct PropagationResult {
    int grid_width{0};
    int grid_height{0};
    int spacing{0};
    std::vector<Displacement2D> points;
    int points_computed{0};
};

class ReliabilityPropagation {
public:
    explicit ReliabilityPropagation(SubsetConfig config = {});

    /**
     * @brief Run reliability-guided DIC propagation.
     *
     * @param reference  Reference (undeformed) image.
     * @param deformed   Deformed (current) image.
     * @param roi        Binary mask defining the valid region of interest.
     * @param seeds      Pre-computed seed points (from SubsetInitializer or SeedSelector).
     * @return PropagationResult containing a dense grid of Displacement2D entries.
     */
    PropagationResult propagate(
        const Image& reference,
        const Image& deformed,
        const Mask& roi,
        const std::vector<PropagationSeed>& seeds
    ) const;

private:
    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_RELIABILITY_PROPAGATION_HPP
