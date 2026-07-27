/**
 * @file subset_dic.cpp
 * @brief High-level 2D Subset-DIC controller.
 *
 * Orchestrates: Seed selection → Subset initialization → RG-DIC propagation.
 */

#include <dic/subset/seed/reliability_propagation.hpp>
#include <dic/subset/seed/seed_selector.hpp>
#include <dic/subset/subset_dic.hpp>

#include <algorithm>
#include <cmath>

namespace dic {

SubsetDIC::SubsetDIC(SubsetConfig config)
    : config_(config)
{
}

// ---------------------------------------------------------------------------
// compute without ROI: use full image as region of interest
// ---------------------------------------------------------------------------
std::vector<Displacement2D> SubsetDIC::compute(
    const Image& reference,
    const Image& deformed
) const
{
    Mask full_roi(reference.width(), reference.height());
    full_roi.fill(true);
    return compute(reference, deformed, full_roi);
}

// ---------------------------------------------------------------------------
// compute with explicit ROI
// ---------------------------------------------------------------------------
std::vector<Displacement2D> SubsetDIC::compute(
    const Image& reference,
    const Image& deformed,
    const Mask& roi
) const
{
    std::vector<Displacement2D> empty_result;

    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return empty_result;
    }

    // 1. Select seed candidates
    SeedSelector selector(config_);
    const auto seed_result = selector.select_best_seed(reference, deformed, roi);

    if (!seed_result.found) {
        return empty_result;
    }

    // 2. Use the single best seed (already quality-passed with largest
    //    displacement norm). The SeedSelector::evaluate_candidate already
    //    ran integer search + subpixel ICGN, so the full 6 affine parameters
    //    are ready — no need to re-run SubsetInitializer.
    const auto& best = seed_result.best_seed;
    PropagationSeed ps;
    ps.point = best.point;
    ps.displacement = best.displacement;

    std::vector<PropagationSeed> propagation_seeds{ps};

    // 3. Run reliability-guided propagation
    ReliabilityPropagation propagation(config_);
    const auto result = propagation.propagate(
        reference, deformed, roi, propagation_seeds);

    if (result.points_computed == 0) {
        return empty_result;
    }

    // 4. Return all grid points (both valid and invalid).
    //    Grid pre-seeded positions + validity flags let callers
    //    reconstruct the full-field displacement map including failed points.
    if (result.points_computed == 0) {
        return empty_result;
    }
    return result.points;
}

} // namespace dic
