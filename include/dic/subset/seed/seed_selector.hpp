/**
 * @file seed_selector.hpp
 * @brief Automatic seed selection for reliability-guided Subset-DIC.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/result.hpp>
#include <dic/subset/subset_config.hpp>

#include <Eigen/Dense>
#include <vector>

namespace dic {

class SubsetInitializer;

struct SeedEvaluation {
    Eigen::Vector2d point{0.0, 0.0};
    Displacement2D displacement{};
    double quality{0.0};
    double displacement_norm{0.0};
    bool quality_passed{false};
    bool valid{false};
};

struct SeedSelectionResult {
    std::vector<SeedEvaluation> candidates;
    SeedEvaluation best_seed{};
    bool found{false};
};

class SeedSelector {
public:
    explicit SeedSelector(SubsetConfig config = {});

    SeedSelectionResult select_best_seed(const Image& reference,
                                         const Image& deformed,
                                         const Mask& roi) const;

    std::vector<Eigen::Vector2d> generate_candidates(const Image& reference,
                                                     const Mask& roi) const;
    std::vector<Eigen::Vector2d> generate_uniform_candidates(const Mask& roi) const;

private:
    bool is_candidate_margin_valid(const Mask& roi, int x, int y) const;
    double local_texture_std(const Image& reference, int x, int y) const;
    bool quality_passes(double quality) const;
    SeedEvaluation evaluate_candidate(const Image& reference,
                                      const Image& deformed,
                                      const Eigen::Vector2d& point,
                                      const SubsetInitializer& initializer,
                                      const BSplineInterpolator& reference_interpolator,
                                      const BSplineInterpolator& deformed_interpolator) const;

    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SEED_SEED_SELECTOR_HPP
