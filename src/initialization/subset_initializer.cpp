#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/subset_initializer.hpp>
#include <dic/subset/solver/forward_gauss_newton.hpp>
#include <dic/subset/solver/icgn.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace dic {
namespace {

double zncc_from_znssd(double znssd)
{
    if (!std::isfinite(znssd)) {
        return -1.0;
    }
    return std::max(-1.0, std::min(1.0, 1.0 - 0.5 * znssd));
}

} // namespace

SubsetInitializer::SubsetInitializer(int search_radius)
{
    config_.seed_initialization.integer_search.search_radius = std::max(0, search_radius);
}

SubsetInitializer::SubsetInitializer(SubsetConfig config)
    : config_(config)
{
    config_.seed_initialization.integer_search.search_radius =
        std::max(0, config_.seed_initialization.integer_search.search_radius);
    config_.seed_initialization.integer_search.subset_radius =
        std::max(1, config_.seed_initialization.integer_search.subset_radius);
    config_.seed_initialization.subpixel.subset_radius =
        std::max(1, config_.seed_initialization.subpixel.subset_radius);
    config_.seed_initialization.subpixel.max_iterations =
        std::max(0, config_.seed_initialization.subpixel.max_iterations);
}

InitialDisplacement SubsetInitializer::estimate(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point
) const
{
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return invalid;
    }

    BSplineInterpolator reference_interpolator(nullptr);
    BSplineInterpolator deformed_interpolator(nullptr);
    std::optional<BSplinePrecomputedImage> deformed_precomputed;

    const auto& subpixel_config = config_.seed_initialization.subpixel;
    if (subpixel_config.enabled &&
        subpixel_config.optimizer == SubsetOptimizationMethod::ICGN) {
        auto precompute_config = config_.image_precompute;
        precompute_config.use_exact_prefilter = false;
        BSplineImagePreprocessor preprocessor(precompute_config);
        deformed_precomputed = preprocessor.compute_lazy(deformed);
        deformed_interpolator = BSplineInterpolator(&(*deformed_precomputed));
    }

    return estimate_with_interpolators(
        reference,
        deformed,
        point,
        reference_interpolator,
        deformed_interpolator
    );
}

InitialDisplacement SubsetInitializer::estimate_with_interpolators(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    IntegerSearchInitializer integer_search(config_.seed_initialization);
    const auto integer_initial = integer_search.estimate(reference, deformed, point);
    return refine_initial_with_interpolators(
        reference, deformed, point, integer_initial, reference_interpolator, deformed_interpolator);
}

InitialDisplacement SubsetInitializer::estimate_with_mask(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point
) const
{
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height()) {
        return invalid;
    }

    BSplineInterpolator reference_interpolator(nullptr);
    BSplineInterpolator deformed_interpolator(nullptr);
    std::optional<BSplinePrecomputedImage> deformed_precomputed;

    const auto& subpixel_config = config_.seed_initialization.subpixel;
    if (subpixel_config.enabled &&
        subpixel_config.optimizer == SubsetOptimizationMethod::ICGN) {
        auto precompute_config = config_.image_precompute;
        precompute_config.use_exact_prefilter = false;
        BSplineImagePreprocessor preprocessor(precompute_config);
        deformed_precomputed = preprocessor.compute_lazy(deformed);
        deformed_interpolator = BSplineInterpolator(&(*deformed_precomputed));
    }

    return estimate_with_mask_interpolators(
        reference,
        deformed,
        roi,
        point,
        reference_interpolator,
        deformed_interpolator
    );
}

InitialDisplacement SubsetInitializer::estimate_with_mask_interpolators(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    if (!config_.truncate_roi_subsets) {
        return estimate_with_interpolators(
            reference, deformed, point, reference_interpolator, deformed_interpolator);
    }

    IntegerSearchInitializer integer_search(config_.seed_initialization);
    const auto integer_initial = integer_search.estimate_with_mask_interpolators(
        reference, deformed, roi, point, reference_interpolator, deformed_interpolator);
    return refine_initial_with_mask_interpolators(
        reference, deformed, roi, point, integer_initial, reference_interpolator, deformed_interpolator);
}

InitialDisplacement SubsetInitializer::refine_initial_with_interpolators(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const InitialDisplacement& integer_initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    if (!integer_initial.valid || !config_.seed_initialization.subpixel.enabled) {
        return integer_initial;
    }

    const auto& integer_config = config_.seed_initialization.integer_search;
    const auto& subpixel_config = config_.seed_initialization.subpixel;

    SubsetConfig solver_config = config_;
    solver_config.subset_radius = subpixel_config.subset_radius;
    solver_config.search_radius = integer_config.search_radius;
    solver_config.convergence_threshold = subpixel_config.convergence_threshold;
    solver_config.max_iterations = subpixel_config.max_iterations;
    solver_config.shape_function = subpixel_config.shape_function;
    solver_config.optimizer = subpixel_config.optimizer;
    solver_config.objective = subpixel_config.objective;
    solver_config.use_second_order =
        subpixel_config.shape_function == SubsetShapeFunctionMethod::SecondOrder;

    Displacement2D refined;
    if (subpixel_config.optimizer == SubsetOptimizationMethod::ICGN) {
        const ICGNSolver solver(solver_config);
        refined = solver.solve_with_interpolators(
            reference,
            deformed,
            point,
            integer_initial,
            reference_interpolator,
            deformed_interpolator
        );
    } else {
        const ForwardGaussNewtonSolver solver(solver_config);
        refined = solver.solve(reference, deformed, point, integer_initial);
    }

    if (refined.valid && refined.status == SolverStatus::Success) {
        return {refined.u, refined.v,
                refined.du_dx, refined.du_dy,
                refined.dv_dx, refined.dv_dy,
                refined.correlation, true,
                zncc_from_znssd(refined.correlation),
                refined.correlation};
    }
    return integer_initial;
}

InitialDisplacement SubsetInitializer::refine_initial_with_mask_interpolators(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const InitialDisplacement& integer_initial,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    if (!config_.truncate_roi_subsets) {
        return refine_initial_with_interpolators(
            reference, deformed, point, integer_initial, reference_interpolator, deformed_interpolator);
    }
    if (!integer_initial.valid || !config_.seed_initialization.subpixel.enabled) {
        return integer_initial;
    }

    const auto& integer_config = config_.seed_initialization.integer_search;
    const auto& subpixel_config = config_.seed_initialization.subpixel;

    SubsetConfig solver_config = config_;
    solver_config.subset_radius = subpixel_config.subset_radius;
    solver_config.search_radius = integer_config.search_radius;
    solver_config.convergence_threshold = subpixel_config.convergence_threshold;
    solver_config.max_iterations = subpixel_config.max_iterations;
    solver_config.shape_function = subpixel_config.shape_function;
    solver_config.optimizer = subpixel_config.optimizer;
    solver_config.objective = subpixel_config.objective;
    solver_config.use_second_order =
        subpixel_config.shape_function == SubsetShapeFunctionMethod::SecondOrder;

    Displacement2D refined;
    if (subpixel_config.optimizer == SubsetOptimizationMethod::ICGN) {
        const ICGNSolver solver(solver_config);
        refined = solver.solve_with_mask(
            reference,
            deformed,
            roi,
            point,
            integer_initial,
            reference_interpolator,
            deformed_interpolator
        );
    } else {
        const ForwardGaussNewtonSolver solver(solver_config);
        refined = solver.solve(reference, deformed, point, integer_initial);
    }

    if (refined.valid && refined.status == SolverStatus::Success) {
        return {refined.u, refined.v,
                refined.du_dx, refined.du_dy,
                refined.dv_dx, refined.dv_dy,
                refined.correlation, true,
                zncc_from_znssd(refined.correlation),
                refined.correlation};
    }
    return integer_initial;
}

} // namespace dic
