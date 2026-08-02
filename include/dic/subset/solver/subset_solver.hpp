/**
 * @file subset_solver.hpp
 * @brief Abstract Subset-DIC solver interface.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_SUBSET_SOLVER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_SUBSET_SOLVER_HPP

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/result.hpp>
#include <dic/initialization/initializer.hpp>

namespace dic {

class BSplineInterpolator;

class SubsetSolver {
public:
    virtual ~SubsetSolver() = default;
    virtual Displacement2D solve(const Image& reference, const Image& deformed, const Eigen::Vector2d& point, const InitialDisplacement& initial) const = 0;

    virtual Displacement2D solve_with_interpolators(const Image& reference,
                                                     const Image& deformed,
                                                     const Eigen::Vector2d& point,
                                                     const InitialDisplacement& initial,
                                                     const BSplineInterpolator& reference_interpolator,
                                                     const BSplineInterpolator& deformed_interpolator) const
    {
        (void)reference_interpolator;
        (void)deformed_interpolator;
        return solve(reference, deformed, point, initial);
    }

    virtual Displacement2D solve_with_mask(const Image& reference,
                                            const Image& deformed,
                                            const Mask& roi,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial,
                                            const BSplineInterpolator& reference_interpolator,
                                            const BSplineInterpolator& deformed_interpolator) const
    {
        (void)roi;
        (void)reference_interpolator;
        (void)deformed_interpolator;
        return solve(reference, deformed, point, initial);
    }
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_SUBSET_SOLVER_HPP
