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
#include <dic/core/result.hpp>
#include <dic/initialization/initializer.hpp>

namespace dic {

class SubsetSolver {
public:
    virtual ~SubsetSolver() = default;
    virtual Displacement2D solve(const Image& reference, const Image& deformed, const Eigen::Vector2d& point, const InitialDisplacement& initial) const = 0;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_SUBSET_SOLVER_HPP
