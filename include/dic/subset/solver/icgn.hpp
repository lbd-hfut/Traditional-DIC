/**
 * @file icgn.hpp
 * @brief Inverse compositional Gauss-Newton Subset-DIC solver skeleton.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_ICGN_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_ICGN_HPP

#include <dic/interpolation/bspline.hpp>
#include <dic/subset/solver/subset_solver.hpp>
#include <dic/subset/subset_config.hpp>

namespace dic {

class ICGNSolver : public SubsetSolver {
public:
    explicit ICGNSolver(SubsetConfig config = {});
    Displacement2D solve(const Image& reference, const Image& deformed, const Eigen::Vector2d& point, const InitialDisplacement& initial) const override;
    Displacement2D solve_with_interpolators(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial,
                                            const BSplineInterpolator& reference_interpolator,
                                            const BSplineInterpolator& deformed_interpolator) const;
private:
    Displacement2D solve_first_order(const Image& reference, const Image& deformed, const Eigen::Vector2d& point, const InitialDisplacement& initial) const;
    Displacement2D solve_first_order(const Image& reference,
                                     const Image& deformed,
                                     const Eigen::Vector2d& point,
                                     const InitialDisplacement& initial,
                                     const BSplineInterpolator& reference_interpolator,
                                     const BSplineInterpolator& deformed_interpolator) const;
    Displacement2D solve_second_order_placeholder(const Eigen::Vector2d& point, const InitialDisplacement& initial) const;

    // TODO: Reference subset -> reference gradient -> shape Jacobian ->
    // steepest descent images -> Hessian precomputation -> warp deformed image
    // -> B-spline interpolation -> ZNSSD residual -> delta parameters ->
    // inverse compositional update -> convergence.
    Eigen::VectorXd extract_reference_subset(const Image& reference, const Eigen::Vector2d& point) const;
    Eigen::MatrixXd compute_reference_gradient(const Image& reference, const Eigen::Vector2d& point) const;
    Eigen::MatrixXd compute_steepest_descent_images() const;
    Eigen::MatrixXd compute_hessian(const Eigen::MatrixXd& steepest_descent) const;
    Eigen::VectorXd compute_residual() const;
    Eigen::VectorXd solve_parameter_increment(const Eigen::MatrixXd& hessian, const Eigen::VectorXd& residual) const;
    Eigen::VectorXd inverse_compositional_update(const Eigen::VectorXd& parameters, const Eigen::VectorXd& delta) const;
    bool check_convergence(const Eigen::VectorXd& delta) const;
    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_ICGN_HPP
