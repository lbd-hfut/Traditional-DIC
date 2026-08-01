#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP

#include <dic/core/mask.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/subset/solver/subset_solver.hpp>
#include <dic/subset/subset_config.hpp>

namespace dic {

class ForwardGaussNewtonSolver : public SubsetSolver {
public:
    explicit ForwardGaussNewtonSolver(SubsetConfig config = {});
    Displacement2D solve(const Image& reference,
                         const Image& deformed,
                         const Eigen::Vector2d& point,
                         const InitialDisplacement& initial) const override;

    Displacement2D solve_with_interpolators(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial,
                                            const BSplineInterpolator& reference_interpolator,
                                            const BSplineInterpolator& deformed_interpolator) const override;

    Displacement2D solve_with_mask(const Image& reference,
                                    const Image& deformed,
                                    const Mask& roi,
                                    const Eigen::Vector2d& point,
                                    const InitialDisplacement& initial,
                                    const BSplineInterpolator& reference_interpolator,
                                    const BSplineInterpolator& deformed_interpolator) const override;

private:
    Displacement2D solve_first_order_znssd(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial) const;
    Displacement2D solve_first_order_znssd(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial,
                                            const BSplineInterpolator& reference_interpolator,
                                            const BSplineInterpolator& deformed_interpolator) const;
    Displacement2D solve_first_order_znssd_masked(const Image& reference,
                                                   const Image& deformed,
                                                   const Mask& roi,
                                                   const Eigen::Vector2d& point,
                                                   const InitialDisplacement& initial,
                                                   const BSplineInterpolator& reference_interpolator,
                                                   const BSplineInterpolator& deformed_interpolator) const;

    Displacement2D solve_first_order_ssd(const Image& reference,
                                           const Image& deformed,
                                           const Eigen::Vector2d& point,
                                           const InitialDisplacement& initial) const;
    Displacement2D solve_first_order_ssd(const Image& reference,
                                           const Image& deformed,
                                           const Eigen::Vector2d& point,
                                           const InitialDisplacement& initial,
                                           const BSplineInterpolator& reference_interpolator,
                                           const BSplineInterpolator& deformed_interpolator) const;
    Displacement2D solve_first_order_ssd_masked(const Image& reference,
                                                  const Image& deformed,
                                                  const Mask& roi,
                                                  const Eigen::Vector2d& point,
                                                  const InitialDisplacement& initial,
                                                  const BSplineInterpolator& reference_interpolator,
                                                  const BSplineInterpolator& deformed_interpolator) const;

    Displacement2D solve_second_order_znssd(const Image& reference,
                                              const Image& deformed,
                                              const Eigen::Vector2d& point,
                                              const InitialDisplacement& initial) const;
    Displacement2D solve_second_order_znssd(const Image& reference,
                                              const Image& deformed,
                                              const Eigen::Vector2d& point,
                                              const InitialDisplacement& initial,
                                              const BSplineInterpolator& reference_interpolator,
                                              const BSplineInterpolator& deformed_interpolator) const;
    Displacement2D solve_second_order_znssd_masked(const Image& reference,
                                                     const Image& deformed,
                                                     const Mask& roi,
                                                     const Eigen::Vector2d& point,
                                                     const InitialDisplacement& initial,
                                                     const BSplineInterpolator& reference_interpolator,
                                                     const BSplineInterpolator& deformed_interpolator) const;

    Displacement2D solve_second_order_ssd(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial) const;
    Displacement2D solve_second_order_ssd(const Image& reference,
                                            const Image& deformed,
                                            const Eigen::Vector2d& point,
                                            const InitialDisplacement& initial,
                                            const BSplineInterpolator& reference_interpolator,
                                            const BSplineInterpolator& deformed_interpolator) const;
    Displacement2D solve_second_order_ssd_masked(const Image& reference,
                                                   const Image& deformed,
                                                   const Mask& roi,
                                                   const Eigen::Vector2d& point,
                                                   const InitialDisplacement& initial,
                                                   const BSplineInterpolator& reference_interpolator,
                                                   const BSplineInterpolator& deformed_interpolator) const;
    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP
