#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP

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

private:
    Displacement2D solve_first_order_znssd_placeholder(const Eigen::Vector2d& point,
                                                       const InitialDisplacement& initial) const;
    Displacement2D solve_first_order_ssd_placeholder(const Eigen::Vector2d& point,
                                                     const InitialDisplacement& initial) const;
    Displacement2D solve_second_order_znssd_placeholder(const Eigen::Vector2d& point,
                                                        const InitialDisplacement& initial) const;
    Displacement2D solve_second_order_ssd_placeholder(const Eigen::Vector2d& point,
                                                      const InitialDisplacement& initial) const;
    Displacement2D solve_unimplemented(const Eigen::Vector2d& point,
                                       const InitialDisplacement& initial) const;
    SubsetConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SOLVER_FORWARD_GAUSS_NEWTON_HPP
