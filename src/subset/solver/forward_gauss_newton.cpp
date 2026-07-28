#include <dic/subset/solver/forward_gauss_newton.hpp>

namespace dic {

ForwardGaussNewtonSolver::ForwardGaussNewtonSolver(SubsetConfig config)
    : config_(config)
{
}

Displacement2D ForwardGaussNewtonSolver::solve(const Image& reference,
                                               const Image& deformed,
                                               const Eigen::Vector2d& point,
                                               const InitialDisplacement& initial) const
{
    (void)reference;
    (void)deformed;
    const bool second_order = config_.shape_function == SubsetShapeFunctionMethod::SecondOrder ||
                              config_.use_second_order;
    if (second_order) {
        if (config_.objective == CorrelationCriterionKind::SSD) {
            return solve_second_order_ssd_placeholder(point, initial);
        }
        return solve_second_order_znssd_placeholder(point, initial);
    }
    if (config_.objective == CorrelationCriterionKind::SSD) {
        return solve_first_order_ssd_placeholder(point, initial);
    }
    return solve_first_order_znssd_placeholder(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_first_order_znssd_placeholder(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    return solve_unimplemented(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_first_order_ssd_placeholder(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    return solve_unimplemented(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_second_order_znssd_placeholder(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    return solve_unimplemented(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_second_order_ssd_placeholder(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    return solve_unimplemented(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_unimplemented(
    const Eigen::Vector2d& point,
    const InitialDisplacement& initial
) const
{
    Displacement2D result;
    result.x = point.x();
    result.y = point.y();
    result.u = initial.u;
    result.v = initial.v;
    result.correlation = initial.confidence;
    result.status = SolverStatus::NotConverged;
    result.valid = false;
    return result;
}

} // namespace dic
