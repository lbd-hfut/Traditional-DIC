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
    if (config_.shape_function == SubsetShapeFunctionMethod::SecondOrder || config_.use_second_order) {
        return solve_second_order_placeholder(point, initial);
    }
    return solve_first_order_placeholder(point, initial);
}

Displacement2D ForwardGaussNewtonSolver::solve_first_order_placeholder(
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

Displacement2D ForwardGaussNewtonSolver::solve_second_order_placeholder(
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
