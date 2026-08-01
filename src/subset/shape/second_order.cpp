/**
 * @file second_order.cpp
 * @brief Minimal implementation placeholder for second_order.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/subset/shape/second_order.hpp>
#include <stdexcept>

namespace dic {

int SecondOrderShapeFunction::parameter_count() const { return 12; }

Eigen::Vector2d SecondOrderShapeFunction::warp(const Eigen::Vector2d& local_point, const Eigen::VectorXd& parameters) const
{
    if (parameters.size() < 12) {
        throw std::invalid_argument("Insufficient shape parameters.");
    }

    const double x = local_point.x();
    const double y = local_point.y();
    const double x2 = x * x;
    const double y2 = y * y;
    const double xy = x * y;

    // Parameters: [u, v, du_dx, du_dy, dv_dx, dv_dy,
    //              d2u_dx2, d2u_dxdy, d2u_dy2, d2v_dx2, d2v_dxdy, d2v_dy2]
    return {
        x + parameters(0) + parameters(2) * x + parameters(3) * y
          + parameters(6) * x2 * 0.5 + parameters(7) * xy + parameters(8) * y2 * 0.5,
        y + parameters(1) + parameters(4) * x + parameters(5) * y
          + parameters(9) * x2 * 0.5 + parameters(10) * xy + parameters(11) * y2 * 0.5
    };
}

Eigen::MatrixXd SecondOrderShapeFunction::jacobian(const Eigen::Vector2d& local_point) const
{
    const double x = local_point.x();
    const double y = local_point.y();
    const double x2_2 = x * x * 0.5;
    const double y2_2 = y * y * 0.5;
    const double xy = x * y;

    // 2×12 Jacobian: ∂warp/∂p
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 12);

    // Row 0 — dwarp_x / dp
    J(0, 0) = 1.0;       // ∂/∂u
    J(0, 2) = x;          // ∂/∂du_dx
    J(0, 3) = y;          // ∂/∂du_dy
    J(0, 6) = x2_2;       // ∂/∂d2u_dx2
    J(0, 7) = xy;         // ∂/∂d2u_dxdy
    J(0, 8) = y2_2;       // ∂/∂d2u_dy2

    // Row 1 — dwarp_y / dp
    J(1, 1) = 1.0;       // ∂/∂v
    J(1, 4) = x;          // ∂/∂dv_dx
    J(1, 5) = y;          // ∂/∂dv_dy
    J(1, 9) = x2_2;       // ∂/∂d2v_dx2
    J(1, 10) = xy;        // ∂/∂d2v_dxdy
    J(1, 11) = y2_2;      // ∂/∂d2v_dy2

    return J;
}

} // namespace dic
