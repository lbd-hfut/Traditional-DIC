/**
 * @file first_order.hpp
 * @brief FirstOrderShapeFunction warp model.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SHAPE_FIRST_ORDER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SHAPE_FIRST_ORDER_HPP

#include <dic/subset/shape/shape_function.hpp>

namespace dic {

class FirstOrderShapeFunction : public ShapeFunction {
public:
    int parameter_count() const override;
    Eigen::Vector2d warp(const Eigen::Vector2d& local_point, const Eigen::VectorXd& parameters) const override;
    Eigen::MatrixXd jacobian(const Eigen::Vector2d& local_point) const override;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_SUBSET_SHAPE_FIRST_ORDER_HPP
