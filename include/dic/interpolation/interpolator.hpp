/**
 * @file interpolator.hpp
 * @brief Abstract subpixel interpolation interface.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_INTERPOLATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_INTERPOLATOR_HPP

#include <Eigen/Dense>

namespace dic {

class Interpolator {
public:
    virtual ~Interpolator() = default;
    virtual double value(double x, double y) const = 0;
    virtual Eigen::Vector2d gradient(double x, double y) const = 0;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_INTERPOLATOR_HPP
