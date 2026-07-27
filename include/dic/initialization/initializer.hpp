/**
 * @file initializer.hpp
 * @brief Initial displacement estimation interface.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INITIALIZER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INITIALIZER_HPP

#include <dic/core/image.hpp>
#include <Eigen/Dense>

namespace dic {

struct InitialDisplacement { double u{0.0}; double v{0.0}; double du_dx{0.0}; double du_dy{0.0}; double dv_dx{0.0}; double dv_dy{0.0}; double confidence{0.0}; bool valid{false}; };
class Initializer {
public:
    virtual ~Initializer() = default;
    virtual InitialDisplacement estimate(const Image& reference, const Image& deformed, const Eigen::Vector2d& point) const = 0;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_INITIALIZER_HPP
