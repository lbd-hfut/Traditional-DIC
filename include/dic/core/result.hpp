/**
 * @file result.hpp
 * @brief 2D and 3D DIC result containers.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP



namespace dic {

struct Displacement2D { double x{0.0}; double y{0.0}; double u{0.0}; double v{0.0}; double correlation{0.0}; bool valid{false}; };
struct Displacement3D { double X{0.0}; double Y{0.0}; double Z{0.0}; double U{0.0}; double V{0.0}; double W{0.0}; bool valid{false}; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP
