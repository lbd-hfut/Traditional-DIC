/**
 * @file natural_coordinate.hpp
 * @brief Natural coordinate result for isoparametric elements.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_NATURAL_COORDINATE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_NATURAL_COORDINATE_HPP



namespace dic {

struct NaturalCoordinate { double xi{0.0}; double eta{0.0}; bool converged{false}; int iterations{0}; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_NATURAL_COORDINATE_HPP
