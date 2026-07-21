/**
 * @file global_to_natural.hpp
 * @brief Global-to-natural coordinate conversion skeleton.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_GLOBAL_TO_NATURAL_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_GLOBAL_TO_NATURAL_HPP

#include <dic/mesh/coordinate/natural_coordinate.hpp>
#include <dic/mesh/element/element.hpp>
#include <dic/mesh/node.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

// TODO: initial guess -> evaluate x(xi,eta), y(xi,eta) -> residual ->
// Jacobian -> Newton-Raphson -> convergence check. This path is especially
// important for Q8 elements.
NaturalCoordinate global_to_natural(const Element& element, const std::vector<Node>& nodes, const Eigen::Vector2d& global_point);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_COORDINATE_GLOBAL_TO_NATURAL_HPP
