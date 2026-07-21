/**
 * @file node.hpp
 * @brief Mesh node data for 2D Mesh-DIC.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_NODE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_NODE_HPP

#include <Eigen/Dense>
#include <cstddef>

namespace dic {

struct Node { std::size_t id{0}; Eigen::Vector2d coordinate{0.0, 0.0}; Eigen::Vector2d displacement{0.0, 0.0}; bool fixed{false}; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_NODE_HPP
