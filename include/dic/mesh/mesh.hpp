/**
 * @file mesh.hpp
 * @brief 2D mesh topology and node container.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_HPP

#include <dic/mesh/node.hpp>
#include <vector>

namespace dic {

class Mesh { public: std::vector<Node>& nodes(); const std::vector<Node>& nodes() const; void add_node(const Node& node); private: std::vector<Node> nodes_; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_HPP
