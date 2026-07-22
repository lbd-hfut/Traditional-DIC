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

#include <dic/mesh/mesh_generation_config.hpp>
#include <dic/mesh/node.hpp>
#include <cstddef>
#include <vector>

namespace dic {

/**
 * @brief Connectivity record for one generated 2D mesh element.
 *
 * Node ids are indices into Mesh::nodes(). The order is expected to be
 * counter-clockwise for area elements. Q8 elements should share midside nodes
 * through generation-time edge deduplication.
 */
struct MeshElementConnectivity {
    mesh::MeshElementType type{mesh::MeshElementType::Q4};
    std::vector<std::size_t> node_ids;
};

class Mesh {
public:
    std::vector<Node>& nodes();
    const std::vector<Node>& nodes() const;

    std::vector<MeshElementConnectivity>& elements();
    const std::vector<MeshElementConnectivity>& elements() const;

    void add_node(const Node& node);
    void add_element(const MeshElementConnectivity& element);
    void clear();

private:
    std::vector<Node> nodes_;
    std::vector<MeshElementConnectivity> elements_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_HPP
