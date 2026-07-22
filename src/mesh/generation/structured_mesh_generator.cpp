/**
 * @file structured_mesh_generator.cpp
 * @brief Minimal structured ROI mesh generation implementation.
 *
 * Responsibilities:
 * - Generate a regular background grid and Q4 connectivity.
 * - Provide linkable placeholders for Q8 and robust ROI clipping.
 *
 * Inputs:
 * - ROI membership queries and mesh generation configuration.
 *
 * Outputs:
 * - Mesh node coordinates and element connectivity.
 *
 * Dependencies:
 * - dic::Mesh, dic::Node, and ROI::contains.
 *
 * TODO:
 * - Input ROI -> compute ROI bounding box -> determine Nx/Ny from
 *   target_element_size -> generate regular grid nodes -> generate Q4/Q8
 *   connectivity -> check element against ROI -> remove outside elements ->
 *   remove orphan nodes -> check element quality -> return Mesh.
 * - Replace center-only classification with robust boundary intersection /
 *   coverage test.
 * - Implement Q8 edge -> midside node deduplication so adjacent elements share
 *   midside nodes.
 */

#include <dic/mesh/generation/structured_mesh_generator.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace dic::mesh {
namespace {

std::size_t grid_index(std::size_t i, std::size_t j, std::size_t nx)
{
    return j * (nx + 1U) + i;
}

} // namespace

Mesh StructuredMeshGenerator::generate(
    const ROI& roi,
    const MeshGenerationConfig& config
) const
{
    Mesh mesh;

    const double element_size = config.target_element_size > 0.0
        ? config.target_element_size
        : 20.0;

    auto nodes = generate_grid_nodes(roi, element_size);
    for (auto& node : nodes) {
        mesh.add_node(node);
    }

    // TODO: Replace this placeholder grid size inference once ROI exposes a
    // bounding box. The current skeleton creates one cell from four nodes.
    const std::size_t nx = 1;
    const std::size_t ny = 1;

    switch (config.element_type) {
    case MeshElementType::Q4:
        build_q4_connectivity(mesh, nx, ny);
        break;
    case MeshElementType::Q8:
        build_q8_connectivity(mesh, nx, ny);
        break;
    case MeshElementType::T3:
        throw std::runtime_error("Structured T3 mesh generation is not implemented yet.");
    }

    if (config.remove_outside_elements) {
        remove_outside_elements(mesh, roi);
    }
    if (config.remove_outside_nodes) {
        remove_unused_nodes(mesh);
    }

    return mesh;
}

std::vector<Node> StructuredMeshGenerator::generate_grid_nodes(
    const ROI& roi,
    double element_size
) const
{
    (void)roi;

    // TODO: Compute RectangleROI/PolygonROI/MaskROI bounding boxes from ROI.
    // All coordinates are image pixels. This placeholder returns a single Q4
    // cell background grid sized by target_element_size.
    std::vector<Node> nodes;
    nodes.reserve(4);
    nodes.push_back(Node{0, Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{0.0, 0.0}, false});
    nodes.push_back(Node{1, Eigen::Vector2d{element_size, 0.0}, Eigen::Vector2d{0.0, 0.0}, false});
    nodes.push_back(Node{2, Eigen::Vector2d{0.0, element_size}, Eigen::Vector2d{0.0, 0.0}, false});
    nodes.push_back(Node{3, Eigen::Vector2d{element_size, element_size}, Eigen::Vector2d{0.0, 0.0}, false});
    return nodes;
}

void StructuredMeshGenerator::build_q4_connectivity(
    Mesh& mesh,
    std::size_t nx,
    std::size_t ny
) const
{
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            // Counter-clockwise in image-grid traversal convention:
            // top-left -> bottom-left -> bottom-right -> top-right.
            mesh.add_element(MeshElementConnectivity{
                MeshElementType::Q4,
                {
                    grid_index(i, j, nx),
                    grid_index(i, j + 1U, nx),
                    grid_index(i + 1U, j + 1U, nx),
                    grid_index(i + 1U, j, nx)
                }
            });
        }
    }
}

void StructuredMeshGenerator::build_q8_connectivity(
    Mesh& mesh,
    std::size_t nx,
    std::size_t ny
) const
{
    (void)mesh;
    (void)nx;
    (void)ny;
    throw std::runtime_error("Structured Q8 mesh generation is not implemented yet.");
}

void StructuredMeshGenerator::remove_outside_elements(
    Mesh& mesh,
    const ROI& roi
) const
{
    auto& elements = mesh.elements();
    const auto& nodes = mesh.nodes();

    elements.erase(
        std::remove_if(elements.begin(), elements.end(), [&](const MeshElementConnectivity& element) {
            if (element.node_ids.empty()) {
                return true;
            }

            Eigen::Vector2d center{0.0, 0.0};
            for (const auto node_id : element.node_ids) {
                if (node_id >= nodes.size()) {
                    return true;
                }
                center += nodes[node_id].coordinate;
            }
            center /= static_cast<double>(element.node_ids.size());

            // TODO: Replace center-only classification with robust boundary
            // intersection / coverage test using element nodes, masks, and
            // polygon clipping.
            return !roi.contains(center);
        }),
        elements.end()
    );
}

void StructuredMeshGenerator::remove_unused_nodes(
    Mesh& mesh
) const
{
    const auto& old_nodes = mesh.nodes();
    auto& elements = mesh.elements();

    std::vector<bool> used(old_nodes.size(), false);
    for (const auto& element : elements) {
        for (const auto node_id : element.node_ids) {
            if (node_id < used.size()) {
                used[node_id] = true;
            }
        }
    }

    std::vector<std::size_t> remap(old_nodes.size(), 0);
    std::vector<Node> compact_nodes;
    compact_nodes.reserve(old_nodes.size());
    for (std::size_t old_id = 0; old_id < old_nodes.size(); ++old_id) {
        if (!used[old_id]) {
            continue;
        }
        remap[old_id] = compact_nodes.size();
        Node node = old_nodes[old_id];
        node.id = compact_nodes.size();
        compact_nodes.push_back(node);
    }

    for (auto& element : elements) {
        for (auto& node_id : element.node_ids) {
            node_id = remap[node_id];
        }
    }

    mesh.nodes() = std::move(compact_nodes);
}

} // namespace dic::mesh
