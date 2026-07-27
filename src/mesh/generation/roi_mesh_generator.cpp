/**
 * @file roi_mesh_generator.cpp
 * @brief High-level dispatcher for ROI-based mesh generation.
 *
 * Responsibilities:
 * - Select structured or unstructured mesh generation.
 * - Keep Mesh Generation independent from displacement initialization.
 *
 * Inputs:
 * - ROI membership and mesh generation configuration.
 *
 * Outputs:
 * - Generated Mesh topology.
 *
 * Dependencies:
 * - StructuredMeshGenerator and UnstructuredMeshGenerator.
 *
 * TODO:
 * - Add ROI geometry analysis and automatic structured/unstructured strategy
 *   selection.
 * - Keep Auto usable by falling back to structured background mesh plus ROI
 *   clipping while unstructured generation is incomplete.
 */

#include <dic/mesh/generation/roi_mesh_generator.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dic::mesh {
namespace {

int nodes_per_element(MeshElementType type)
{
    switch (type) {
    case MeshElementType::T3:
        return 3;
    case MeshElementType::Q4:
        return 4;
    case MeshElementType::Q8:
        return 8;
    }
    return 4;
}

Mesh load_manual_mesh(const MeshGenerationConfig& config)
{
    if (config.nodes_file.empty() || config.elements_file.empty()) {
        throw std::invalid_argument(
            "Manual mesh generation requires mesh_generation.nodes_file and mesh_generation.elements_file.");
    }

    Mesh mesh;
    {
        std::ifstream in(config.nodes_file);
        if (!in) {
            throw std::runtime_error("Failed to open manual mesh nodes_file: " + config.nodes_file);
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            std::size_t id = 0;
            double x = 0.0;
            double y = 0.0;
            if (iss >> id >> x >> y) {
                Node node;
                node.id = mesh.nodes().size();
                node.coordinate = Eigen::Vector2d{x, y};
                mesh.add_node(node);
            }
        }
    }

    const int nn = nodes_per_element(config.element_type);
    {
        std::ifstream in(config.elements_file);
        if (!in) {
            throw std::runtime_error("Failed to open manual mesh elements_file: " + config.elements_file);
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            std::size_t element_id = 0;
            if (!(iss >> element_id)) {
                continue;
            }

            MeshElementConnectivity element;
            element.type = config.element_type;
            element.node_ids.reserve(static_cast<std::size_t>(nn));
            for (int i = 0; i < nn; ++i) {
                std::size_t node_id = 0;
                if (!(iss >> node_id)) {
                    throw std::runtime_error("Manual mesh element row has too few node ids: " + line);
                }
                if (node_id == 0 || node_id > mesh.nodes().size()) {
                    throw std::runtime_error("Manual mesh element references an out-of-range node id: " + line);
                }
                element.node_ids.push_back(node_id - 1U);
            }
            mesh.add_element(element);
        }
    }

    return mesh;
}

} // namespace

Mesh ROIMeshGenerator::generate(
    const ROI& roi,
    const MeshGenerationConfig& config
) const
{
    switch (config.method) {
    case MeshGenerationMethod::Structured:
        return structured_generator_.generate(roi, config);
    case MeshGenerationMethod::Unstructured:
        return unstructured_generator_.generate(roi, config);
    case MeshGenerationMethod::Manual:
        return load_manual_mesh(config);
    case MeshGenerationMethod::Auto:
        // TODO: Detect rectangular ROI -> Structured; irregular ROI ->
        // structured background mesh plus ROI clipping or Unstructured when
        // available. Current ROI API only exposes contains(), so Auto remains
        // usable through StructuredMeshGenerator.
        return structured_generator_.generate(roi, config);
    }

    return structured_generator_.generate(roi, config);
}

} // namespace dic::mesh
