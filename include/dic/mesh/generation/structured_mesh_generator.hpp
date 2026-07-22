/**
 * @file structured_mesh_generator.hpp
 * @brief Structured background grid mesh generation for ROI-defined domains.
 *
 * Responsibilities:
 * - Generate regular Q4/Q8-ready background meshes in image pixel coordinates.
 * - Clip generated elements against ROI membership without performing DIC solve.
 *
 * Inputs:
 * - ROI membership checks and MeshGenerationConfig sizing/element options.
 *
 * Outputs:
 * - Mesh nodes and element connectivity.
 *
 * Dependencies:
 * - MeshGenerator base interface.
 * - dic::Mesh and dic::Node topology containers.
 *
 * TODO:
 * - Compute real ROI bounding boxes once ROI exposes rectangle/polygon/mask data.
 * - Replace center-only classification with robust boundary intersection /
 *   coverage tests.
 * - Add Q8 edge -> midside node deduplication and quality checks.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_STRUCTURED_MESH_GENERATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_STRUCTURED_MESH_GENERATOR_HPP

#include <dic/mesh/generation/mesh_generator.hpp>
#include <dic/mesh/node.hpp>
#include <cstddef>
#include <vector>

namespace dic::mesh {

class StructuredMeshGenerator : public MeshGenerator {
public:
    Mesh generate(
        const ROI& roi,
        const MeshGenerationConfig& config
    ) const override;

private:
    std::vector<Node> generate_grid_nodes(
        const ROI& roi,
        double element_size
    ) const;

    void build_q4_connectivity(
        Mesh& mesh,
        std::size_t nx,
        std::size_t ny
    ) const;

    void build_q8_connectivity(
        Mesh& mesh,
        std::size_t nx,
        std::size_t ny
    ) const;

    void remove_outside_elements(
        Mesh& mesh,
        const ROI& roi
    ) const;

    void remove_unused_nodes(
        Mesh& mesh
    ) const;
};

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_STRUCTURED_MESH_GENERATOR_HPP
