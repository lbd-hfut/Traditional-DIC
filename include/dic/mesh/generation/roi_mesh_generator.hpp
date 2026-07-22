/**
 * @file roi_mesh_generator.hpp
 * @brief High-level ROI mesh generation controller.
 *
 * Responsibilities:
 * - Dispatch mesh generation to structured or unstructured strategies.
 * - Keep mesh generation separate from displacement initialization and solvers.
 *
 * Inputs:
 * - ROI membership and MeshGenerationConfig strategy options.
 *
 * Outputs:
 * - Generated Mesh with node coordinates and element connectivity.
 *
 * Dependencies:
 * - StructuredMeshGenerator and UnstructuredMeshGenerator.
 *
 * TODO:
 * - Add ROI geometry analysis and automatic structured/unstructured strategy
 *   selection.
 * - Add mask-aware generation when ROI exposes unified mask metadata.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ROI_MESH_GENERATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ROI_MESH_GENERATOR_HPP

#include <dic/mesh/generation/structured_mesh_generator.hpp>
#include <dic/mesh/generation/unstructured_mesh_generator.hpp>

namespace dic::mesh {

class ROIMeshGenerator {
public:
    Mesh generate(
        const ROI& roi,
        const MeshGenerationConfig& config
    ) const;

private:
    StructuredMeshGenerator structured_generator_;
    UnstructuredMeshGenerator unstructured_generator_;
};

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ROI_MESH_GENERATOR_HPP
