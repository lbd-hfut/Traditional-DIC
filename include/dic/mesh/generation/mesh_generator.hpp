/**
 * @file mesh_generator.hpp
 * @brief Abstract interface for ROI/Mask-driven mesh generation.
 *
 * Responsibilities:
 * - Define the common mesh generation contract.
 * - Keep mesh topology creation separate from DIC displacement optimization.
 *
 * Inputs:
 * - ROI membership queries and mesh generation configuration.
 *
 * Outputs:
 * - Mesh nodes, element connectivity, and topology in image-pixel coordinates.
 *
 * Dependencies:
 * - dic::ROI for point-in-region tests.
 * - dic::Mesh for generated topology storage.
 *
 * TODO:
 * - Add mask-aware generation overloads once Mask ROI has a unified API.
 * - Add validation for generated node ids and element quality.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_MESH_GENERATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_MESH_GENERATOR_HPP

#include <dic/core/roi.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/mesh_generation_config.hpp>

namespace dic::mesh {

class MeshGenerator {
public:
    virtual ~MeshGenerator() = default;

    virtual Mesh generate(
        const ROI& roi,
        const MeshGenerationConfig& config
    ) const = 0;
};

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_MESH_GENERATOR_HPP
