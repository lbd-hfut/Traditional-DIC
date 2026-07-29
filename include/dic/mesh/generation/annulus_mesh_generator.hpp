/**
 * @file annulus_mesh_generator.hpp
 * @brief Annulus ROI mesh generation utilities for T3/Q4/Q8 image-domain meshes.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ANNULUS_MESH_GENERATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ANNULUS_MESH_GENERATOR_HPP

#include <dic/core/mask.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/mesh_generation_config.hpp>

#include <array>

namespace dic::mesh {

struct AnnulusMeshGenerationSummary {
    double center_x{0.0};
    double center_y{0.0};
    double inner_radius{0.0};
    double outer_radius{0.0};
    int radial_divisions{0};
    int circumferential_divisions{0};
    double target_element_size{0.0};
    double min_element_size{0.0};
    double max_element_size{0.0};
};

struct AnnulusMeshGenerationResult {
    Mesh t3;
    Mesh q4;
    Mesh q8;
    AnnulusMeshGenerationSummary summary;
};

AnnulusMeshGenerationResult generate_annulus_meshes_from_mask(
    const Mask& mask,
    const MeshGenerationConfig& config);

Mesh generate_annulus_mesh_from_mask(
    const Mask& mask,
    const MeshGenerationConfig& config);

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_ANNULUS_MESH_GENERATOR_HPP
