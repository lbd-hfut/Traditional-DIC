/**
 * @file mesh_generation_config.hpp
 * @brief Configuration types for ROI/Mask-driven 2D mesh generation.
 *
 * Responsibilities:
 * - Define mesh generation element choices and strategy selection.
 * - Keep mesh generation settings independent from Mesh-DIC optimization.
 *
 * Inputs:
 * - ROI geometry, mask membership, and image-plane sizing parameters.
 *
 * Outputs:
 * - Configuration consumed by MeshGenerator implementations.
 *
 * Dependencies:
 * - No third-party dependency; all sizes are image-plane pixel units.
 *
 * TODO:
 * - Add local refinement controls, boundary sizing fields, and quality policy.
 * - Add schema validation for YAML/Python configuration loading.
 */

#pragma once

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_GENERATION_CONFIG_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_GENERATION_CONFIG_HPP

#include <string>

namespace dic::mesh {

enum class MeshElementType {
    T3,
    Q4,
    Q8
};

enum class MeshGenerationMethod {
    Structured,
    Unstructured,
    Manual,
    Auto
};

/**
 * @brief Mesh generation options for 2D image-domain meshes.
 *
 * All 2D mesh coordinates are expressed in image pixels by default. The target
 * element size is a characteristic size on the image plane, not a physical
 * world-space length. This skeleton only defines configuration and interfaces;
 * future work can extend it with local refinement and boundary layer controls.
 */
struct MeshGenerationConfig {
    MeshElementType element_type{MeshElementType::Q4};
    MeshGenerationMethod method{MeshGenerationMethod::Auto};

    // Target characteristic element size in image pixels.
    double target_element_size{20.0};
    double min_element_size{0.0};
    double max_element_size{0.0};

    // Manual mesh input files. Required when method == Manual.
    // Format:
    //   nodes_file:    node_id, x, y
    //   elements_file: element_id, n1, n2, ...
    // Node ids in elements_file are 1-based, matching common exported mesh
    // tables and the diagnostic tools in tools/.
    std::string nodes_file{};
    std::string elements_file{};

    // Whether the generated mesh should follow the ROI boundary.
    bool fit_roi_boundary{true};

    // Remove elements whose domain lies outside the ROI.
    bool remove_outside_elements{true};

    // Remove unused / outside nodes after element filtering.
    bool remove_outside_nodes{true};

    // Whether partially intersecting boundary elements are allowed.
    bool allow_partial_elements{false};

    // Minimum accepted element quality.
    double min_element_quality{0.1};
    double max_aspect_ratio{6.0};
    double min_jacobian{0.05};
};

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_GENERATION_CONFIG_HPP
