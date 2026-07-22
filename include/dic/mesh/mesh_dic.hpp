/**
 * @file mesh_dic.hpp
 * @brief High-level 2D Mesh-DIC controller.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_DIC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_DIC_HPP

#include <dic/core/image.hpp>
#include <dic/core/roi.hpp>
#include <dic/mesh/mesh_generation_config.hpp>
#include <dic/core/result.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <vector>

namespace dic {

class MeshDIC {
public:
    explicit MeshDIC(MeshConfig config = {});

    std::vector<Displacement2D> compute(
        const Image& reference,
        const Image& deformed,
        Mesh mesh
    ) const;

    /**
     * @brief Future convenience path for ROI -> Mesh Generation -> Mesh-DIC.
     *
     * This overload only reserves orchestration. Mesh generation initializes
     * node coordinates, element topology, and connectivity. Displacement
     * initialization for u0/v0 remains a separate Mesh-DIC stage.
     */
    std::vector<Displacement2D> compute(
        const Image& reference,
        const Image& deformed,
        const ROI& roi,
        const mesh::MeshGenerationConfig& mesh_generation_config
    ) const;

private:
    MeshConfig config_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_DIC_HPP
