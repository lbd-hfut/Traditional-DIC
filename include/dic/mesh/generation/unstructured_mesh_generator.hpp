/**
 * @file unstructured_mesh_generator.hpp
 * @brief Future unstructured mesh generator for irregular ROI domains.
 *
 * Responsibilities:
 * - Reserve interfaces for irregular boundaries, holes, and T3 meshes.
 * - Keep third-party triangulation choices isolated from Mesh-DIC solvers.
 *
 * Inputs:
 * - ROI geometry and MeshGenerationConfig.
 *
 * Outputs:
 * - Mesh topology once implemented; currently explicit not-implemented errors.
 *
 * Dependencies:
 * - MeshGenerator base interface.
 *
 * TODO:
 * - Evaluate CGAL, Gmsh, Triangle, OpenCV/Subdiv2D, or in-house Delaunay.
 * - Implement boundary extraction, interior sampling, triangulation, invalid
 *   element removal, and quality improvement without adding dependencies yet.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_UNSTRUCTURED_MESH_GENERATOR_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_UNSTRUCTURED_MESH_GENERATOR_HPP

#include <dic/mesh/generation/mesh_generator.hpp>

namespace dic::mesh {

class UnstructuredMeshGenerator : public MeshGenerator {
public:
    Mesh generate(
        const ROI& roi,
        const MeshGenerationConfig& config
    ) const override;

private:
    void extract_boundary() const;
    void generate_boundary_nodes() const;
    void generate_interior_nodes() const;
    void triangulate() const;
    void remove_invalid_elements() const;
    void improve_mesh_quality() const;
};

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_UNSTRUCTURED_MESH_GENERATOR_HPP
