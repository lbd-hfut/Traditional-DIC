/**
 * @file unstructured_mesh_generator.cpp
 * @brief Explicit placeholder for future unstructured ROI mesh generation.
 *
 * Responsibilities:
 * - Reserve implementation stages for irregular ROI and T3 mesh generation.
 * - Avoid introducing unselected third-party triangulation dependencies.
 *
 * Inputs:
 * - ROI geometry and mesh generation configuration.
 *
 * Outputs:
 * - Not implemented yet; future output will be Mesh topology.
 *
 * Dependencies:
 * - MeshGenerator interface.
 *
 * TODO:
 * - Extract boundary -> generate boundary nodes -> generate interior nodes ->
 *   triangulate -> remove invalid elements -> improve mesh quality.
 * - Evaluate CGAL, Gmsh, Triangle, OpenCV/Subdiv2D, and in-house Delaunay.
 */

#include <dic/mesh/generation/unstructured_mesh_generator.hpp>

#include <stdexcept>

namespace dic::mesh {

Mesh UnstructuredMeshGenerator::generate(
    const ROI& roi,
    const MeshGenerationConfig& config
) const
{
    (void)roi;
    (void)config;
    throw std::runtime_error("Unstructured mesh generation is not implemented yet.");
}

void UnstructuredMeshGenerator::extract_boundary() const {}
void UnstructuredMeshGenerator::generate_boundary_nodes() const {}
void UnstructuredMeshGenerator::generate_interior_nodes() const {}
void UnstructuredMeshGenerator::triangulate() const {}
void UnstructuredMeshGenerator::remove_invalid_elements() const {}
void UnstructuredMeshGenerator::improve_mesh_quality() const {}

} // namespace dic::mesh
