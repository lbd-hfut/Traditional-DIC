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

namespace dic::mesh {

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
