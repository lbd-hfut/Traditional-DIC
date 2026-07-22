/**
 * @file mesh_dic.cpp
 * @brief Minimal implementation placeholder for Mesh-DIC controller.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/mesh/generation/roi_mesh_generator.hpp>
#include <dic/mesh/mesh_dic.hpp>
#include <dic/mesh/solver/global_gauss_newton.hpp>
#include <dic/mesh/solver/global_icgn.hpp>

namespace dic {

MeshDIC::MeshDIC(MeshConfig config) : config_(config) {}

std::vector<Displacement2D> MeshDIC::compute(
    const Image& reference,
    const Image& deformed,
    Mesh mesh
) const
{
    // TODO: Run Mesh displacement initialization before the nonlinear solver.
    if (config_.solver_method == MeshSolverMethod::ForwardGaussNewton) {
        GlobalGaussNewton solver;
        (void)solver.solve(reference, deformed, mesh);
    } else {
        GlobalICGN solver(config_);
        (void)solver.solve(reference, deformed, mesh);
    }

    // TODO: Convert solved nodal displacements into Displacement2D results.
    return {};
}

std::vector<Displacement2D> MeshDIC::compute(
    const Image& reference,
    const Image& deformed,
    const ROI& roi,
    const mesh::MeshGenerationConfig& mesh_generation_config
) const
{
    mesh::ROIMeshGenerator generator;
    auto generated_mesh = generator.generate(roi, mesh_generation_config);
    return compute(reference, deformed, generated_mesh);
}

} // namespace dic
