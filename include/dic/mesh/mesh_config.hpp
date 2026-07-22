/**
 * @file mesh_config.hpp
 * @brief Configuration for 2D Mesh-DIC.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_CONFIG_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_CONFIG_HPP



namespace dic {

enum class MeshSolverMethod {
    GlobalICGN,
    ForwardGaussNewton
};

struct MeshConfig {
    MeshSolverMethod solver_method{MeshSolverMethod::GlobalICGN};
    int max_iterations{30};
    double convergence_threshold{1e-3};
    int search_radius{20};
    double regularization_alpha{0.0};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_CONFIG_HPP
