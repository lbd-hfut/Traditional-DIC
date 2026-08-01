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

#include <dic/interpolation/bspline.hpp>

#include <limits>

namespace dic {

// Kept mesh-local so subset initialization keeps its existing API and behavior.
enum class MeshNodalInitializationMethod {
    FEDICFFT
};

enum class MeshOptimizationMethod {
    FEDICElementICGN,
    FEDICElementFGN
};

struct MeshConfig {
    int max_iterations{30};
    double convergence_threshold{1e-3};
    int search_radius{20};
    double regularization_alpha{0.0};
    bool mirror_image_padding{false};
    BSplinePrecomputeConfig image_precompute{};
    MeshOptimizationMethod optimization_method{MeshOptimizationMethod::FEDICElementICGN};
    MeshNodalInitializationMethod nodal_initialization_method{
        MeshNodalInitializationMethod::FEDICFFT};

    struct FEDICFFTInitializationConfig {
        int window_size{75};
        int search_radius{30};
        bool mirror_boundary_fallback{true};
    } fedic_fft_initialization{};

    struct InitializationQualityControlConfig {
        bool enabled{false};
        double min_zncc{-1.0};
        double max_znssd{std::numeric_limits<double>::infinity()};
        bool fedic_qfactor_enabled{false};
        double fedic_qfactor_std_factor{1.25};
        double neighbor_mad_factor{4.0};
        double max_neighbor_deviation{0.0};
        int interpolation_neighbors{8};
    } initialization_quality{};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_MESH_CONFIG_HPP
