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

enum class MeshPhotometricObjective {
    SSD,
    ElementAffineZNSSD
};

struct MeshConfig {
    int max_iterations{30};
    double convergence_threshold{1e-3};
    int search_radius{20};
    double regularization_alpha{0.0};
    bool mirror_image_padding{false};

    // When true (and a ROI mask is supplied to MeshDIC::compute), nodes whose
    // FFT correlation window would span outside the ROI skip the FFT lock and
    // are marked invalid, so fill_missing_nodal_initialization interpolates
    // their seed from interior nodes. The plain FFT initializer only checks the
    // image boundary (not the ROI), so a boundary-crossing window matches
    // against background content and locks onto a wrong peak (disp case: right
    // ROI boundary underestimated u by ~20 px). Skipping those nodes and
    // inheriting the smooth interior displacement fixes the boundary band
    // without a mask-aware FFT.
    bool boundary_interpolation_init{true};
    // When a boundary node has a valid SIFT/pyramid prior offset, seed it
    // directly with that offset instead of running the FFT lock. The prior is a
    // global match unaffected by the ROI crop, whereas the FFT window spanning
    // outside the ROI contains no valid content and re-locks to a shifted peak
    // (disp right edge: FFT after a good SIFT offset still lands 5-10 px off,
    // while the raw SIFT offset is ~1 px). Only consulted when
    // boundary_interpolation_init is also true; interior nodes are unaffected.
    //
    // Default OFF (2026-08-08): on the multiview ComplexCylinder case this
    // regresses dist-to-GT by +0.3 mm on T3/Q4 because those nodes lose the
    // FFT refinement inside the displacement field. It is a clear win on the
    // mono disp case (~226 px single-direction displacement, right ROI edge
    // >5px band 32% -> 0.1%), so disp enables it explicitly via
    // initialization.boundary_direct_prior_seed: true.
    bool boundary_direct_prior_seed{false};
    BSplinePrecomputeConfig image_precompute{};
    MeshOptimizationMethod optimization_method{MeshOptimizationMethod::FEDICElementICGN};
    MeshPhotometricObjective photometric_objective{MeshPhotometricObjective::SSD};
    MeshNodalInitializationMethod nodal_initialization_method{
        MeshNodalInitializationMethod::FEDICFFT};

    struct FEDICFFTInitializationConfig {
        int window_size{75};
        int search_radius{30};
        bool mirror_boundary_fallback{true};
    } fedic_fft_initialization{};

    // True coarse-to-fine image-pyramid initialization. Coarse levels shrink
    // large disparities into small pixel shifts so a small radius covers the
    // whole disparity range; per-node coarse displacements then seed the FFT
    // search center instead of a blind (0,0) search.
    struct PyramidInitializationConfig {
        bool enabled{false};              // default off; enable per case
        int num_levels{4};                // truncated adaptively to image size
        double scale_factor{0.5};         // downsample ratio per level
        int coarse_search_radius{32};     // search radius at the coarsest level (must
                                          // cover max_disp * scale_factor^(num_levels-1))
        int refinement_radius{4};         // per-level refinement radius
        int window_size{75};              // FFT correlation window at full res
    } pyramid_initialization{};

    // SIFT feature-prior initialization. SIFT matches are rotation/scale
    // invariant, so a per-node IDW-interpolated SIFT displacement gives a far
    // more reliable FFT search center than the rigid pyramid NCC on
    // perspective/distorted cases. When enabled, the SIFT seed takes priority
    // over the pyramid seed as the FFT search center.
    struct SiftPriorInitializationConfig {
        bool enabled{false};               // default off; enable per case
        int max_features{4000};            // FeatureMatcher max SIFT features
        double ratio_threshold{0.75};      // SIFT Lowe ratio test
        double robust_mad_factor{5.0};     // MAD gate for robust inliers
        int interpolation_neighbors{8};    // SIFTInitializer IDW neighbors
        double interpolation_radius{180.0}; // IDW search radius (px)
    } sift_prior_initialization{};

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
