#pragma once

#include <dic/mesh/mesh_generation_config.hpp>

#include <cstdint>
#include <vector>

namespace dic::mesh::internal {

// ============================================================
// G2L Parameters and Output Structures
// ============================================================

struct G2LParams {
    double tol_global = 0.1;    // convergence tolerance in global coords [px]
    double tol_local  = 1e-8;   // convergence tolerance in natural coords
    double q8_coarse_natural_bound = 1.35; // Q8 bbox candidate prefilter bound
    int    max_iter   = 1000;   // max Newton iterations
};

// A successful element/pixel mapping.  The image-indexed arrays below are
// convenient for the existing path, but cannot retain both owners of a
// shared element edge.  FE-DIC integrates those edge pixels per element.
struct G2LElementSample {
    int element_id = -1; // zero-based
    int pixel_index = -1;
    double xi = 0.0;
    double eta = 0.0;
    double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
};

struct G2LOutput {
    std::vector<double>   xi;       // [n_total] xi
    std::vector<double>   eta;      // [n_total] eta
    std::vector<double>   J11, J12, J21, J22;
    std::vector<uint8_t>  valid;
    std::vector<int>      elem_id;  // 1-based
    std::vector<G2LElementSample> element_samples;
    int img_h = 0;
    int img_w = 0;
};

// ============================================================
// Single-point solvers -- distinct strategies per element type
// ============================================================

// ---- T3: direct analytic solve (linear mapping, closed-form inverse) ----
bool solve_point_t3(
    double gx, double gy,
    const double* elem_nodes,       // [2*3]
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22);

// ---- Q4: simple Newton-Raphson (from center, no fallback) ----
bool solve_point_q4(
    double gx, double gy,
    const double* elem_nodes,       // [2*4]
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    int max_iter);

// ---- Q4: FE-DIC compatible inverse used by the mesh batch path ----
bool solve_point_q4_fedic(
    double gx, double gy,
    const double* elem_nodes,       // [2*4]
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22);

// ---- Q8: iterative Newton-Raphson (requires initial seed from Q4) ----
bool solve_point_q8(
    double gx, double gy,
    const double* elem_nodes,       // [2*8]
    double xi0, double eta0,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    double tol, int max_iter);

// ---- Q8: robust fallback (normalized coords + line search) ----
bool solve_point_q8_fallback(
    double gx, double gy,
    const double* elem_nodes,       // [2*8]
    double xi0, double eta0,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    double tol_global, double tol_local, int max_iter);

// ============================================================
// Batch G2L mapping
// ============================================================

G2LOutput compute_global_to_local(
    const double* inform,           // [n_pixels * 3] (gx, gy, elem_id) 1-based
    int n_pixels,
    const double* nodes_coord,      // [2 * n_nodes]
    int n_nodes,
    const int*    elements,          // [n_elements * stride]
    int n_elements,
    int img_h, int img_w,
    MeshElementType element_type,
    const G2LParams& params = G2LParams{});

} // namespace dic::mesh::internal
