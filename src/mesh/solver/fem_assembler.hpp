#pragma once

#include "../coordinate/g2l_internal.hpp"
#include "../element/shape_func_internal.hpp"
#include <dic/mesh/mesh_generation_config.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace dic {
class BSplineInterpolator;
}

namespace dic::mesh::internal {

// ============================================================
// StiffnessCache: holds the constant Hessian (assembled from
// reference image gradients) and per-element metadata for
// fast residual assembly during ICGN/Forward-GN iterations.
// ============================================================

struct StiffnessCache {
    Eigen::SparseMatrix<double> A;         // global stiffness (Hessian)
    int fem_size = 0;                       // total DOFs = 2 * n_nodes

    mesh::MeshElementType element_type;
    bool fedic_compatible{false};
    bool element_owned_samples{false};
    std::vector<int> fedic_free_dofs;
    std::vector<std::vector<G2LElementSample>> elem_samples;
    std::vector<std::vector<int>> elem_pixels;   // per-element: pixel indices into G2L
    std::vector<std::vector<int>> elem_dofs;     // per-element: global DOF indices
    std::vector<Eigen::MatrixXd> elem_N_cache;   // per-element: N_g (shape×gradient) rows
    std::vector<Eigen::MatrixXd> elem_DN_cache;  // per-element: DN (derivative) rows
};

// ============================================================
// Assemble constant stiffness matrix from reference image
// ============================================================

StiffnessCache assemble_stiffness(
    const G2LOutput& g2l,
    int img_h, int img_w,
    const double* fx_ref, const double* fy_ref,
    int n_nodes,
    const int* elements, int n_elements,
    mesh::MeshElementType element_type,
    double alpha, double beta = 0.0,
    bool fedic_compatible = false,
    bool element_owned_samples = false);

// ============================================================
// Assemble residual vector (RHS) given current displacement U
// ============================================================

Eigen::VectorXd assemble_residual(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    const Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double beta = 0.0);

// ============================================================
// Compute objective function value (for line search)
// ============================================================

double compute_objective(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    const Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double beta = 0.0);

// ============================================================
// Global ICGN solver (constant Hessian with damped line search)
// Returns number of iterations on success, -1 on failure.
// ============================================================

int global_icgn(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double tol, int max_iter, double beta);

} // namespace dic::mesh::internal
