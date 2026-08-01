#include "fem_assembler.hpp"
#include <dic/interpolation/bspline.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace dic::mesh::internal {

// ============================================================
// Helpers
// ============================================================

static bool jacobian_is_usable(double J11, double J12, double J21, double J22) {
    double detJ = J11 * J22 - J12 * J21;
    if (std::abs(detJ) < 1.0) return false;
    double iJ11 = J22 / detJ, iJ12 = -J12 / detJ;
    double iJ21 = -J21 / detJ, iJ22 = J11 / detJ;
    double inv_norm = std::sqrt(iJ11 * iJ11 + iJ12 * iJ12 +
                                iJ21 * iJ21 + iJ22 * iJ22);
    return std::isfinite(inv_norm) && inv_norm <= 10.0;
}

static bool in_bounds(mesh::MeshElementType type, double xi, double eta) {
    switch (type) {
        case mesh::MeshElementType::T3:
            return xi >= -0.05 && eta >= -0.05 && (xi + eta) <= 1.05;
        case mesh::MeshElementType::Q4:
            return std::abs(xi) <= 1.5 && std::abs(eta) <= 1.5;
        case mesh::MeshElementType::Q8:
            return std::abs(xi) <= 1.05 && std::abs(eta) <= 1.05;
    }
    return false;
}

// Compute N_g (shape × image-gradient coupling) and DN (derivative matrix)
// for a single pixel.
// N_g = [N0*fx, N0*fy, N1*fx, N1*fy, ..., N_{nn-1}*fx, N_{nn-1}*fy]  (dof × 1)
// DN  = J^{-1} * DN_loc  (4 × dof), where DN_loc contains dN/dxi and dN/deta
static void compute_N_DN(mesh::MeshElementType type, double xi, double eta,
                         double invJ11, double invJ12,
                         double invJ21, double invJ22,
                         double fx, double fy,
                         double* N_g, double* DN_out) {
    int nn = nodes_per_element(type);
    int dof = 2 * nn;

    // Shape functions and derivatives
    std::vector<double> N(nn), dN_dxi(nn), dN_deta(nn);
    shape_functions(type, xi, eta, N.data(), dN_dxi.data(), dN_deta.data());

    // N_g
    for (int k = 0; k < nn; ++k) {
        N_g[2 * k]     = N[k] * fx;
        N_g[2 * k + 1] = N[k] * fy;
    }

    // DN in local coords (4 × dof), stored row-major
    std::vector<double> DN_loc(4 * dof, 0.0);
    for (int k = 0; k < nn; ++k) {
        int u = 2 * k, v = 2 * k + 1;
        DN_loc[u]           = dN_dxi[k];   // row 0 (d/dxi for u)
        DN_loc[dof + u]     = dN_deta[k];  // row 1 (d/deta for u)
        DN_loc[2*dof + v]   = dN_dxi[k];   // row 2 (d/dxi for v)
        DN_loc[3*dof + v]   = dN_deta[k];  // row 3 (d/deta for v)
    }

    // DN = J^{-1} * DN_loc
    for (int c = 0; c < dof; ++c) {
        DN_out[c]           = invJ11 * DN_loc[c]           + invJ21 * DN_loc[dof + c];
        DN_out[dof + c]     = invJ12 * DN_loc[c]           + invJ22 * DN_loc[dof + c];
        DN_out[2*dof + c]   = invJ11 * DN_loc[2*dof + c]   + invJ21 * DN_loc[3*dof + c];
        DN_out[3*dof + c]   = invJ12 * DN_loc[2*dof + c]   + invJ22 * DN_loc[3*dof + c];
    }
}

// Get shape function values only (no derivatives needed)
static void shape_values_only(mesh::MeshElementType type, double xi, double eta,
                              std::vector<double>& N) {
    int nn = nodes_per_element(type);
    N.resize(nn);
    std::vector<double> dxi(nn), deta(nn);
    shape_functions(type, xi, eta, N.data(), dxi.data(), deta.data());
}

// Batch-interpolate deformed image at warped positions
static void interpolate_deformed_batch(
    const BSplineInterpolator* interp,
    const double* wx, const double* wy, int n,
    double* g_vals)
{
    if (interp) {
        for (int i = 0; i < n; ++i) {
            g_vals[i] = interp->value(wx[i], wy[i]);
        }
    } else {
        std::fill(g_vals, g_vals + n, 0.0);
    }
}

// ============================================================
// assemble_stiffness
// ============================================================

StiffnessCache assemble_stiffness(
    const G2LOutput& g2l,
    int img_h, int img_w,
    const double* fx_ref, const double* fy_ref,
    int n_nodes,
    const int* elements, int n_elements,
    mesh::MeshElementType element_type,
    double alpha, double beta,
    bool fedic_compatible,
    bool element_owned_samples)
{
    StiffnessCache cache;
    int nn = nodes_per_element(element_type);
    int dof = 2 * nn;
    int fem_size = 2 * n_nodes;
    int elem_stride = (element_type == mesh::MeshElementType::Q8) ? 9 : nn;
    int dn_size = 4 * dof;

    cache.fem_size = fem_size;
    cache.element_type = element_type;
    cache.fedic_compatible = fedic_compatible;
    cache.element_owned_samples = element_owned_samples;
    cache.elem_pixels.resize(n_elements);
    cache.elem_samples.resize(n_elements);
    cache.elem_dofs.resize(n_elements);
    cache.elem_N_cache.resize(n_elements);
    cache.elem_DN_cache.resize(n_elements);

    // Build per-element DOF lists
    for (int e = 0; e < n_elements; ++e) {
        std::vector<int> dofs(dof);
        for (int k = 0; k < nn; ++k) {
            int nid = elements[e * elem_stride + k] - 1;
            dofs[2 * k]     = 2 * nid;
            dofs[2 * k + 1] = 2 * nid + 1;
        }
        cache.elem_dofs[e] = dofs;
        if (fedic_compatible || element_owned_samples) {
            cache.fedic_free_dofs.insert(cache.fedic_free_dofs.end(),
                                         dofs.begin(), dofs.end());
        }
    }
    if (fedic_compatible || element_owned_samples) {
        std::sort(cache.fedic_free_dofs.begin(), cache.fedic_free_dofs.end());
        cache.fedic_free_dofs.erase(
            std::unique(cache.fedic_free_dofs.begin(), cache.fedic_free_dofs.end()),
            cache.fedic_free_dofs.end());
    }

    // Map pixels to elements
    int total = img_h * img_w;
    std::vector<std::vector<int>> elem_pix_map(n_elements);
    if (fedic_compatible || element_owned_samples) {
        for (const auto& sample : g2l.element_samples) {
            if (sample.element_id < 0 || sample.element_id >= n_elements) continue;
            if (sample.pixel_index < 0 || sample.pixel_index >= total) continue;
            if (!fedic_compatible &&
                (!in_bounds(element_type, sample.xi, sample.eta) ||
                 !jacobian_is_usable(sample.J11, sample.J12, sample.J21, sample.J22))) {
                continue;
            }
            elem_pix_map[sample.element_id].push_back(sample.pixel_index);
            cache.elem_samples[sample.element_id].push_back(sample);
        }
    } else {
        for (int idx = 0; idx < total; ++idx) {
            if (!g2l.valid[idx]) continue;
            int eid = g2l.elem_id[idx] - 1;
            if (eid < 0 || eid >= n_elements) continue;
            if (!in_bounds(element_type, g2l.xi[idx], g2l.eta[idx])) continue;
            if (!jacobian_is_usable(g2l.J11[idx], g2l.J12[idx],
                                    g2l.J21[idx], g2l.J22[idx])) continue;
            elem_pix_map[eid].push_back(idx);
            cache.elem_samples[eid].push_back({eid, idx, g2l.xi[idx], g2l.eta[idx],
                                                g2l.J11[idx], g2l.J12[idx],
                                                g2l.J21[idx], g2l.J22[idx]});
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n_elements * dof * dof);

    std::vector<double> N_buf(dof), DN_buf(dn_size);

    for (int e = 0; e < n_elements; ++e) {
        const auto& pix_idx = elem_pix_map[e];
        int n_pix = static_cast<int>(pix_idx.size());
        cache.elem_pixels[e] = pix_idx;
        if (n_pix == 0) continue;

        const auto& dofs = cache.elem_dofs[e];
        Eigen::MatrixXd N_cache(n_pix, dof);
        Eigen::MatrixXd DN_cache(n_pix, dn_size);
        Eigen::MatrixXd A_e = Eigen::MatrixXd::Zero(dof, dof);

        for (int pi = 0; pi < n_pix; ++pi) {
            const auto& sample = cache.elem_samples[e][pi];
            int idx = sample.pixel_index;
            int py = idx / img_w, px = idx % img_w;

            double xi = sample.xi, eta = sample.eta;
            double J11 = sample.J11, J12 = sample.J12;
            double J21 = sample.J21, J22 = sample.J22;
            double detJ = J11 * J22 - J12 * J21;
            if (std::abs(detJ) < 1e-12) continue;
            double iJ11 = J22 / detJ, iJ12 = -J12 / detJ;
            double iJ21 = -J21 / detJ, iJ22 = J11 / detJ;

            double fx = fx_ref[py * img_w + px];
            double fy = fy_ref[py * img_w + px];

            compute_N_DN(element_type, xi, eta, iJ11, iJ12, iJ21, iJ22,
                         fx, fy, N_buf.data(), DN_buf.data());

            // g = [N0*fx, N0*fy, N1*fx, N1*fy, ...]^T
            Eigen::Map<Eigen::VectorXd> g(N_buf.data(), dof);
            A_e += g * g.transpose();

            // Regularization: alpha * DN^T * DN
            Eigen::Map<Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::RowMajor>>
                DN_mat(DN_buf.data(), 4, dof);
            A_e += alpha * DN_mat.transpose() * DN_mat;

            for (int j = 0; j < dof; ++j) N_cache(pi, j) = g(j);
            for (int j = 0; j < dn_size; ++j) DN_cache(pi, j) = DN_buf[j];
        }

        cache.elem_N_cache[e] = N_cache;
        cache.elem_DN_cache[e] = DN_cache;

        for (int i = 0; i < dof; ++i)
            for (int j = 0; j < dof; ++j)
                if (std::abs(A_e(i, j)) > 1e-15)
                    triplets.push_back({dofs[i], dofs[j], A_e(i, j)});
    }

    // Soft boundary condition: displacement penalty
    if (beta > 0.0) {
        for (int i = 0; i < fem_size; ++i)
            triplets.push_back({i, i, beta});
    }

    cache.A.resize(fem_size, fem_size);
    cache.A.setFromTriplets(triplets.begin(), triplets.end());
    cache.A.makeCompressed();

    return cache;
}

// ============================================================
// assemble_residual
// ============================================================

Eigen::VectorXd assemble_residual(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    const Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double beta)
{
    int fem_size = cache.fem_size;
    mesh::MeshElementType type = cache.element_type;
    int nn = nodes_per_element(type);
    int dof = 2 * nn;
    int dn_size = 4 * dof;

    Eigen::VectorXd b = Eigen::VectorXd::Zero(fem_size);

    for (int e = 0; e < n_elements; ++e) {
        const auto& pix_idx = cache.elem_pixels[e];
        int n_pix = static_cast<int>(pix_idx.size());
        if (n_pix == 0) continue;

        const auto& dofs = cache.elem_dofs[e];

        // Extract element displacement
        Eigen::VectorXd U_e(dof);
        for (int k = 0; k < nn; ++k) {
            U_e(2 * k)     = U(dofs[2 * k]);
            U_e(2 * k + 1) = U(dofs[2 * k + 1]);
        }

        Eigen::VectorXd b_e = Eigen::VectorXd::Zero(dof);

        // Warp pixel positions
        std::vector<double> warp_x(n_pix), warp_y(n_pix);
        std::vector<double> N_loc;
        for (int pi = 0; pi < n_pix; ++pi) {
            const auto& sample = cache.elem_samples[e][pi];
            int idx = sample.pixel_index;
            int py = idx / img_w, px = idx % img_w;

            shape_values_only(type, sample.xi, sample.eta, N_loc);

            double u_ip = 0.0, v_ip = 0.0;
            for (int k = 0; k < nn; ++k) {
                u_ip += N_loc[k] * U_e(2 * k);
                v_ip += N_loc[k] * U_e(2 * k + 1);
            }
            warp_x[pi] = cache.fedic_compatible ? px + u_ip
                                                 : std::max(0.0, std::min(static_cast<double>(img_w - 1), px + u_ip));
            warp_y[pi] = cache.fedic_compatible ? py + v_ip
                                                 : std::max(0.0, std::min(static_cast<double>(img_h - 1), py + v_ip));
        }

        // Interpolate deformed image at warped positions
        std::vector<double> g_vals(n_pix);
        interpolate_deformed_batch(def_interp, warp_x.data(), warp_y.data(),
                                   n_pix, g_vals.data());

        // Accumulate residual
        for (int pi = 0; pi < n_pix; ++pi) {
            int idx = cache.elem_samples[e][pi].pixel_index;
            int py = idx / img_w, px = idx % img_w;
            double r_img = ref_img[py * img_w + px] - g_vals[pi];

            // g_row^T * r
            Eigen::Map<const Eigen::Matrix<double, 1, Eigen::Dynamic>> g_row(
                cache.elem_N_cache[e].row(pi).data(), dof);
            b_e += r_img * g_row.transpose();

            // -alpha * DN^T * DN * U_e
            Eigen::Map<const Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::RowMajor>>
                DN_mat(cache.elem_DN_cache[e].row(pi).data(), 4, dof);
            b_e -= alpha * DN_mat.transpose() * DN_mat * U_e;
        }

        // Displacement penalty
        if (beta > 0.0) {
            for (int k = 0; k < dof; ++k)
                b_e(k) -= beta * U_e(k);
        }

        for (int i = 0; i < dof; ++i)
            b(dofs[i]) += b_e(i);
    }

    return b;
}

// ============================================================
// compute_objective
// ============================================================

double compute_objective(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    const Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double beta)
{
    mesh::MeshElementType type = cache.element_type;
    int nn = nodes_per_element(type);
    int dof = 2 * nn;

    double energy = 0.0;

    for (int e = 0; e < n_elements; ++e) {
        const auto& pix_idx = cache.elem_pixels[e];
        int n_pix = static_cast<int>(pix_idx.size());
        if (n_pix == 0) continue;

        const auto& dofs = cache.elem_dofs[e];

        Eigen::VectorXd U_e(dof);
        for (int k = 0; k < nn; ++k) {
            U_e(2 * k)     = U(dofs[2 * k]);
            U_e(2 * k + 1) = U(dofs[2 * k + 1]);
        }

        std::vector<double> warp_x(n_pix), warp_y(n_pix);
        std::vector<double> N_loc;
        for (int pi = 0; pi < n_pix; ++pi) {
            const auto& sample = cache.elem_samples[e][pi];
            int idx = sample.pixel_index;
            int py = idx / img_w, px = idx % img_w;

            shape_values_only(type, sample.xi, sample.eta, N_loc);

            double u_ip = 0.0, v_ip = 0.0;
            for (int k = 0; k < nn; ++k) {
                u_ip += N_loc[k] * U_e(2 * k);
                v_ip += N_loc[k] * U_e(2 * k + 1);
            }
            warp_x[pi] = cache.fedic_compatible ? px + u_ip
                                                 : std::max(0.0, std::min(static_cast<double>(img_w - 1), px + u_ip));
            warp_y[pi] = cache.fedic_compatible ? py + v_ip
                                                 : std::max(0.0, std::min(static_cast<double>(img_h - 1), py + v_ip));
        }

        std::vector<double> g_vals(n_pix);
        interpolate_deformed_batch(def_interp, warp_x.data(), warp_y.data(),
                                   n_pix, g_vals.data());

        for (int pi = 0; pi < n_pix; ++pi) {
            int idx = cache.elem_samples[e][pi].pixel_index;
            int py = idx / img_w, px = idx % img_w;
            double r = ref_img[py * img_w + px] - g_vals[pi];
            energy += 0.5 * r * r;

            Eigen::Map<const Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::RowMajor>>
                DN_mat(cache.elem_DN_cache[e].row(pi).data(), 4, dof);
            Eigen::Vector4d grad_u = DN_mat * U_e;
            energy += 0.5 * alpha * grad_u.squaredNorm();
        }
    }

    if (beta > 0.0) {
        energy += 0.5 * beta * U.squaredNorm();
    }

    return energy;
}

// ============================================================
// global_icgn (constant Hessian with damped line search)
// ============================================================
int global_icgn(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double tol, int max_iter, double beta)
{
    const std::vector<int>& free_nodes = cache.fedic_free_dofs;
    int n_free = static_cast<int>(free_nodes.size());
    if (n_free == 0) return -1;

    // Build free-DOF subset of the constant Hessian
    std::vector<Eigen::Triplet<double>> ft;
    ft.reserve(cache.A.nonZeros());
    for (int k = 0; k < cache.A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(cache.A, k); it; ++it) {
            auto ir = std::lower_bound(free_nodes.begin(), free_nodes.end(),
                                       static_cast<int>(it.row()));
            auto ic = std::lower_bound(free_nodes.begin(), free_nodes.end(),
                                       static_cast<int>(it.col()));
            if (ir != free_nodes.end() && *ir == static_cast<int>(it.row()) &&
                ic != free_nodes.end() && *ic == static_cast<int>(it.col())) {
                ft.push_back({static_cast<int>(ir - free_nodes.begin()),
                              static_cast<int>(ic - free_nodes.begin()), it.value()});
            }
        }
    }

    Eigen::SparseMatrix<double> A_free(n_free, n_free);
    A_free.setFromTriplets(ft.begin(), ft.end());
    A_free.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_free);
    if (solver.info() != Eigen::Success) return -1;

    bool has_initial = U.squaredNorm() > 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd b_vec = assemble_residual(
            cache, g2l, ref_img, img_h, img_w,
            elements, n_elements, U, def_interp, alpha, beta);

        Eigen::VectorXd b_free(n_free);
        for (int i = 0; i < n_free; ++i) b_free(i) = b_vec(free_nodes[i]);

        Eigen::VectorXd dU_free = solver.solve(b_free);
        if (solver.info() != Eigen::Success) return -1;

        const double norm_dof = cache.element_type == mesh::MeshElementType::T3
            ? static_cast<double>(cache.fem_size)
            : static_cast<double>(n_free);
        double raw_normW = dU_free.norm() / std::sqrt(norm_dof);
        if (!std::isfinite(raw_normW)) return -1;
        if (raw_normW > 0.1 / tol) {
            return (iter == 0 && !has_initial) ? -1 : iter;
        }

        double step = 1.0;
        constexpr double max_normW = 0.1;
        if (raw_normW > max_normW) step = max_normW / raw_normW;

        double obj0 = compute_objective(
            cache, g2l, ref_img, img_h, img_w,
            elements, n_elements, U, def_interp, alpha, beta);
        if (!std::isfinite(obj0)) return -1;

        Eigen::VectorXd U_trial = U;
        bool accepted = false;
        for (int bt = 0; bt < 12; ++bt) {
            U_trial = U;
            for (int i = 0; i < n_free; ++i)
                U_trial(free_nodes[i]) += step * dU_free(i);

            double obj1 = compute_objective(
                cache, g2l, ref_img, img_h, img_w,
                elements, n_elements, U_trial, def_interp, alpha, beta);
            if (std::isfinite(obj1) && obj1 <= obj0) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if (!accepted) {
            return (iter == 0 && !has_initial) ? -1 : iter;
        }

        U = U_trial;

        double normW = step * raw_normW;
        if (normW < tol) return iter + 1;
    }

    return max_iter;
}

int global_fgn(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha, double tol, int max_iter, double beta)
{
    (void)g2l;
    const std::vector<int>& free_dofs = cache.fedic_free_dofs;
    const int n_free = static_cast<int>(free_dofs.size());
    if (n_free == 0 || def_interp == nullptr) return -1;

    const mesh::MeshElementType type = cache.element_type;
    const int nn = nodes_per_element(type);
    const int dof = 2 * nn;
    const bool has_initial = U.squaredNorm() > 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<Eigen::Triplet<double>> triplets;
        Eigen::VectorXd b = Eigen::VectorXd::Zero(cache.fem_size);
        triplets.reserve(cache.A.nonZeros());

        for (int e = 0; e < n_elements; ++e) {
            const int n_pix = static_cast<int>(cache.elem_samples[e].size());
            if (n_pix == 0) continue;

            const auto& dofs = cache.elem_dofs[e];
            Eigen::VectorXd U_e(dof);
            for (int k = 0; k < nn; ++k) {
                U_e(2 * k) = U(dofs[2 * k]);
                U_e(2 * k + 1) = U(dofs[2 * k + 1]);
            }

            Eigen::MatrixXd A_e = Eigen::MatrixXd::Zero(dof, dof);
            Eigen::VectorXd b_e = Eigen::VectorXd::Zero(dof);
            std::vector<double> N_loc;
            for (int pi = 0; pi < n_pix; ++pi) {
                const auto& sample = cache.elem_samples[e][pi];
                const int idx = sample.pixel_index;
                const int py = idx / img_w;
                const int px = idx % img_w;

                shape_values_only(type, sample.xi, sample.eta, N_loc);
                double u_ip = 0.0;
                double v_ip = 0.0;
                for (int k = 0; k < nn; ++k) {
                    u_ip += N_loc[k] * U_e(2 * k);
                    v_ip += N_loc[k] * U_e(2 * k + 1);
                }
                const double wx = cache.fedic_compatible ? px + u_ip
                    : std::max(0.0, std::min(static_cast<double>(img_w - 1), px + u_ip));
                const double wy = cache.fedic_compatible ? py + v_ip
                    : std::max(0.0, std::min(static_cast<double>(img_h - 1), py + v_ip));

                const double r = ref_img[idx] - def_interp->value(wx, wy);
                const Eigen::Vector2d grad = def_interp->gradient(wx, wy);
                if (!std::isfinite(r) || !grad.allFinite()) continue;

                Eigen::VectorXd jac(dof);
                for (int k = 0; k < nn; ++k) {
                    jac(2 * k) = N_loc[k] * grad.x();
                    jac(2 * k + 1) = N_loc[k] * grad.y();
                }
                A_e.noalias() += jac * jac.transpose();
                b_e.noalias() += r * jac;

                Eigen::Map<const Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::RowMajor>>
                    DN_mat(cache.elem_DN_cache[e].row(pi).data(), 4, dof);
                A_e.noalias() += alpha * DN_mat.transpose() * DN_mat;
                b_e.noalias() -= alpha * DN_mat.transpose() * DN_mat * U_e;
            }

            if (beta > 0.0) {
                A_e.diagonal().array() += beta;
                b_e.noalias() -= beta * U_e;
            }
            for (int i = 0; i < dof; ++i) {
                b(dofs[i]) += b_e(i);
                for (int j = 0; j < dof; ++j) {
                    if (std::abs(A_e(i, j)) > 1e-15)
                        triplets.push_back({dofs[i], dofs[j], A_e(i, j)});
                }
            }
        }

        Eigen::SparseMatrix<double> A(cache.fem_size, cache.fem_size);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();

        std::vector<Eigen::Triplet<double>> free_triplets;
        free_triplets.reserve(A.nonZeros());
        for (int k = 0; k < A.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                const auto ir = std::lower_bound(free_dofs.begin(), free_dofs.end(), static_cast<int>(it.row()));
                const auto ic = std::lower_bound(free_dofs.begin(), free_dofs.end(), static_cast<int>(it.col()));
                if (ir != free_dofs.end() && *ir == it.row() &&
                    ic != free_dofs.end() && *ic == it.col()) {
                    free_triplets.push_back({static_cast<int>(ir - free_dofs.begin()),
                                             static_cast<int>(ic - free_dofs.begin()), it.value()});
                }
            }
        }
        Eigen::SparseMatrix<double> A_free(n_free, n_free);
        A_free.setFromTriplets(free_triplets.begin(), free_triplets.end());
        A_free.makeCompressed();
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A_free);
        if (solver.info() != Eigen::Success) return -1;

        Eigen::VectorXd b_free(n_free);
        for (int i = 0; i < n_free; ++i) b_free(i) = b(free_dofs[i]);
        const Eigen::VectorXd dU_free = solver.solve(b_free);
        if (solver.info() != Eigen::Success) return -1;

        const double norm_dof = type == mesh::MeshElementType::T3
            ? static_cast<double>(cache.fem_size) : static_cast<double>(n_free);
        const double raw_normW = dU_free.norm() / std::sqrt(norm_dof);
        if (!std::isfinite(raw_normW)) return -1;
        if (raw_normW > 0.1 / tol) return (iter == 0 && !has_initial) ? -1 : iter;

        double step = raw_normW > 0.1 ? 0.1 / raw_normW : 1.0;
        const double obj0 = compute_objective(cache, g2l, ref_img, img_h, img_w,
                                              elements, n_elements, U, def_interp, alpha, beta);
        if (!std::isfinite(obj0)) return -1;
        Eigen::VectorXd U_trial = U;
        bool accepted = false;
        for (int bt = 0; bt < 12; ++bt) {
            U_trial = U;
            for (int i = 0; i < n_free; ++i) U_trial(free_dofs[i]) += step * dU_free(i);
            const double obj1 = compute_objective(cache, g2l, ref_img, img_h, img_w,
                                                  elements, n_elements, U_trial, def_interp, alpha, beta);
            if (std::isfinite(obj1) && obj1 <= obj0) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) return (iter == 0 && !has_initial) ? -1 : iter;
        U = U_trial;
        if (step * raw_normW < tol) return iter + 1;
    }
    return max_iter;
}

} // namespace dic::mesh::internal
