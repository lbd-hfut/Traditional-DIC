#include "g2l_internal.hpp"
#include "../element/shape_func_internal.hpp"

#include <algorithm>
#include <cmath>

namespace dic::mesh::internal {

// ============================================================

//



// ============================================================
bool solve_point_t3(
    double gx, double gy,
    const double* elem_nodes,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22)
{
    
    
    double dx_dxi  = elem_nodes[2] - elem_nodes[0];
    double dx_deta = elem_nodes[4] - elem_nodes[0];
    double dy_dxi  = elem_nodes[3] - elem_nodes[1];
    double dy_deta = elem_nodes[5] - elem_nodes[1];

    double det = dx_dxi * dy_deta - dx_deta * dy_dxi;
    if (std::abs(det) < 1e-12) return false;

    double inv_det = 1.0 / det;
    double dx = gx - elem_nodes[0];
    double dy = gy - elem_nodes[1];

    
    xi  = inv_det * ( dy_deta * dx - dx_deta * dy);
    eta = inv_det * (-dy_dxi  * dx + dx_dxi  * dy);

    
    if (xi < -0.05 || eta < -0.05 || xi + eta > 1.05) return false;

    
    J11 = dx_dxi;  J12 = dx_deta;
    J21 = dy_dxi;  J22 = dy_deta;
    return true;
}

// ============================================================

//



// ============================================================
// Q4: analytic quadratic inverse
//
// Bilinear mapping:  x = a0 + a1*xi + a2*eta + a3*xi*eta
//                    y = b0 + b1*xi + b2*eta + b3*xi*eta
//
// Eliminate eta -> quadratic in xi:  A*xi^2 + B*xi + C = 0
// Eliminate xi  -> quadratic in eta: A'*eta^2 + B'*eta + C' = 0
//
// Pick the better-conditioned path, verify with round-trip check.
// Newton-Raphson fallback for degenerate (near-rectangular, A~0) cases.
// ============================================================
bool solve_point_q4(
    double gx, double gy,
    const double* elem_nodes,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    int max_iter)
{
    // ---- 1. Compute bilinear coefficients ----
    double x0 = elem_nodes[0], y0 = elem_nodes[1];
    double x1 = elem_nodes[2], y1 = elem_nodes[3];
    double x2 = elem_nodes[4], y2 = elem_nodes[5];
    double x3 = elem_nodes[6], y3 = elem_nodes[7];

    double a0 = 0.25 * (x0 + x1 + x2 + x3);
    double a1 = 0.25 * (-x0 + x1 + x2 - x3);
    double a2 = 0.25 * (-x0 - x1 + x2 + x3);
    double a3 = 0.25 * (x0 - x1 + x2 - x3);

    double b0 = 0.25 * (y0 + y1 + y2 + y3);
    double b1 = 0.25 * (-y0 + y1 + y2 - y3);
    double b2 = 0.25 * (-y0 - y1 + y2 + y3);
    double b3 = 0.25 * (y0 - y1 + y2 - y3);

    double dx = gx - a0;
    double dy = gy - b0;

    // ---- 2. Choose primary unknown: xi or eta ----
    // At center, denominator for eta is a2 (since a3*0=0),
    // denominator for xi  is a1. Pick the larger one.
    bool solve_xi_first = (std::abs(a1) >= std::abs(a2));

    // ---- 3. Solve quadratic ----
    double A, B, C;
    if (solve_xi_first) {
        A = b1 * a3 - a1 * b3;
        B = dx * b3 - a1 * b2 - dy * a3 + b1 * a2;
        C = dx * b2 - dy * a2;
    } else {
        A = b2 * a3 - a2 * b3;
        B = dx * b3 - a2 * b1 - dy * a3 + b2 * a1;
        C = dx * b1 - dy * a1;
    }

    // ---- 3a. Degenerate: A ~ 0 (nearly rectangular -> linear equation) ----
    const double epsA = 1e-14;
    double u1 = 999.0, u2 = 999.0;
    bool has_roots = false;

    if (std::abs(A) < epsA) {
        if (std::abs(B) > 1e-14) {
            u1 = -C / B;
            has_roots = true;
        }
    } else {
        double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0) {
            double sqrt_disc = std::sqrt(disc);
            u1 = (-B + sqrt_disc) / (2.0 * A);
            u2 = (-B - sqrt_disc) / (2.0 * A);
            has_roots = true;
        }
    }

    // ---- 4. Test candidate roots ----
    if (has_roots) {
        for (int trial = 0; trial < 2; ++trial) {
            double u = (trial == 0) ? u1 : u2;
            if (std::abs(u) > 999.0) continue;

            double cs, ct;  // candidate (xi, eta) or (eta, xi) depending on strategy
            if (solve_xi_first) {
                cs = u;
                double den = a2 + a3 * cs;
                if (std::abs(den) < 1e-12) continue;
                ct = (dx - a1 * cs) / den;
            } else {
                ct = u;
                double den = a1 + a3 * ct;
                if (std::abs(den) < 1e-12) continue;
                cs = (dx - a2 * ct) / den;
            }

            double xi_c  = solve_xi_first ? cs : ct;
            double eta_c = solve_xi_first ? ct : cs;

            // Range check
            if (xi_c < -1.15 || xi_c > 1.15) continue;
            if (eta_c < -1.15 || eta_c > 1.15) continue;

            // ---- 5. Verify with round-trip residual ----
            double N[4], dN_dxi[4], dN_deta[4];
            shape_functions_q4(xi_c, eta_c, N, dN_dxi, dN_deta);
            double xp = 0.0, yp = 0.0;
            for (int i = 0; i < 4; ++i) {
                xp += N[i] * elem_nodes[2 * i];
                yp += N[i] * elem_nodes[2 * i + 1];
            }
            double err = std::sqrt((gx - xp) * (gx - xp) + (gy - yp) * (gy - yp));
            if (err < 1e-8) {
                xi  = xi_c;
                eta = eta_c;
                // Jacobian at solution
                J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
                for (int i = 0; i < 4; ++i) {
                    J11 += dN_dxi[i]  * elem_nodes[2 * i];
                    J12 += dN_deta[i] * elem_nodes[2 * i];
                    J21 += dN_dxi[i]  * elem_nodes[2 * i + 1];
                    J22 += dN_deta[i] * elem_nodes[2 * i + 1];
                }
                return true;
            }
        }
    }

    // ---- 6. Fallback: Newton-Raphson ----
    xi = 0.0; eta = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        double N[4], dN_dxi[4], dN_deta[4];
        shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);

        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 4; ++i) {
            xp += N[i] * elem_nodes[2 * i];
            yp += N[i] * elem_nodes[2 * i + 1];
        }
        double rx = gx - xp, ry = gy - yp;

        if (std::sqrt(rx * rx + ry * ry) < 1e-8) {
            if (std::abs(xi) > 1.2 || std::abs(eta) > 1.2) return false;
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 4; ++i) {
                J11 += dN_dxi[i]  * elem_nodes[2 * i];
                J12 += dN_deta[i] * elem_nodes[2 * i];
                J21 += dN_dxi[i]  * elem_nodes[2 * i + 1];
                J22 += dN_deta[i] * elem_nodes[2 * i + 1];
            }
            return true;
        }

        double j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
        for (int i = 0; i < 4; ++i) {
            j11 += dN_dxi[i]  * elem_nodes[2 * i];
            j12 += dN_deta[i] * elem_nodes[2 * i];
            j21 += dN_dxi[i]  * elem_nodes[2 * i + 1];
            j22 += dN_deta[i] * elem_nodes[2 * i + 1];
        }
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-12) return false;

        double inv_det = 1.0 / det;
        xi  += inv_det * (j22 * rx - j12 * ry);
        eta += inv_det * (-j21 * rx + j11 * ry);

        if (xi  < -2.0) xi  = -2.0;
        if (xi  >  2.0) xi  =  2.0;
        if (eta < -2.0) eta = -2.0;
        if (eta >  2.0) eta =  2.0;
    }
    return false;
}

// ============================================================

//



// ============================================================
bool solve_point_q8(
    double gx, double gy,
    const double* elem8,
    double xi0, double eta0,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    double tol, int max_iter)
{
    xi = xi0; eta = eta0;
    const double max_step = 0.5;   
    const double clip     = 2.0;   

    for (int it = 0; it < max_iter; ++it) {
        double N[8], dN_dxi[8], dN_deta[8];
        shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);

        
        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp += N[i] * elem8[2 * i];
            yp += N[i] * elem8[2 * i + 1];
        }
        double rx = gx - xp, ry = gy - yp;

        
        if (std::sqrt(rx * rx + ry * ry) < tol) {
            if (std::abs(xi) > 2.0 || std::abs(eta) > 2.0) return false;
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dN_dxi[i]  * elem8[2 * i];
                J12 += dN_deta[i] * elem8[2 * i];
                J21 += dN_dxi[i]  * elem8[2 * i + 1];
                J22 += dN_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        
        double j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
        for (int i = 0; i < 8; ++i) {
            j11 += dN_dxi[i]  * elem8[2 * i];
            j12 += dN_deta[i] * elem8[2 * i];
            j21 += dN_dxi[i]  * elem8[2 * i + 1];
            j22 += dN_deta[i] * elem8[2 * i + 1];
        }
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-12) return false;

        double inv_det = 1.0 / det;
        double dxi  = inv_det * (j22 * rx - j12 * ry);
        double deta = inv_det * (-j21 * rx + j11 * ry);

        
        double dn = std::sqrt(dxi * dxi + deta * deta);
        if (dn > max_step) {
            double s = max_step / dn;
            dxi  *= s; deta *= s;
        }
        xi  += dxi;
        eta += deta;

        
        if (xi  < -clip) xi  = -clip;
        if (xi  >  clip) xi  =  clip;
        if (eta < -clip) eta = -clip;
        if (eta >  clip) eta =  clip;
    }
    return false;
}

// ============================================================

//




// ============================================================
bool solve_point_q8_fallback(
    double gx, double gy,
    const double* elem8,
    double xi0, double eta0,
    double& xi, double& eta,
    double& J11, double& J12, double& J21, double& J22,
    double tol_global, double tol_local, int max_iter)
{
    
    double xmin = elem8[0], xmax = elem8[0];
    double ymin = elem8[1], ymax = elem8[1];
    for (int i = 1; i < 8; ++i) {
        xmin = std::min(xmin, elem8[2 * i]);
        xmax = std::max(xmax, elem8[2 * i]);
        ymin = std::min(ymin, elem8[2 * i + 1]);
        ymax = std::max(ymax, elem8[2 * i + 1]);
    }
    double cx = 0.5 * (xmin + xmax);
    double cy = 0.5 * (ymin + ymax);
    double sx = std::max(0.5 * (xmax - xmin) * 2.0, 1.0);  
    double sy = std::max(0.5 * (ymax - ymin) * 2.0, 1.0);

    
    double nodes_norm[16];
    for (int i = 0; i < 8; ++i) {
        nodes_norm[2 * i]     = (elem8[2 * i]     - cx) / sx;
        nodes_norm[2 * i + 1] = (elem8[2 * i + 1] - cy) / sy;
    }
    double px = (gx - cx) / sx;
    double py = (gy - cy) / sy;

    xi = xi0; eta = eta0;
    double prev_xi = xi, prev_eta = eta;
    double prev2_xi = xi, prev2_eta = eta;

    for (int it = 0; it < max_iter; ++it) {
        double N[8], dN_dxi[8], dN_deta[8];
        shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);

        
        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp += N[i] * nodes_norm[2 * i];
            yp += N[i] * nodes_norm[2 * i + 1];
        }
        double rx = px - xp, ry = py - yp;

        
        double xp_orig = 0.0, yp_orig = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp_orig += N[i] * elem8[2 * i];
            yp_orig += N[i] * elem8[2 * i + 1];
        }
        double res_orig = std::sqrt((gx - xp_orig) * (gx - xp_orig) +
                                    (gy - yp_orig) * (gy - yp_orig));

        if (res_orig < tol_global) {
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dN_dxi[i]  * elem8[2 * i];
                J12 += dN_deta[i] * elem8[2 * i];
                J21 += dN_dxi[i]  * elem8[2 * i + 1];
                J22 += dN_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        
        double jn11 = 0.0, jn12 = 0.0, jn21 = 0.0, jn22 = 0.0;
        for (int i = 0; i < 8; ++i) {
            jn11 += dN_dxi[i]  * nodes_norm[2 * i];
            jn12 += dN_deta[i] * nodes_norm[2 * i];
            jn21 += dN_dxi[i]  * nodes_norm[2 * i + 1];
            jn22 += dN_deta[i] * nodes_norm[2 * i + 1];
        }
        double det = jn11 * jn22 - jn12 * jn21;
        if (std::abs(det) < 1e-12) return false;

        double inv_det = 1.0 / det;
        double dxi  = inv_det * (jn22 * rx - jn12 * ry);
        double deta = inv_det * (-jn21 * rx + jn11 * ry);

        
        if (std::abs(det) < 1e-2) {
            double dn = std::sqrt(dxi * dxi + deta * deta);
            const double max_step = 0.3;
            if (dn > max_step) {
                double s = max_step / dn;
                dxi  *= s; deta *= s;
            }
        }

        
        double alpha = 1.0;
        double xi_trial  = xi  + alpha * dxi;
        double eta_trial = eta + alpha * deta;

        
        if (std::abs(xi_trial - prev2_xi) < 1e-12 &&
            std::abs(eta_trial - prev2_eta) < 1e-12) {
            alpha = 0.25;
            xi_trial  = xi  + alpha * dxi;
            eta_trial = eta + alpha * deta;
        }

        for (int bt = 0; bt < 10; ++bt) {
            double Nb[8], dNb_dxi[8], dNb_deta[8];
            shape_functions_q8(xi_trial, eta_trial, Nb, dNb_dxi, dNb_deta);
            double xb = 0.0, yb = 0.0;
            for (int i = 0; i < 8; ++i) {
                xb += Nb[i] * elem8[2 * i];
                yb += Nb[i] * elem8[2 * i + 1];
            }
            double res_b = std::sqrt((gx - xb) * (gx - xb) + (gy - yb) * (gy - yb));
            if (res_b <= res_orig * 1.01 || alpha < 1e-4) break;
            alpha *= 0.5;
            xi_trial  = xi  + alpha * dxi;
            eta_trial = eta + alpha * deta;
        }

        prev2_xi = prev_xi; prev2_eta = prev_eta;
        prev_xi = xi; prev_eta = eta;
        xi  = xi_trial; eta = eta_trial;

        
        double step_norm = alpha * std::sqrt(dxi * dxi + deta * deta);
        if (step_norm < tol_local) {
            double Nf[8], dNf_dxi[8], dNf_deta[8];
            shape_functions_q8(xi, eta, Nf, dNf_dxi, dNf_deta);
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dNf_dxi[i]  * elem8[2 * i];
                J12 += dNf_deta[i] * elem8[2 * i];
                J21 += dNf_dxi[i]  * elem8[2 * i + 1];
                J22 += dNf_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        
        if (res_orig > 100.0) return false;
    }
    return false;
}

// ============================================================

//






// ============================================================
G2LOutput compute_global_to_local(
    const double* inform, int n_pixels,
    const double* nodes_coord, int n_nodes,
    const int* elements, int n_elements,
    int img_h, int img_w,
    MeshElementType element_type,
    const G2LParams& params)
{
    G2LOutput out;
    int total = img_h * img_w;

    out.xi.resize(total, 0.0);
    out.eta.resize(total, 0.0);
    out.J11.resize(total, 0.0);
    out.J12.resize(total, 0.0);
    out.J21.resize(total, 0.0);
    out.J22.resize(total, 0.0);
    out.valid.resize(total, 0);
    out.elem_id.resize(total, -1);
    out.img_h = img_h;
    out.img_w = img_w;

    int nn = nodes_per_element(element_type);
    
    int elem_stride = (element_type == MeshElementType::Q8) ? 9 : nn;

    
    std::vector<std::vector<double>> elem_nodes(n_elements);
    for (int e = 0; e < n_elements; ++e) {
        elem_nodes[e].resize(2 * nn);
        for (int k = 0; k < nn; ++k) {
            int nid = elements[e * elem_stride + k] - 1;  
            elem_nodes[e][2 * k]     = nodes_coord[2 * nid];
            elem_nodes[e][2 * k + 1] = nodes_coord[2 * nid + 1];
        }
    }

    
    for (int p = 0; p < n_pixels; ++p) {
        double gx  = inform[p * 3];
        double gy  = inform[p * 3 + 1];
        int    eid = static_cast<int>(inform[p * 3 + 2]) - 1;  
        int    idx = static_cast<int>(gy) * img_w + static_cast<int>(gx);
        if (idx < 0 || idx >= total) continue;

        double xi = 0.0, eta = 0.0;
        double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
        bool ok = false;

        switch (element_type) {
            
            case MeshElementType::T3:
                ok = solve_point_t3(gx, gy, elem_nodes[eid].data(),
                                    xi, eta, J11, J12, J21, J22);
                break;

            
            case MeshElementType::Q4:
                ok = solve_point_q4(gx, gy, elem_nodes[eid].data(),
                                    xi, eta, J11, J12, J21, J22,
                                    params.max_iter);
                break;

            
            case MeshElementType::Q8: {
                
                double xi0 = 0.0, eta0 = 0.0;
                double q4_J11 = 0.0, q4_J12 = 0.0, q4_J21 = 0.0, q4_J22 = 0.0;
                bool q4_seed_ok = solve_point_q4(
                    gx, gy, elem_nodes[eid].data(),
                    xi0, eta0, q4_J11, q4_J12, q4_J21, q4_J22,
                    params.max_iter);
                if (!q4_seed_ok) {
                    xi0 = 0.0; eta0 = 0.0;  
                }

                
                ok = solve_point_q8(gx, gy, elem_nodes[eid].data(),
                                    xi0, eta0,
                                    xi, eta, J11, J12, J21, J22,
                                    params.tol_global, params.max_iter);

                
                if (!ok) {
                    ok = solve_point_q8_fallback(gx, gy, elem_nodes[eid].data(),
                                                 xi0, eta0,
                                                 xi, eta, J11, J12, J21, J22,
                                                 params.tol_global, params.tol_local,
                                                 params.max_iter);
                }
                break;
            }
        }

        if (ok) {
            out.xi[idx]      = xi;
            out.eta[idx]     = eta;
            out.J11[idx]     = J11;
            out.J12[idx]     = J12;
            out.J21[idx]     = J21;
            out.J22[idx]     = J22;
            out.valid[idx]   = 1;
            out.elem_id[idx] = eid + 1;  
        }
    }

    return out;
}

} // namespace dic::mesh::internal
