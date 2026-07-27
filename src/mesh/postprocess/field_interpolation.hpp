#pragma once

#include "../coordinate/g2l_internal.hpp"
#include "../element/shape_func_internal.hpp"

#include <Eigen/Dense>
#include <vector>
#include <limits>

namespace dic::mesh::internal {

// Interpolate nodal displacements to full-field pixel maps using
// isoparametric shape functions.
//
// For each valid G2L pixel (xi, eta, elem_id):
//   u_field[idx] = sum_k N_k(xi,eta) * U(2*k)
//   v_field[idx] = sum_k N_k(xi,eta) * U(2*k+1)
//
// Pixels without valid G2L are set to NaN.

inline void interpolate_displacement_to_pixels(
    const G2LOutput& g2l,
    const int* elements, int n_elements,
    mesh::MeshElementType element_type,
    const Eigen::VectorXd& U,
    int img_h, int img_w,
    std::vector<double>& u_field,
    std::vector<double>& v_field)
{
    int total = img_h * img_w;
    double nan = std::numeric_limits<double>::quiet_NaN();

    u_field.assign(total, nan);
    v_field.assign(total, nan);

    int nn = nodes_per_element(element_type);
    int elem_stride = (element_type == mesh::MeshElementType::Q8) ? 9 : nn;

    std::vector<double> N(nn), dN_dxi(nn), dN_deta(nn);

    for (int idx = 0; idx < total; ++idx) {
        if (!g2l.valid[idx]) continue;

        int eid = g2l.elem_id[idx] - 1;
        if (eid < 0 || eid >= n_elements) continue;

        double xi = g2l.xi[idx];
        double eta = g2l.eta[idx];

        // Evaluate shape functions at (xi, eta)
        shape_functions(element_type, xi, eta,
                        N.data(), dN_dxi.data(), dN_deta.data());

        // Interpolate displacement
        double u_ip = 0.0, v_ip = 0.0;
        for (int k = 0; k < nn; ++k) {
            int nid = elements[eid * elem_stride + k] - 1;
            u_ip += N[k] * U(2 * nid);
            v_ip += N[k] * U(2 * nid + 1);
        }

        u_field[idx] = u_ip;
        v_field[idx] = v_ip;
    }
}

} // namespace dic::mesh::internal
