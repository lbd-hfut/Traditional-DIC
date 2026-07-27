#include "shape_func_internal.hpp"

#include <algorithm>

namespace dic::mesh::internal {

// ============================================================
// Q8  shape functions (8-node serendipity quadrilateral)
// ============================================================
void shape_functions_q8(double xi, double eta,
                        double N[8], double dN_dxi[8], double dN_deta[8]) {
    double xi2 = xi * xi;
    double eta2 = eta * eta;

    N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta);
    N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta);
    N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta);
    N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta);
    N[4] =  0.50 * (1.0 - xi2) * (1.0 - eta);
    N[5] =  0.50 * (1.0 + xi) * (1.0 - eta2);
    N[6] =  0.50 * (1.0 - xi2) * (1.0 + eta);
    N[7] =  0.50 * (1.0 - xi) * (1.0 - eta2);

    dN_dxi[0] = -0.25 * ((eta - 1.0) * (2.0 * xi + eta));
    dN_dxi[1] =  0.25 * ((1.0 - eta) * (2.0 * xi - eta));
    dN_dxi[2] =  0.25 * ((1.0 + eta) * (2.0 * xi + eta));
    dN_dxi[3] =  0.25 * ((1.0 + eta) * (2.0 * xi - eta));
    dN_dxi[4] = -xi * (1.0 - eta);
    dN_dxi[5] =  0.50 * (1.0 - eta2);
    dN_dxi[6] = -xi * (1.0 + eta);
    dN_dxi[7] = -0.50 * (1.0 - eta2);

    dN_deta[0] = -0.25 * ((xi - 1.0) * (xi + 2.0 * eta));
    dN_deta[1] = -0.25 * ((xi + 1.0) * (xi - 2.0 * eta));
    dN_deta[2] =  0.25 * ((xi + 1.0) * (xi + 2.0 * eta));
    dN_deta[3] =  0.25 * ((xi - 1.0) * (xi - 2.0 * eta));
    dN_deta[4] = -0.50 * (1.0 - xi2);
    dN_deta[5] = -eta * (1.0 + xi);
    dN_deta[6] =  0.50 * (1.0 - xi2);
    dN_deta[7] = -eta * (1.0 - xi);
}

// ============================================================
// Q4  shape functions  (bilinear quadrilateral, xi,eta in [-1,1])
// ============================================================
void shape_functions_q4(double xi, double eta,
                        double N[4], double dN_dxi[4], double dN_deta[4]) {
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    dN_dxi[0] = -0.25 * (1.0 - eta);
    dN_dxi[1] =  0.25 * (1.0 - eta);
    dN_dxi[2] =  0.25 * (1.0 + eta);
    dN_dxi[3] = -0.25 * (1.0 + eta);

    dN_deta[0] = -0.25 * (1.0 - xi);
    dN_deta[1] = -0.25 * (1.0 + xi);
    dN_deta[2] =  0.25 * (1.0 + xi);
    dN_deta[3] =  0.25 * (1.0 - xi);
}

// ============================================================
// T3  shape functions  (linear triangle)
//   N1 = 1 - xi - eta,  N2 = xi,  N3 = eta
//   Domain: xi >= 0, eta >= 0, xi + eta <= 1
// ============================================================
void shape_functions_t3(double xi, double eta,
                        double N[3], double dN_dxi[3], double dN_deta[3]) {
    N[0] = 1.0 - xi - eta;
    N[1] = xi;
    N[2] = eta;

    dN_dxi[0] = -1.0;
    dN_dxi[1] =  1.0;
    dN_dxi[2] =  0.0;

    dN_deta[0] = -1.0;
    dN_deta[1] =  0.0;
    dN_deta[2] =  1.0;
}

// ============================================================
// Generic dispatch
// ============================================================
void shape_functions(MeshElementType type, double xi, double eta,
                     double* N, double* dN_dxi, double* dN_deta) {
    switch (type) {
        case MeshElementType::T3:
            shape_functions_t3(xi, eta, N, dN_dxi, dN_deta);
            break;
        case MeshElementType::Q4:
            shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);
            break;
        case MeshElementType::Q8:
        default: {
            double n8[8], dx8[8], de8[8];
            shape_functions_q8(xi, eta, n8, dx8, de8);
            for (int i = 0; i < 8; ++i) {
                N[i] = n8[i];
                dN_dxi[i] = dx8[i];
                dN_deta[i] = de8[i];
            }
            break;
        }
    }
}

} // namespace dic::mesh::internal
