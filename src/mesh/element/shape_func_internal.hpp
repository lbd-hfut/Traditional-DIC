#pragma once

#include <dic/mesh/mesh_generation_config.hpp>

#include <cstddef>

namespace dic::mesh::internal {

// ---- Element metadata ----

constexpr int nodes_per_element(MeshElementType t) {
    switch (t) {
        case MeshElementType::T3: return 3;
        case MeshElementType::Q4: return 4;
        case MeshElementType::Q8: return 8;
    }
    return 8;
}

constexpr int dofs_per_element(MeshElementType t) {
    return 2 * nodes_per_element(t);
}

// ---- Q8 (8-node serendipity quadrilateral) ----
// Corner nodes: 0(-1,-1) 1(+1,-1) 2(+1,+1) 3(-1,+1)
// Mid-sides:    4(0,-1)  5(+1,0)  6(0,+1)  7(-1,0)
void shape_functions_q8(double xi, double eta,
                        double N[8], double dN_dxi[8], double dN_deta[8]);

// ---- Q4 (4-node bilinear quadrilateral) ----
// Nodes: 0(-1,-1) 1(+1,-1) 2(+1,+1) 3(-1,+1)
void shape_functions_q4(double xi, double eta,
                        double N[4], double dN_dxi[4], double dN_deta[4]);

// ---- T3 (3-node linear triangle) ----
// N1 = 1 - xi - eta, N2 = xi, N3 = eta
// Domain: xi >= 0, eta >= 0, xi + eta <= 1
void shape_functions_t3(double xi, double eta,
                        double N[3], double dN_dxi[3], double dN_deta[3]);

// ---- Generic dispatch ----
void shape_functions(MeshElementType type, double xi, double eta,
                     double* N, double* dN_dxi, double* dN_deta);

} // namespace dic::mesh::internal
