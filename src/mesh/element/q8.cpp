#include <dic/mesh/element/q8.hpp>

#include "shape_func_internal.hpp"

namespace dic {

int Q8Element::node_count() const { return 8; }

Eigen::VectorXd Q8Element::shape_functions(double xi, double eta) const {
    double N[8], dN_dxi[8], dN_deta[8];
    mesh::internal::shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);
    Eigen::VectorXd result(8);
    for (int i = 0; i < 8; ++i) result[i] = N[i];
    return result;
}

Eigen::MatrixXd Q8Element::shape_function_derivatives(double xi, double eta) const {
    double N[8], dN_dxi[8], dN_deta[8];
    mesh::internal::shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);
    Eigen::MatrixXd result(8, 2);
    for (int i = 0; i < 8; ++i) {
        result(i, 0) = dN_dxi[i];
        result(i, 1) = dN_deta[i];
    }
    return result;
}

} // namespace dic
