#include <dic/mesh/element/q4.hpp>

#include "shape_func_internal.hpp"

namespace dic {

int Q4Element::node_count() const { return 4; }

Eigen::VectorXd Q4Element::shape_functions(double xi, double eta) const {
    double N[4], dN_dxi[4], dN_deta[4];
    mesh::internal::shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);
    Eigen::VectorXd result(4);
    for (int i = 0; i < 4; ++i) result[i] = N[i];
    return result;
}

Eigen::MatrixXd Q4Element::shape_function_derivatives(double xi, double eta) const {
    double N[4], dN_dxi[4], dN_deta[4];
    mesh::internal::shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);
    Eigen::MatrixXd result(4, 2);
    for (int i = 0; i < 4; ++i) {
        result(i, 0) = dN_dxi[i];
        result(i, 1) = dN_deta[i];
    }
    return result;
}

} // namespace dic
