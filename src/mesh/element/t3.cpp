#include <dic/mesh/element/t3.hpp>

#include "shape_func_internal.hpp"

namespace dic {

int T3Element::node_count() const { return 3; }

Eigen::VectorXd T3Element::shape_functions(double xi, double eta) const {
    double N[3], dN_dxi[3], dN_deta[3];
    mesh::internal::shape_functions_t3(xi, eta, N, dN_dxi, dN_deta);
    Eigen::VectorXd result(3);
    for (int i = 0; i < 3; ++i) result[i] = N[i];
    return result;
}

Eigen::MatrixXd T3Element::shape_function_derivatives(double xi, double eta) const {
    double N[3], dN_dxi[3], dN_deta[3];
    mesh::internal::shape_functions_t3(xi, eta, N, dN_dxi, dN_deta);
    Eigen::MatrixXd result(3, 2);
    for (int i = 0; i < 3; ++i) {
        result(i, 0) = dN_dxi[i];
        result(i, 1) = dN_deta[i];
    }
    return result;
}

} // namespace dic
