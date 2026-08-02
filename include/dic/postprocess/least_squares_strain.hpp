#pragma once

#include <Eigen/Dense>

#include <vector>

namespace dic {

struct LeastSquaresStrain2D {
    double du_dx{0.0};
    double du_dy{0.0};
    double dv_dx{0.0};
    double dv_dy{0.0};
    double exx{0.0};
    double eyy{0.0};
    double exy{0.0};
    int sample_count{0};
    bool valid{false};
};

std::vector<LeastSquaresStrain2D> compute_least_squares_strain_2d(
    const Eigen::MatrixXd& points,
    const Eigen::MatrixXd& displacement,
    double radius,
    int min_samples,
    bool green_lagrange);

std::vector<LeastSquaresStrain2D> compute_mesh_least_squares_strain_2d(
    const Eigen::MatrixXd& nodes,
    const Eigen::MatrixXd& displacement,
    const Eigen::MatrixXi& elements,
    int min_samples,
    bool green_lagrange);

} // namespace dic
