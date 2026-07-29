/**
 * @file triangulation.cpp
 * @brief Camera-agnostic point triangulation.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/geometry/triangulation.hpp>

#include <dic/geometry/projection.hpp>

#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dic {

namespace {

Eigen::Vector2d pixel_to_normalized_undistorted(const Eigen::Vector2d& pixel,
                                                const CameraModel& camera)
{
    double x = (pixel.x() - camera.K(0, 2)) / camera.K(0, 0);
    double y = (pixel.y() - camera.K(1, 2)) / camera.K(1, 1);

    const double k1 = camera.distortion.size() > 0 ? camera.distortion[0] : 0.0;
    const double k2 = camera.distortion.size() > 1 ? camera.distortion[1] : 0.0;
    const double p1 = camera.distortion.size() > 2 ? camera.distortion[2] : 0.0;
    const double p2 = camera.distortion.size() > 3 ? camera.distortion[3] : 0.0;
    const double k3 = camera.distortion.size() > 4 ? camera.distortion[4] : 0.0;

    if (std::abs(k1) < 1.0e-12 && std::abs(k2) < 1.0e-12 &&
        std::abs(p1) < 1.0e-12 && std::abs(p2) < 1.0e-12 &&
        std::abs(k3) < 1.0e-12) {
        return {x, y};
    }

    double xu = x;
    double yu = y;
    for (int iter = 0; iter < 12; ++iter) {
        const double r2 = xu * xu + yu * yu;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        if (std::abs(radial) <= 1.0e-12) {
            break;
        }
        const double tangential_x = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
        const double tangential_y = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
        xu = (x - tangential_x) / radial;
        yu = (y - tangential_y) / radial;
    }
    return {xu, yu};
}

Eigen::Matrix<double, 3, 4> extrinsic_matrix(const CameraModel& camera)
{
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = camera.R;
    P.col(3) = camera.t;
    return P;
}

} // namespace

TriangulationResult triangulate_points_checked(const std::vector<Eigen::Vector2d>& observations,
                                               const std::vector<CameraModel>& cameras,
                                               const TriangulationOptions& options)
{
    if (observations.size() != cameras.size()) {
        throw std::invalid_argument("triangulate_points requires the same number of observations and cameras.");
    }
    if (observations.size() < 2) {
        throw std::invalid_argument("triangulate_points requires at least two observations.");
    }

    Eigen::Vector4d homogeneous;
    if (observations.size() == 2) {
        Eigen::Matrix4d A;
        for (std::size_t i = 0; i < 2; ++i) {
            const Eigen::Vector2d normalized = pixel_to_normalized_undistorted(observations[i], cameras[i]);
            const auto P = extrinsic_matrix(cameras[i]);
            A.row(static_cast<Eigen::Index>(2 * i)) = normalized.x() * P.row(2) - P.row(0);
            A.row(static_cast<Eigen::Index>(2 * i + 1)) = normalized.y() * P.row(2) - P.row(1);
        }
        const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        homogeneous = svd.matrixV().col(3);
    } else {
        Eigen::MatrixXd A(static_cast<Eigen::Index>(observations.size() * 2), 4);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const Eigen::Vector2d normalized = pixel_to_normalized_undistorted(observations[i], cameras[i]);
            const auto P = extrinsic_matrix(cameras[i]);
            A.row(static_cast<Eigen::Index>(2 * i)) = normalized.x() * P.row(2) - P.row(0);
            A.row(static_cast<Eigen::Index>(2 * i + 1)) = normalized.y() * P.row(2) - P.row(1);
        }
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        homogeneous = svd.matrixV().col(3);
    }
    TriangulationResult result;
    result.observations_used = static_cast<int>(observations.size());
    if (std::abs(homogeneous.w()) < 1e-12) {
        return result;
    }

    result.point = homogeneous.head<3>() / homogeneous.w();
    if (!result.point.allFinite()) {
        return result;
    }

    double sum_error = 0.0;
    result.max_reprojection_error = 0.0;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        if (options.require_positive_depth && camera_depth(result.point, cameras[i]) <= 0.0) {
            return result;
        }
        const double error = reprojection_error(result.point, observations[i], cameras[i]);
        if (!std::isfinite(error)) {
            return result;
        }
        sum_error += error;
        result.max_reprojection_error = std::max(result.max_reprojection_error, error);
    }
    result.mean_reprojection_error = sum_error / static_cast<double>(observations.size());
    result.valid = result.max_reprojection_error <= options.max_reprojection_error;
    return result;
}

Eigen::Vector3d triangulate_points(const std::vector<Eigen::Vector2d>& observations,
                                   const std::vector<CameraModel>& cameras)
{
    const TriangulationResult result = triangulate_points_checked(observations, cameras);
    if (!result.valid) {
        throw std::runtime_error("Point triangulation failed quality checks.");
    }
    return result.point;
}

} // namespace dic
