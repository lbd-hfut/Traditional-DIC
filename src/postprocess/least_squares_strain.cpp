#include <dic/postprocess/least_squares_strain.hpp>

#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>

namespace dic {
namespace {

LeastSquaresStrain2D fit_strain(const Eigen::MatrixXd& points, const Eigen::MatrixXd& displacement,
                                 int center, const std::vector<int>& ids, int min_samples, bool green_lagrange) {
    LeastSquaresStrain2D out;
    out.sample_count = static_cast<int>(ids.size());
    if (static_cast<int>(ids.size()) < min_samples) return out;
    Eigen::MatrixXd a(ids.size(), 3);
    Eigen::VectorXd u(ids.size()), v(ids.size());
    const double x0 = points(center, 0), y0 = points(center, 1);
    for (int row = 0; row < static_cast<int>(ids.size()); ++row) {
        const int id = ids[row];
        a(row, 0) = 1.0;
        a(row, 1) = points(id, 0) - x0;
        a(row, 2) = points(id, 1) - y0;
        u(row) = displacement(id, 0);
        v(row) = displacement(id, 1);
    }
    const Eigen::Vector3d au = a.colPivHouseholderQr().solve(u);
    const Eigen::Vector3d av = a.colPivHouseholderQr().solve(v);
    if (!au.allFinite() || !av.allFinite()) return out;
    out.du_dx = au(1); out.du_dy = au(2); out.dv_dx = av(1); out.dv_dy = av(2);
    if (green_lagrange) {
        const Eigen::Matrix2d h = (Eigen::Matrix2d() << out.du_dx, out.du_dy, out.dv_dx, out.dv_dy).finished();
        const Eigen::Matrix2d e = 0.5 * (h + h.transpose() + h.transpose() * h);
        out.exx = e(0, 0); out.eyy = e(1, 1); out.exy = e(0, 1);
    } else {
        out.exx = out.du_dx; out.eyy = out.dv_dy; out.exy = 0.5 * (out.du_dy + out.dv_dx);
    }
    out.valid = true;
    return out;
}

void validate_input(const Eigen::MatrixXd& points, const Eigen::MatrixXd& displacement) {
    if (points.cols() != 2 || displacement.cols() != 2 || points.rows() != displacement.rows())
        throw std::invalid_argument("points and displacement must both be Nx2 matrices");
}
} // namespace

std::vector<LeastSquaresStrain2D> compute_least_squares_strain_2d(
    const Eigen::MatrixXd& points, const Eigen::MatrixXd& displacement,
    double radius, int min_samples, bool green_lagrange) {
    validate_input(points, displacement);
    if (!(radius > 0.0) || min_samples < 3) throw std::invalid_argument("radius must be positive and min_samples at least 3");
    std::vector<LeastSquaresStrain2D> result(points.rows());
    const double radius_sq = radius * radius;
    using Bucket = std::pair<int, int>;
    const auto key = [](int x, int y) { return (static_cast<long long>(x) << 32) ^ static_cast<unsigned int>(y); };
    std::unordered_map<long long, std::vector<int>> buckets;
    for (int i = 0; i < points.rows(); ++i) {
        if (points.row(i).allFinite() && displacement.row(i).allFinite()) {
            buckets[key(static_cast<int>(std::floor(points(i, 0) / radius)), static_cast<int>(std::floor(points(i, 1) / radius)))].push_back(i);
        }
    }
    for (int i = 0; i < points.rows(); ++i) {
        if (!points.row(i).allFinite() || !displacement.row(i).allFinite()) continue;
        std::vector<int> ids;
        const int bx = static_cast<int>(std::floor(points(i, 0) / radius));
        const int by = static_cast<int>(std::floor(points(i, 1) / radius));
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const auto found = buckets.find(key(bx + dx, by + dy));
            if (found == buckets.end()) continue;
            for (const int j : found->second) if ((points.row(j) - points.row(i)).squaredNorm() <= radius_sq) ids.push_back(j);
        }
        result[i] = fit_strain(points, displacement, i, ids, min_samples, green_lagrange);
    }
    return result;
}

std::vector<LeastSquaresStrain2D> compute_mesh_least_squares_strain_2d(
    const Eigen::MatrixXd& nodes, const Eigen::MatrixXd& displacement,
    const Eigen::MatrixXi& elements, int min_samples, bool green_lagrange) {
    validate_input(nodes, displacement);
    if (min_samples < 3) throw std::invalid_argument("min_samples must be at least 3");
    std::vector<std::unordered_set<int>> adjacency(nodes.rows());
    for (int e = 0; e < elements.rows(); ++e) for (int a = 0; a < elements.cols(); ++a) {
        const int i = elements(e, a);
        if (i < 0 || i >= nodes.rows()) continue;
        for (int b = 0; b < elements.cols(); ++b) {
            const int j = elements(e, b);
            if (j >= 0 && j < nodes.rows()) adjacency[i].insert(j);
        }
    }
    std::vector<LeastSquaresStrain2D> result(nodes.rows());
    for (int i = 0; i < nodes.rows(); ++i) {
        std::vector<int> ids(adjacency[i].begin(), adjacency[i].end());
        result[i] = fit_strain(nodes, displacement, i, ids, min_samples, green_lagrange);
    }
    return result;
}

} // namespace dic
