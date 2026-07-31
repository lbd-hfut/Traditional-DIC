#include <dic/core/observation_mask.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
#include <opencv2/imgproc.hpp>
#endif

namespace dic {
namespace {

double median(std::vector<double> values)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }), values.end());
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    double out = *mid;
    if (values.size() % 2 == 0) {
        const auto mid_low = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2 - 1);
        std::nth_element(values.begin(), mid_low, values.end());
        out = 0.5 * (out + *mid_low);
    }
    return out;
}

double distance2(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return (a - b).squaredNorm();
}

std::vector<double> sorted_neighbor_distances(const std::vector<Eigen::Vector2d>& uv, int idx)
{
    std::vector<double> dists;
    dists.reserve(uv.size() > 0 ? uv.size() - 1 : 0);
    for (int j = 0; j < static_cast<int>(uv.size()); ++j) {
        if (j == idx) {
            continue;
        }
        dists.push_back(std::sqrt(distance2(uv[static_cast<size_t>(idx)], uv[static_cast<size_t>(j)])));
    }
    std::sort(dists.begin(), dists.end());
    return dists;
}

double median_nn_distance(const std::vector<Eigen::Vector2d>& uv)
{
    if (uv.size() < 2) {
        return 0.0;
    }
    std::vector<double> nearest;
    nearest.reserve(uv.size());
    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < static_cast<int>(uv.size()); ++j) {
            if (i == j) {
                continue;
            }
            best = std::min(best, std::sqrt(distance2(uv[static_cast<size_t>(i)], uv[static_cast<size_t>(j)])));
        }
        if (std::isfinite(best) && best > 0.0) {
            nearest.push_back(best);
        }
    }
    return median(nearest);
}

std::vector<Eigen::Vector2d> keep_largest_radius_component(const std::vector<Eigen::Vector2d>& uv, double radius)
{
    if (uv.size() < 3 || radius <= 0.0) {
        return uv;
    }
    const double radius2 = radius * radius;
    std::vector<int> parent(uv.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int value) {
        int root = value;
        while (parent[static_cast<size_t>(root)] != root) {
            root = parent[static_cast<size_t>(root)];
        }
        while (parent[static_cast<size_t>(value)] != value) {
            const int next = parent[static_cast<size_t>(value)];
            parent[static_cast<size_t>(value)] = root;
            value = next;
        }
        return root;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a);
        const int rb = find(b);
        if (ra != rb) {
            parent[static_cast<size_t>(rb)] = ra;
        }
    };
    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(uv.size()); ++j) {
            if (distance2(uv[static_cast<size_t>(i)], uv[static_cast<size_t>(j)]) <= radius2) {
                unite(i, j);
            }
        }
    }
    std::unordered_map<int, int> counts;
    int best_root = -1;
    int best_count = 0;
    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        const int root = find(i);
        const int count = ++counts[root];
        if (count > best_count) {
            best_count = count;
            best_root = root;
        }
    }
    if (best_count < 3) {
        return uv;
    }
    std::vector<Eigen::Vector2d> out;
    out.reserve(static_cast<size_t>(best_count));
    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        if (find(i) == best_root) {
            out.push_back(uv[static_cast<size_t>(i)]);
        }
    }
    return out;
}

std::vector<Eigen::Vector2d> remove_feature_outliers(
    const std::vector<Eigen::Vector2d>& uv,
    const ObservationMaskOptions& options)
{
    if (static_cast<int>(uv.size()) <= std::max(3, options.outlier_k + 1)) {
        return uv;
    }
    std::vector<double> nn;
    std::vector<double> kth;
    nn.reserve(uv.size());
    kth.reserve(uv.size());
    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        auto dists = sorted_neighbor_distances(uv, i);
        if (dists.empty()) {
            continue;
        }
        nn.push_back(dists.front());
        const int kth_idx = std::min(static_cast<int>(dists.size()) - 1, std::max(0, options.outlier_k - 1));
        kth.push_back(dists[static_cast<size_t>(kth_idx)]);
    }
    const double median_nn = median(nn);
    if (!std::isfinite(median_nn) || median_nn <= 0.0) {
        return uv;
    }
    std::vector<Eigen::Vector2d> density;
    density.reserve(uv.size());
    int k = 0;
    for (const auto& point : uv) {
        if (k < static_cast<int>(kth.size()) && kth[static_cast<size_t>(k)] <= options.outlier_knn_scale * median_nn) {
            density.push_back(point);
        }
        ++k;
    }
    if (density.size() < 3) {
        density = uv;
    }
    return keep_largest_radius_component(density, options.component_radius_scale * median_nn);
}

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
cv::Mat zeros_mask(int width, int height)
{
    return cv::Mat::zeros(height, width, CV_8UC1);
}

std::vector<cv::Point2f> to_cv_points(const std::vector<Eigen::Vector2d>& uv)
{
    std::vector<cv::Point2f> out;
    out.reserve(uv.size());
    for (const auto& p : uv) {
        out.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }
    return out;
}

void fill_polygon(cv::Mat& mask, const std::vector<cv::Point2f>& poly)
{
    std::vector<cv::Point> rounded;
    rounded.reserve(poly.size());
    for (const auto& p : poly) {
        rounded.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    const std::vector<std::vector<cv::Point>> polys{rounded};
    cv::fillPoly(mask, polys, cv::Scalar(1));
}

cv::Mat convex_hull_mask(const std::vector<Eigen::Vector2d>& uv, int width, int height)
{
    cv::Mat mask = zeros_mask(width, height);
    auto points = to_cv_points(uv);
    if (points.size() < 3) {
        return mask;
    }
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    fill_polygon(mask, hull);
    return mask;
}

bool valid_triangle(const cv::Vec6f& tri, double d_nn, const ObservationMaskOptions& options)
{
    const Eigen::Vector2d p0(tri[0], tri[1]);
    const Eigen::Vector2d p1(tri[2], tri[3]);
    const Eigen::Vector2d p2(tri[4], tri[5]);
    const double l0 = (p1 - p0).norm();
    const double l1 = (p2 - p1).norm();
    const double l2 = (p0 - p2).norm();
    const double l_max = std::max({l0, l1, l2});
    const double area = 0.5 * std::abs((p1.x() - p0.x()) * (p2.y() - p1.y()) - (p1.y() - p0.y()) * (p2.x() - p1.x()));
    const double radius = (l0 * l1 * l2) / (4.0 * std::max(area, 1.0e-12));
    return l_max < options.edge_scale * d_nn && radius < options.radius_scale * d_nn;
}

bool triangle_inside_rect(const cv::Vec6f& tri, int width, int height)
{
    for (int i = 0; i < 3; ++i) {
        const float x = tri[2 * i];
        const float y = tri[2 * i + 1];
        if (x < 0.0f || y < 0.0f || x >= static_cast<float>(width) || y >= static_cast<float>(height)) {
            return false;
        }
    }
    return true;
}

cv::Mat delaunay_support_mask(
    const std::vector<Eigen::Vector2d>& uv,
    int width,
    int height,
    const ObservationMaskOptions& options,
    int& raw_count,
    int& valid_count)
{
    cv::Mat mask = zeros_mask(width, height);
    raw_count = 0;
    valid_count = 0;
    const double d_nn = median_nn_distance(uv);
    if (uv.size() < 3 || d_nn <= 0.0) {
        return mask;
    }
    cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
    for (const auto& p : uv) {
        if (p.x() >= 0.0 && p.y() >= 0.0 && p.x() < width && p.y() < height) {
            subdiv.insert(cv::Point2f(static_cast<float>(p.x()), static_cast<float>(p.y())));
        }
    }
    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    raw_count = static_cast<int>(triangles.size());
    std::vector<std::vector<cv::Point>> polys;
    polys.reserve(triangles.size());
    for (const auto& tri : triangles) {
        if (!triangle_inside_rect(tri, width, height) || !valid_triangle(tri, d_nn, options)) {
            continue;
        }
        polys.push_back({
            {cvRound(tri[0]), cvRound(tri[1])},
            {cvRound(tri[2]), cvRound(tri[3])},
            {cvRound(tri[4]), cvRound(tri[5])},
        });
    }
    valid_count = static_cast<int>(polys.size());
    if (!polys.empty()) {
        cv::fillPoly(mask, polys, cv::Scalar(1));
    }
    return mask;
}

void process_holes(
    const cv::Mat& hull_mask,
    const cv::Mat& supported_mask,
    const ObservationMaskOptions& options,
    cv::Mat& final_mask,
    cv::Mat& rejected_mask,
    int& detected,
    int& filled,
    int& rejected)
{
    cv::Mat candidate;
    cv::bitwise_and(hull_mask, 1 - supported_mask, candidate);
    final_mask = supported_mask.clone();
    rejected_mask = zeros_mask(hull_mask.cols, hull_mask.rows);
    detected = 0;
    filled = 0;
    rejected = 0;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int n_labels = cv::connectedComponentsWithStats(candidate, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < n_labels; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < options.min_hole_area) {
            continue;
        }
        ++detected;
        const cv::Mat hole = labels == label;
        if (area <= options.tiny_hole_fill_area) {
            final_mask.setTo(1, hole);
            ++filled;
        } else {
            rejected_mask.setTo(1, hole);
            ++rejected;
        }
    }
    cv::bitwise_and(final_mask, hull_mask, final_mask);
}

std::vector<unsigned char> mat_to_vector(const cv::Mat& mat)
{
    std::vector<unsigned char> out(static_cast<size_t>(mat.rows * mat.cols));
    for (int y = 0; y < mat.rows; ++y) {
        const auto* row = mat.ptr<unsigned char>(y);
        std::copy(row, row + mat.cols, out.begin() + static_cast<std::ptrdiff_t>(y * mat.cols));
    }
    return out;
}
#endif

std::vector<unsigned char> empty_data(int width, int height)
{
    return std::vector<unsigned char>(static_cast<size_t>(std::max(0, width) * std::max(0, height)), 0);
}

} // namespace

ObservationMaskResult build_observation_mask(
    int camera_index,
    int width,
    int height,
    const std::vector<Eigen::Vector2d>& observations,
    const ObservationMaskOptions& options)
{
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Observation mask dimensions must be positive.");
    }
    ObservationMaskResult result;
    result.camera_index = camera_index;
    result.width = width;
    result.height = height;

    for (const auto& p : observations) {
        if (std::isfinite(p.x()) && std::isfinite(p.y()) && p.x() >= 0.0 && p.y() >= 0.0 && p.x() < width && p.y() < height) {
            result.observations.push_back(p);
        }
    }
    result.clean_observations = remove_feature_outliers(result.observations, options);
    if (result.clean_observations.size() < 3) {
        result.mask = empty_data(width, height);
        result.hull_mask = result.mask;
        result.supported_mask = result.mask;
        result.rejected_hole_mask = result.mask;
        return result;
    }

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
    cv::Mat hull = convex_hull_mask(result.clean_observations, width, height);
    cv::Mat supported = delaunay_support_mask(
        result.clean_observations,
        width,
        height,
        options,
        result.n_triangles_raw,
        result.n_triangles_valid);
    cv::bitwise_and(supported, hull, supported);
    cv::Mat final_mask;
    cv::Mat rejected;
    process_holes(
        hull,
        supported,
        options,
        final_mask,
        rejected,
        result.n_holes_detected,
        result.n_holes_filled_as_speckle,
        result.n_holes_rejected);
    result.mask = mat_to_vector(final_mask);
    result.hull_mask = mat_to_vector(hull);
    result.supported_mask = mat_to_vector(supported);
    result.rejected_hole_mask = mat_to_vector(rejected);
#else
    throw std::runtime_error("Observation mask generation requires OpenCV support.");
#endif
    return result;
}

std::vector<ObservationMaskResult> build_observation_masks(
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& camera_indices,
    const std::vector<Eigen::Vector2d>& observation_uv,
    const ObservationMaskOptions& options)
{
    if (widths.size() != heights.size()) {
        throw std::invalid_argument("widths and heights must have the same length.");
    }
    if (camera_indices.size() != observation_uv.size()) {
        throw std::invalid_argument("camera_indices and observation_uv must have the same length.");
    }
    std::vector<std::vector<Eigen::Vector2d>> by_camera(widths.size());
    for (size_t i = 0; i < camera_indices.size(); ++i) {
        const int cam = camera_indices[i];
        if (cam >= 0 && cam < static_cast<int>(by_camera.size())) {
            by_camera[static_cast<size_t>(cam)].push_back(observation_uv[i]);
        }
    }
    std::vector<ObservationMaskResult> out;
    out.reserve(widths.size());
    for (int cam = 0; cam < static_cast<int>(widths.size()); ++cam) {
        out.push_back(build_observation_mask(cam, widths[static_cast<size_t>(cam)], heights[static_cast<size_t>(cam)], by_camera[static_cast<size_t>(cam)], options));
    }
    return out;
}

} // namespace dic
