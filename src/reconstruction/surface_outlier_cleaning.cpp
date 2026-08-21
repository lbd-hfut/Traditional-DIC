#include <dic/reconstruction/surface_outlier_cleaning.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/flann/miniflann.hpp>
#endif

namespace dic {
namespace {

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    double value = *middle;
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), middle);
        value = 0.5 * (value + *lower);
    }
    return value;
}

double robust_threshold(const std::vector<double>& values, double sigma, double relative_floor)
{
    const double center = median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) deviations.push_back(std::abs(value - center));
    const double mad = median(std::move(deviations));
    return center + sigma * std::max(std::max(1.4826 * mad, center * relative_floor), 1.0e-12);
}

bool finite(const Eigen::Vector3d& value)
{
    return value.array().isFinite().all();
}

#ifdef TRADITIONAL_DIC_HAS_OPENCV
class ScopedFlannRng
{
public:
    ScopedFlannRng()
        : lock_(mutex()), previous_(cv::theRNG())
    {
        // OpenCV FLANN randomizes KD-tree construction through the process
        // global RNG.  Keep this boundary repeatable without leaking a
        // changed RNG stream to the rest of the application.
        cv::theRNG() = cv::RNG(0xF0C0D1CEULL);
    }

    ~ScopedFlannRng()
    {
        cv::theRNG() = previous_;
    }

    ScopedFlannRng(const ScopedFlannRng&) = delete;
    ScopedFlannRng& operator=(const ScopedFlannRng&) = delete;

private:
    static std::mutex& mutex()
    {
        static std::mutex value;
        return value;
    }

    std::unique_lock<std::mutex> lock_;
    cv::RNG previous_;
};
#endif

} // namespace

SurfaceOutlierCleaningResult clean_surface_outliers(
    const std::vector<Eigen::Vector3d>& reference,
    const std::vector<Eigen::Vector3d>& deformed,
    const std::vector<std::array<std::int64_t, 3>>& faces,
    const std::vector<std::uint8_t>& initial_valid,
    const SurfaceOutlierCleaningOptions& options)
{
    if (reference.size() != deformed.size() || reference.size() != initial_valid.size()) {
        throw std::invalid_argument("reference, deformed, and initial_valid must have equal lengths");
    }
    if (options.neighbor_count < 2) throw std::invalid_argument("neighbor_count must be at least 2");

    SurfaceOutlierCleaningResult result;
    result.valid_points.resize(reference.size(), 0);
    result.valid_faces.resize(faces.size(), 0);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        result.valid_points[i] = initial_valid[i] && finite(reference[i]) && finite(deformed[i]);
    }

#ifndef TRADITIONAL_DIC_HAS_OPENCV
    throw std::runtime_error("surface outlier cleaning requires the OpenCV FLANN backend");
#else
    std::vector<int> point_ids;
    point_ids.reserve(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (result.valid_points[i]) point_ids.push_back(static_cast<int>(i));
    }
    const int k = std::min(options.neighbor_count, static_cast<int>(point_ids.size()) - 1);
    if (k >= 2) {
        ScopedFlannRng flann_rng;
        cv::Mat point_matrix(static_cast<int>(point_ids.size()), 3, CV_32F);
        for (int row = 0; row < point_matrix.rows; ++row) {
            const auto& point = reference[static_cast<std::size_t>(point_ids[row])];
            point_matrix.at<float>(row, 0) = static_cast<float>(point.x());
            point_matrix.at<float>(row, 1) = static_cast<float>(point.y());
            point_matrix.at<float>(row, 2) = static_cast<float>(point.z());
        }
        cv::flann::Index index(point_matrix, cv::flann::KDTreeIndexParams(4));
        cv::Mat query = point_matrix.clone();
        cv::Mat neighbor_indices;
        cv::Mat neighbor_distances;
        index.knnSearch(query, neighbor_indices, neighbor_distances, k + 1, cv::flann::SearchParams(256));

        std::vector<double> local_scales(point_ids.size(), 0.0);
        std::vector<std::vector<int>> neighbors(point_ids.size());
        for (int row = 0; row < point_matrix.rows; ++row) {
            neighbors[static_cast<std::size_t>(row)].reserve(static_cast<std::size_t>(k));
            local_scales[static_cast<std::size_t>(row)] = std::sqrt(
                std::max(0.0, static_cast<double>(neighbor_distances.at<float>(row, k))));
            for (int col = 1; col <= k; ++col) {
                neighbors[static_cast<std::size_t>(row)].push_back(neighbor_indices.at<int>(row, col));
            }
        }

        const double distance_limit = robust_threshold(local_scales, options.distance_sigma, 0.25);
        std::vector<double> displacement_residuals(point_ids.size(), 0.0);
        for (std::size_t row = 0; row < point_ids.size(); ++row) {
            std::vector<double> x_values;
            std::vector<double> y_values;
            std::vector<double> z_values;
            x_values.reserve(neighbors[row].size());
            y_values.reserve(neighbors[row].size());
            z_values.reserve(neighbors[row].size());
            for (int neighbor : neighbors[row]) {
                const std::size_t global = static_cast<std::size_t>(point_ids[neighbor]);
                const Eigen::Vector3d displacement = deformed[global] - reference[global];
                x_values.push_back(displacement.x());
                y_values.push_back(displacement.y());
                z_values.push_back(displacement.z());
            }
            const Eigen::Vector3d local_median(
                median(std::move(x_values)), median(std::move(y_values)), median(std::move(z_values)));
            const std::size_t global = static_cast<std::size_t>(point_ids[row]);
            displacement_residuals[row] =
                ((deformed[global] - reference[global]) - local_median).norm();
        }
        const double displacement_limit = robust_threshold(displacement_residuals, options.displacement_sigma, 0.5);
        for (std::size_t row = 0; row < point_ids.size(); ++row) {
            if (local_scales[row] > distance_limit || displacement_residuals[row] > displacement_limit) {
                result.valid_points[static_cast<std::size_t>(point_ids[row])] = 0;
            }
        }
    }

    std::vector<double> edge_lengths;
    edge_lengths.reserve(faces.size() * 3);
    for (const auto& face : faces) {
        bool valid = true;
        for (auto id : face) {
            if (id < 0 || id >= static_cast<std::int64_t>(reference.size()) ||
                !result.valid_points[static_cast<std::size_t>(id)]) {
                valid = false;
            }
        }
        if (!valid) continue;
        for (int edge = 0; edge < 3; ++edge) {
            const auto a = reference[static_cast<std::size_t>(face[edge])];
            const auto b = reference[static_cast<std::size_t>(face[(edge + 1) % 3])];
            const double length = (a - b).norm();
            if (std::isfinite(length) && length > 0.0) edge_lengths.push_back(length);
        }
    }
    const double typical_edge = median(edge_lengths);
    const double max_edge = options.face_edge_scale * std::max(typical_edge, 1.0e-12);
    std::vector<std::uint8_t> used_points(reference.size(), 0);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        bool valid = true;
        double largest_edge = 0.0;
        for (int vertex = 0; vertex < 3; ++vertex) {
            const auto id = face[vertex];
            if (id < 0 || id >= static_cast<std::int64_t>(reference.size()) ||
                !result.valid_points[static_cast<std::size_t>(id)]) {
                valid = false;
                continue;
            }
            const auto& a = reference[static_cast<std::size_t>(id)];
            const auto& b = reference[static_cast<std::size_t>(face[(vertex + 1) % 3])];
            largest_edge = std::max(largest_edge, (a - b).norm());
        }
        valid = valid && (largest_edge <= max_edge);
        result.valid_faces[i] = valid ? 1 : 0;
        if (valid) {
            for (auto id : face) used_points[static_cast<std::size_t>(id)] = 1;
        }
    }
    for (std::size_t i = 0; i < result.valid_points.size(); ++i) {
        if (result.valid_points[i] && !used_points[i]) result.valid_points[i] = 0;
    }
    for (std::size_t i = 0; i < faces.size(); ++i) {
        if (!result.valid_faces[i]) continue;
        for (auto id : faces[i]) {
            if (!result.valid_points[static_cast<std::size_t>(id)]) {
                result.valid_faces[i] = 0;
                break;
            }
        }
    }
    result.removed_points = reference.size() - static_cast<std::size_t>(std::count(result.valid_points.begin(), result.valid_points.end(), 1));
    result.removed_faces = faces.size() - static_cast<std::size_t>(std::count(result.valid_faces.begin(), result.valid_faces.end(), 1));
    return result;
#endif
}

} // namespace dic
