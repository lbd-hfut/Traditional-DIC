#include <dic/calibration/multiview_calibration.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace dic {
namespace {

void validate_scale_options(const MultiviewScaleOptions& options)
{
    if (options.board_rows <= 0 || options.board_cols <= 0) {
        throw std::invalid_argument("Scale board rows and cols must be positive.");
    }
    if (options.square_size <= 0.0) {
        throw std::invalid_argument("Scale square size must be positive.");
    }
    if (options.trim_fraction < 0.0 || options.trim_fraction >= 0.5) {
        throw std::invalid_argument("Scale trim_fraction must be in [0, 0.5).");
    }
}

Eigen::Vector3d triangulate_linear(const std::vector<CameraModel>& cameras,
                                   const std::vector<FeatureTrackObservation>& observations)
{
    if (observations.size() < 2) {
        throw std::invalid_argument("Triangulation requires at least two observations.");
    }
    Eigen::MatrixXd A(static_cast<int>(observations.size() * 2), 4);
    int row = 0;
    for (const auto& observation : observations) {
        if (observation.image_index < 0 || observation.image_index >= static_cast<int>(cameras.size())) {
            throw std::out_of_range("Scale observation camera index is out of range.");
        }
        const auto P = cameras[static_cast<size_t>(observation.image_index)].projection_matrix();
        const double u = observation.point.x();
        const double v = observation.point.y();
        A.row(row++) = u * P.row(2) - P.row(0);
        A.row(row++) = v * P.row(2) - P.row(1);
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
    if (std::abs(homogeneous.w()) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Triangulated homogeneous point has near-zero scale.");
    }
    return homogeneous.head<3>() / homogeneous.w();
}

double reprojection_error(const Eigen::Vector3d& point,
                          const CameraModel& camera,
                          const Eigen::Vector2d& observation)
{
    const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    const Eigen::Vector3d projected = camera.projection_matrix() * homogeneous;
    if (std::abs(projected.z()) < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::Vector2d uv(projected.x() / projected.z(), projected.y() / projected.z());
    return (uv - observation).norm();
}

double mean_reprojection_error(const Eigen::Vector3d& point,
                               const std::vector<CameraModel>& cameras,
                               const std::vector<FeatureTrackObservation>& observations)
{
    double sum = 0.0;
    for (const auto& observation : observations) {
        sum += reprojection_error(point, cameras[static_cast<size_t>(observation.image_index)], observation.point);
    }
    return sum / static_cast<double>(observations.size());
}

double mean_value(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

double std_value(const std::vector<double>& values, const double mean)
{
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

std::vector<double> trim_values(std::vector<double> values, const double trim_fraction)
{
    if (values.empty() || trim_fraction <= 0.0) {
        return values;
    }
    std::sort(values.begin(), values.end());
    const size_t trim = static_cast<size_t>(std::floor(trim_fraction * static_cast<double>(values.size())));
    if (2 * trim >= values.size()) {
        return values;
    }
    return {values.begin() + static_cast<std::ptrdiff_t>(trim), values.end() - static_cast<std::ptrdiff_t>(trim)};
}

#ifdef TRADITIONAL_DIC_HAS_OPENCV
struct ImageFeatures {
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct PairGeometry {
    int i = -1;
    int j = -1;
    std::vector<cv::DMatch> matches;
    std::vector<unsigned char> inlier_mask;
    cv::Mat R;
    cv::Mat t;
    int inliers = 0;
};

Eigen::Matrix3d cv_to_eigen3x3(const cv::Mat& mat)
{
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

Eigen::Vector3d cv_to_eigen3(const cv::Mat& mat)
{
    return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

CameraModel make_initial_camera(const std::string& label, const int width, const int height)
{
    CameraModel camera;
    const double focal = 1.2 * static_cast<double>(std::max(width, height));
    camera.K << focal, 0.0, static_cast<double>(width - 1) * 0.5, 0.0, focal,
        static_cast<double>(height - 1) * 0.5, 0.0, 0.0, 1.0;
    camera.distortion = {0.0, 0.0, 0.0, 0.0};
    camera.image_width = width;
    camera.image_height = height;
    camera.label = label;
    return camera;
}

cv::Point2f normalize_point(const cv::Point2f& point, const CameraModel& camera)
{
    return {static_cast<float>((point.x - camera.K(0, 2)) / camera.K(0, 0)),
            static_cast<float>((point.y - camera.K(1, 2)) / camera.K(1, 1))};
}

cv::Point2f project_point_cv(const Eigen::Vector3d& point, const CameraModel& camera)
{
    const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    const Eigen::Vector3d projected = camera.projection_matrix() * homogeneous;
    return {static_cast<float>(projected.x() / projected.z()), static_cast<float>(projected.y() / projected.z())};
}

std::vector<cv::DMatch> ratio_match(const cv::Mat& desc1, const cv::Mat& desc2, const double ratio)
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) {
        return good;
    }
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

PairGeometry estimate_pair_geometry(const int i,
                                    const int j,
                                    const std::vector<ImageFeatures>& features,
                                    const std::vector<CameraModel>& cameras,
                                    const MultiviewCalibrationOptions& options)
{
    PairGeometry geometry;
    geometry.i = i;
    geometry.j = j;
    geometry.matches = ratio_match(features[static_cast<size_t>(i)].descriptors,
                                   features[static_cast<size_t>(j)].descriptors,
                                   options.match_ratio);
    if (geometry.matches.size() < static_cast<size_t>(options.min_inlier_matches)) {
        return geometry;
    }

    std::vector<cv::Point2f> points_i;
    std::vector<cv::Point2f> points_j;
    points_i.reserve(geometry.matches.size());
    points_j.reserve(geometry.matches.size());
    for (const auto& match : geometry.matches) {
        const auto pi = features[static_cast<size_t>(i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const auto pj = features[static_cast<size_t>(j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        points_i.push_back(normalize_point(pi, cameras[static_cast<size_t>(i)]));
        points_j.push_back(normalize_point(pj, cameras[static_cast<size_t>(j)]));
    }

    cv::Mat inlier_mask;
    const double threshold = options.ransac_reprojection_threshold /
                             std::max(cameras[static_cast<size_t>(i)].K(0, 0), 1.0);
    cv::Mat E = cv::findEssentialMat(points_i, points_j, 1.0, {0.0, 0.0}, cv::RANSAC, 0.999, threshold, inlier_mask);
    if (E.empty()) {
        return geometry;
    }
    cv::Mat R;
    cv::Mat t;
    const int inliers = cv::recoverPose(E, points_i, points_j, R, t, 1.0, {0.0, 0.0}, inlier_mask);
    geometry.R = R;
    geometry.t = t;
    geometry.inliers = inliers;
    geometry.inlier_mask.assign(inlier_mask.begin<unsigned char>(), inlier_mask.end<unsigned char>());
    return geometry;
}

cv::Mat make_normalized_projection(const CameraModel& camera)
{
    cv::Mat P(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            P.at<double>(r, c) = camera.R(r, c);
        }
        P.at<double>(r, 3) = camera.t(r);
    }
    return P;
}

void triangulate_pair(const PairGeometry& pair,
                      const std::vector<ImageFeatures>& features,
                      const std::vector<CameraModel>& cameras,
                      MultiviewCalibrationResult& result)
{
    const CameraModel& cam_i = cameras[static_cast<size_t>(pair.i)];
    const CameraModel& cam_j = cameras[static_cast<size_t>(pair.j)];
    const cv::Mat P1 = make_normalized_projection(cam_i);
    const cv::Mat P2 = make_normalized_projection(cam_j);

    std::vector<cv::Point2f> points_i;
    std::vector<cv::Point2f> points_j;
    std::vector<cv::DMatch> inlier_matches;
    for (size_t k = 0; k < pair.matches.size(); ++k) {
        if (k >= pair.inlier_mask.size() || pair.inlier_mask[k] == 0) {
            continue;
        }
        const auto& match = pair.matches[k];
        const auto pi = features[static_cast<size_t>(pair.i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const auto pj = features[static_cast<size_t>(pair.j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        points_i.push_back(normalize_point(pi, cam_i));
        points_j.push_back(normalize_point(pj, cam_j));
        inlier_matches.push_back(match);
    }
    if (points_i.empty()) {
        return;
    }

    cv::Mat homogeneous;
    cv::triangulatePoints(P1, P2, points_i, points_j, homogeneous);
    const auto homogeneous_at = [&homogeneous](const int r, const int c) {
        return homogeneous.depth() == CV_64F ? homogeneous.at<double>(r, c)
                                             : static_cast<double>(homogeneous.at<float>(r, c));
    };
    for (int c = 0; c < homogeneous.cols; ++c) {
        const double w = homogeneous_at(3, c);
        if (std::abs(w) < std::numeric_limits<double>::epsilon()) {
            continue;
        }
        Eigen::Vector3d point(homogeneous_at(0, c) / w, homogeneous_at(1, c) / w, homogeneous_at(2, c) / w);
        const Eigen::Vector3d xi = cam_i.R * point + cam_i.t;
        const Eigen::Vector3d xj = cam_j.R * point + cam_j.t;
        if (xi.z() <= 0.0 || xj.z() <= 0.0) {
            continue;
        }

        const auto& match = inlier_matches[static_cast<size_t>(c)];
        const cv::Point2f raw_i = features[static_cast<size_t>(pair.i)].keypoints[static_cast<size_t>(match.queryIdx)].pt;
        const cv::Point2f raw_j = features[static_cast<size_t>(pair.j)].keypoints[static_cast<size_t>(match.trainIdx)].pt;
        const cv::Point2f proj_i = project_point_cv(point, cam_i);
        const cv::Point2f proj_j = project_point_cv(point, cam_j);
        const cv::Point2f di = proj_i - raw_i;
        const cv::Point2f dj = proj_j - raw_j;
        const double error =
            0.5 * (std::sqrt(di.x * di.x + di.y * di.y) + std::sqrt(dj.x * dj.x + dj.y * dj.y));

        SparsePoint3D sparse;
        sparse.point = point;
        sparse.reprojection_error = error;
        sparse.observations.push_back({pair.i, {raw_i.x, raw_i.y}});
        sparse.observations.push_back({pair.j, {raw_j.x, raw_j.y}});
        result.sparse_points.push_back(std::move(sparse));
    }
}
#endif

} // namespace

MultiviewCalibrationResult calibrate_multiview_colmap_like(const std::vector<std::string>& image_paths,
                                                           const MultiviewCalibrationOptions& options)
{
    if (image_paths.size() < 2) {
        throw std::invalid_argument("Multiview calibration requires at least two images.");
    }
#ifndef TRADITIONAL_DIC_HAS_OPENCV
    (void)options;
    throw std::runtime_error("OpenCV is required for the simplified COLMAP-style multiview calibration.");
#else
    std::vector<ImageFeatures> features(image_paths.size());
    std::vector<CameraModel> cameras = options.initial_cameras;
    cameras.resize(image_paths.size());

    cv::Ptr<cv::ORB> orb = cv::ORB::create(options.max_features);
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to read SfM image: " + image_paths[i]);
        }
        features[i].image = image;
        orb->detectAndCompute(image, cv::noArray(), features[i].keypoints, features[i].descriptors);
        if (cameras[i].image_width == 0 || cameras[i].image_height == 0) {
            cameras[i] = make_initial_camera(image_paths[i], image.cols, image.rows);
        } else if (cameras[i].label.empty()) {
            cameras[i].label = image_paths[i];
        }
    }

    MultiviewCalibrationResult result;
    result.inlier_match_counts.assign(image_paths.size(), std::vector<int>(image_paths.size(), 0));

    std::vector<PairGeometry> pairs;
    pairs.reserve(image_paths.size() * image_paths.size());
    for (int i = 0; i < static_cast<int>(image_paths.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(image_paths.size()); ++j) {
            PairGeometry pair = estimate_pair_geometry(i, j, features, cameras, options);
            result.inlier_match_counts[static_cast<size_t>(i)][static_cast<size_t>(j)] = pair.inliers;
            result.inlier_match_counts[static_cast<size_t>(j)][static_cast<size_t>(i)] = pair.inliers;
            if (pair.inliers >= options.min_inlier_matches) {
                pairs.push_back(std::move(pair));
            }
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error("No robust image pair was found for multiview calibration.");
    }

    std::vector<bool> registered(image_paths.size(), false);
    registered[0] = true;
    cameras[0].R = Eigen::Matrix3d::Identity();
    cameras[0].t = Eigen::Vector3d::Zero();

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& pair : pairs) {
            int known = -1;
            int unknown = -1;
            bool forward = true;
            if (registered[static_cast<size_t>(pair.i)] && !registered[static_cast<size_t>(pair.j)]) {
                known = pair.i;
                unknown = pair.j;
                forward = true;
            } else if (registered[static_cast<size_t>(pair.j)] && !registered[static_cast<size_t>(pair.i)]) {
                known = pair.j;
                unknown = pair.i;
                forward = false;
            }
            if (known < 0) {
                continue;
            }

            const Eigen::Matrix3d R_rel = cv_to_eigen3x3(pair.R);
            const Eigen::Vector3d t_rel = cv_to_eigen3(pair.t);
            if (forward) {
                cameras[static_cast<size_t>(unknown)].R = R_rel * cameras[static_cast<size_t>(known)].R;
                cameras[static_cast<size_t>(unknown)].t = R_rel * cameras[static_cast<size_t>(known)].t + t_rel;
            } else {
                cameras[static_cast<size_t>(unknown)].R = R_rel.transpose() * cameras[static_cast<size_t>(known)].R;
                cameras[static_cast<size_t>(unknown)].t =
                    R_rel.transpose() * (cameras[static_cast<size_t>(known)].t - t_rel);
            }
            registered[static_cast<size_t>(unknown)] = true;
            changed = true;
        }
    }

    for (const auto& pair : pairs) {
        if (registered[static_cast<size_t>(pair.i)] && registered[static_cast<size_t>(pair.j)]) {
            triangulate_pair(pair, features, cameras, result);
        }
    }

    result.cameras.reserve(cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (registered[i]) {
            result.cameras.push_back(cameras[i]);
        }
    }
    if (result.cameras.size() < 2) {
        throw std::runtime_error("Multiview calibration could not register a connected camera model.");
    }

    double error_sum = 0.0;
    for (const auto& point : result.sparse_points) {
        error_sum += point.reprojection_error;
    }
    if (!result.sparse_points.empty()) {
        result.mean_reprojection_error = error_sum / static_cast<double>(result.sparse_points.size());
    }
    return result;
#endif
}

MultiviewScaleResult estimate_multiview_chessboard_scale(
    const std::vector<CameraModel>& cameras,
    const std::vector<SparsePoint3D>& sparse_points,
    const std::vector<MultiviewScaleObservation>& observations,
    const MultiviewScaleOptions& options)
{
    validate_scale_options(options);
    if (cameras.size() < 2) {
        throw std::invalid_argument("Scale estimation requires at least two cameras.");
    }
    const int corner_count = options.board_rows * options.board_cols;
    std::vector<std::vector<FeatureTrackObservation>> tracks(static_cast<size_t>(corner_count));
    for (const auto& observation_set : observations) {
        if (observation_set.camera_index < 0 || observation_set.camera_index >= static_cast<int>(cameras.size())) {
            throw std::out_of_range("Scale observation camera index is out of range.");
        }
        if (static_cast<int>(observation_set.image_points.size()) != corner_count) {
            throw std::invalid_argument("Every scale observation must contain board_rows * board_cols image points.");
        }
        for (int i = 0; i < corner_count; ++i) {
            tracks[static_cast<size_t>(i)].push_back({observation_set.camera_index, observation_set.image_points[static_cast<size_t>(i)]});
        }
    }

    std::vector<Eigen::Vector3d> board_points(static_cast<size_t>(corner_count), Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()));
    std::vector<bool> valid(static_cast<size_t>(corner_count), false);
    int triangulated = 0;
    for (int i = 0; i < corner_count; ++i) {
        const auto& track = tracks[static_cast<size_t>(i)];
        if (static_cast<int>(track.size()) < std::max(2, options.min_common_corners > corner_count ? 2 : 2)) {
            continue;
        }
        try {
            Eigen::Vector3d point = triangulate_linear(cameras, track);
            if (mean_reprojection_error(point, cameras, track) <= options.max_reprojection_error) {
                board_points[static_cast<size_t>(i)] = point;
                valid[static_cast<size_t>(i)] = true;
                ++triangulated;
            }
        } catch (const std::exception&) {
        }
    }

    std::vector<double> edges;
    for (int r = 0; r < options.board_rows; ++r) {
        for (int c = 0; c < options.board_cols; ++c) {
            const int idx = r * options.board_cols + c;
            if (c + 1 < options.board_cols) {
                const int right = r * options.board_cols + c + 1;
                if (valid[static_cast<size_t>(idx)] && valid[static_cast<size_t>(right)]) {
                    edges.push_back((board_points[static_cast<size_t>(idx)] - board_points[static_cast<size_t>(right)]).norm());
                }
            }
            if (r + 1 < options.board_rows) {
                const int down = (r + 1) * options.board_cols + c;
                if (valid[static_cast<size_t>(idx)] && valid[static_cast<size_t>(down)]) {
                    edges.push_back((board_points[static_cast<size_t>(idx)] - board_points[static_cast<size_t>(down)]).norm());
                }
            }
        }
    }
    if (static_cast<int>(edges.size()) < options.min_common_corners) {
        throw std::runtime_error("Not enough valid chessboard edges for scale estimation.");
    }

    std::vector<double> trimmed = trim_values(edges, options.trim_fraction);
    const double sfm_square_mean = mean_value(trimmed);
    if (sfm_square_mean <= 0.0) {
        throw std::runtime_error("Estimated SfM square size is non-positive.");
    }
    const double sfm_square_median = median_value(trimmed);
    const double sfm_square_std = std_value(trimmed, sfm_square_mean);
    const double scale = options.square_size / sfm_square_mean;

    MultiviewScaleResult result;
    result.sfm_to_world_scale = scale;
    result.world_to_sfm_scale = 1.0 / scale;
    result.sfm_square_size_mean = sfm_square_mean;
    result.sfm_square_size_median = sfm_square_median;
    result.sfm_square_size_std = sfm_square_std;
    result.edge_cv = sfm_square_mean > 0.0 ? sfm_square_std / sfm_square_mean : 0.0;
    result.triangulated_corners = triangulated;
    result.valid_edges = static_cast<int>(edges.size());
    result.triangulated_board_points_sfm = std::move(board_points);
    result.edge_lengths_sfm = std::move(edges);

    result.scaled_cameras = cameras;
    for (auto& camera : result.scaled_cameras) {
        camera.t *= scale;
    }
    result.scaled_sparse_points = sparse_points;
    for (auto& point : result.scaled_sparse_points) {
        point.point *= scale;
    }
    return result;
}

std::vector<CameraModel> calibrate_multiview(int image_count)
{
    (void)image_count;
    throw std::runtime_error("Use calibrate_multiview_colmap_like with image paths.");
}

} // namespace dic
