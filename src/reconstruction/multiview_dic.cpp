/**
 * @file multiview_dic.cpp
 * @brief Multi-view 3D DIC implementation with outlier filtering.
 */

#include <dic/reconstruction/multiview_dic.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

namespace dic {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Median-of-absolute-deviations robust upper outlier detection.
/// Returns a mask of outliers (true = outlier).
std::vector<bool> robust_upper_outliers(
    const std::vector<double>& values,
    const std::vector<bool>& mask,
    double mad_z,
    double absolute_max)
{
    std::vector<bool> outliers(values.size(), false);
    if (static_cast<int>(mask.size()) < 3 || mad_z <= 0.0)
        return outliers;

    // Apply absolute threshold first
    if (absolute_max > 0.0) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (mask[i] && values[i] > absolute_max)
                outliers[i] = true;
        }
    }

    // Collect valid values for MAD computation
    std::vector<double> sample;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i] && std::isfinite(values[i]))
            sample.push_back(values[i]);
    }
    if (sample.size() < 3) return outliers;

    // Median
    std::nth_element(sample.begin(), sample.begin() + sample.size() / 2, sample.end());
    double median = sample[sample.size() / 2];
    if (sample.size() % 2 == 0) {
        std::nth_element(sample.begin(), sample.begin() + sample.size() / 2 - 1, sample.end());
        median = (median + sample[sample.size() / 2 - 1]) * 0.5;
    }

    // MAD
    std::vector<double> abs_devs;
    abs_devs.reserve(sample.size());
    for (double v : sample)
        abs_devs.push_back(std::abs(v - median));
    std::nth_element(abs_devs.begin(), abs_devs.begin() + abs_devs.size() / 2, abs_devs.end());
    double mad = abs_devs[abs_devs.size() / 2];
    if (abs_devs.size() % 2 == 0) {
        std::nth_element(abs_devs.begin(), abs_devs.begin() + abs_devs.size() / 2 - 1, abs_devs.end());
        mad = (mad + abs_devs[abs_devs.size() / 2 - 1]) * 0.5;
    }

    double sigma = 1.4826 * mad;
    if (sigma <= 1.0e-12 || !std::isfinite(sigma))
        return outliers;

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i] && std::isfinite(values[i]))
            if ((values[i] - median) / sigma > mad_z)
                outliers[i] = true;
    }
    return outliers;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MultiviewDIC::MultiviewDIC(const MultiviewDICOptions& opts) : opts_(opts) {}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

ShapeReconstructionResult MultiviewDIC::reconstruct(
    const std::vector<TrackObservations>& tracks,
    const std::vector<CameraModel>& cameras)
{
    // Convert TrackObservations to the format expected by ShapeReconstruction
    std::vector<std::vector<PointObservation>> grouped;
    grouped.reserve(tracks.size());

    for (const auto& track : tracks) {
        grouped.push_back(track.observations);
    }

    ShapeReconstructionOptions rec_opts;
    rec_opts.min_views = opts_.min_views;
    rec_opts.min_correlation = opts_.min_correlation;
    rec_opts.max_reprojection_error_px = opts_.max_reprojection_error_px;
    rec_opts.world_scale = opts_.world_scale;
    rec_opts.remove_rigid_body_motion = opts_.remove_rigid_body_motion;

    ShapeReconstruction reconstructor(rec_opts);
    ShapeReconstructionResult result = reconstructor.reconstruct(grouped, cameras);

    // Apply outlier filtering
    if (opts_.outlier_filter_enabled)
        filter_outliers(result);

    return result;
}

// ---------------------------------------------------------------------------
// Outlier filtering
// ---------------------------------------------------------------------------

int MultiviewDIC::filter_outliers(ShapeReconstructionResult& result)
{
    int n = static_cast<int>(result.points.size());
    if (n < opts_.outlier_filter_min_points) return 0;

    // Step 1: collect valid points for computing statistics
    std::vector<bool> valid(n, false);
    std::vector<Eigen::Vector3d> points(n);
    std::vector<Eigen::Vector3d> disp(n);
    for (int i = 0; i < n; ++i) {
        valid[i] = result.points[i].valid;
        points[i] = result.points[i].point_ref_world;
        disp[i] = result.points[i].displacement_world;
    }

    int initial_valid = 0;
    for (bool v : valid) if (v) ++initial_valid;
    if (initial_valid < opts_.outlier_filter_min_points) return 0;

    // Step 2: filter by position radius from centroid
    // Compute centroid (median)
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    {
        std::vector<double> xs, ys, zs;
        for (int i = 0; i < n; ++i) {
            if (valid[i]) {
                xs.push_back(points[i].x());
                ys.push_back(points[i].y());
                zs.push_back(points[i].z());
            }
        }
        auto med3 = [](std::vector<double>& v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            double m = v[v.size() / 2];
            if (v.size() % 2 == 0) {
                std::nth_element(v.begin(), v.begin() + v.size() / 2 - 1, v.end());
                m = (m + v[v.size() / 2 - 1]) * 0.5;
            }
            return m;
        };
        centroid = Eigen::Vector3d(med3(xs), med3(ys), med3(zs));
    }

    std::vector<double> pos_radius(n);
    std::vector<double> disp_norm(n);
    std::vector<bool> finite_mask(n, false);

    for (int i = 0; i < n; ++i) {
        if (valid[i] && points[i].allFinite() && disp[i].allFinite()) {
            finite_mask[i] = true;
            pos_radius[i] = (points[i] - centroid).norm();
            disp_norm[i] = disp[i].norm();
        }
    }

    // Position outliers
    auto pos_outliers = robust_upper_outliers(
        pos_radius, finite_mask,
        opts_.outlier_filter_position_mad_z,
        opts_.outlier_filter_max_position_radius_world);

    // Displacement outliers
    auto disp_outliers = robust_upper_outliers(
        disp_norm, finite_mask,
        opts_.outlier_filter_displacement_mad_z,
        opts_.outlier_filter_max_displacement_norm_world);

    int removed = 0;
    for (int i = 0; i < n; ++i) {
        if (result.points[i].valid && (pos_outliers[i] || disp_outliers[i] || !finite_mask[i])) {
            result.points[i].valid = false;
            ++removed;
        }
    }

    return removed;
}

} // namespace dic
