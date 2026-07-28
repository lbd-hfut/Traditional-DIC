/**
 * @file filtering.cpp
 * @brief Outlier filtering implementation using MAD-based robust statistics.
 */

#include <dic/postprocess/filtering.hpp>

#include <algorithm>
#include <cmath>

namespace dic {

// ---------------------------------------------------------------------------
// Robust statistics
// ---------------------------------------------------------------------------

RobustStats compute_robust_stats(const std::vector<double>& values,
                                  const std::vector<bool>& mask)
{
    RobustStats stats;

    // Collect valid values
    std::vector<double> sample;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i] && std::isfinite(values[i]))
            sample.push_back(values[i]);
    }
    stats.count = static_cast<int>(sample.size());
    if (stats.count < 3) return stats;

    // Median
    std::nth_element(sample.begin(), sample.begin() + sample.size() / 2, sample.end());
    stats.median = sample[sample.size() / 2];
    if (sample.size() % 2 == 0) {
        std::nth_element(sample.begin(), sample.begin() + sample.size() / 2 - 1, sample.end());
        stats.median = (stats.median + sample[sample.size() / 2 - 1]) * 0.5;
    }

    // MAD
    std::vector<double> abs_devs;
    abs_devs.reserve(sample.size());
    for (double v : sample)
        abs_devs.push_back(std::abs(v - stats.median));
    std::nth_element(abs_devs.begin(), abs_devs.begin() + abs_devs.size() / 2, abs_devs.end());
    stats.mad = abs_devs[abs_devs.size() / 2];
    if (abs_devs.size() % 2 == 0) {
        std::nth_element(abs_devs.begin(), abs_devs.begin() + abs_devs.size() / 2 - 1, abs_devs.end());
        stats.mad = (stats.mad + abs_devs[abs_devs.size() / 2 - 1]) * 0.5;
    }

    stats.sigma = 1.4826 * stats.mad;
    return stats;
}

// ---------------------------------------------------------------------------
// MAD upper-tail outlier detection
// ---------------------------------------------------------------------------

std::vector<bool> mad_upper_outliers(
    const std::vector<double>& values,
    const std::vector<bool>& mask,
    double mad_z,
    double absolute_max)
{
    int n = static_cast<int>(values.size());
    std::vector<bool> outliers(n, false);

    // Absolute maximum threshold
    if (absolute_max > 0.0) {
        for (int i = 0; i < n; ++i)
            if (mask[i] && values[i] > absolute_max)
                outliers[i] = true;
    }

    if (mad_z <= 0.0) return outliers;

    RobustStats stats = compute_robust_stats(values, mask);
    if (stats.count < 3 || stats.sigma <= 1.0e-12 || !std::isfinite(stats.sigma))
        return outliers;

    for (int i = 0; i < n; ++i) {
        if (mask[i] && !outliers[i] && std::isfinite(values[i])) {
            double z = (values[i] - stats.median) / stats.sigma;
            if (z > mad_z)
                outliers[i] = true;
        }
    }
    return outliers;
}

// ---------------------------------------------------------------------------
// Combined 3D point + displacement filtering
// ---------------------------------------------------------------------------

OutlierFilterResult filter_displacements(
    std::vector<Eigen::Vector3d>& points,
    std::vector<Eigen::Vector3d>& displacements,
    std::vector<bool>& valid,
    const OutlierFilterOptions& options)
{
    OutlierFilterResult result;
    int n = static_cast<int>(points.size());
    if (n == 0) return result;

    result.input_count = 0;
    for (bool v : valid) if (v) ++result.input_count;
    if (result.input_count < options.min_points || !options.enabled) {
        result.keep = valid;
        result.output_count = result.input_count;
        for (std::size_t i = 0; i < valid.size(); ++i) result.reason.push_back(0);
        return result;
    }

    result.keep.resize(n, false);
    result.reason.resize(n, 0);
    std::vector<bool> finite_mask(n, false);

    // Step 1: check finiteness
    for (int i = 0; i < n; ++i) {
        if (valid[i] && points[i].allFinite() && displacements[i].allFinite()) {
            finite_mask[i] = true;
        } else if (valid[i]) {
            result.reason[i] |= 1;  // non-finite
        }
    }

    // Step 2: compute centroid (median) and position radius
    std::vector<double> xs, ys, zs;
    for (int i = 0; i < n; ++i) {
        if (finite_mask[i]) {
            xs.push_back(points[i].x());
            ys.push_back(points[i].y());
            zs.push_back(points[i].z());
        }
    }
    auto med = [](std::vector<double>& v) {
        if (v.empty()) return 0.0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        double m = v[v.size() / 2];
        if (v.size() % 2 == 0) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2 - 1, v.end());
            m = (m + v[v.size() / 2 - 1]) * 0.5;
        }
        return m;
    };
    Eigen::Vector3d centroid(med(xs), med(ys), med(zs));

    std::vector<double> pos_radius(n), disp_norm(n);
    for (int i = 0; i < n; ++i) {
        if (finite_mask[i]) {
            pos_radius[i] = (points[i] - centroid).norm();
            disp_norm[i] = displacements[i].norm();
        }
    }

    // Position outliers
    auto pos_out = mad_upper_outliers(pos_radius, finite_mask,
                                       options.position_mad_z,
                                       options.max_position_radius);
    // Displacement outliers
    auto disp_out = mad_upper_outliers(disp_norm, finite_mask,
                                        options.displacement_mad_z,
                                        options.max_displacement_norm);

    // Combine
    for (int i = 0; i < n; ++i) {
        bool keep = true;
        if (!finite_mask[i]) {
            keep = false;
        }
        if (pos_out[i]) {
            keep = false;
            result.reason[i] |= 2;
        }
        if (disp_out[i]) {
            keep = false;
            result.reason[i] |= 4;
        }
        result.keep[i] = keep;
    }

    // Apply to valid
    for (int i = 0; i < n; ++i)
        valid[i] = result.keep[i];

    result.output_count = 0;
    for (bool v : valid) if (v) ++result.output_count;
    result.removed = result.input_count - result.output_count;

    return result;
}

} // namespace dic
