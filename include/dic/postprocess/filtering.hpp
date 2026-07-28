/**
 * @file filtering.hpp
 * @brief Displacement and position outlier filtering for 2D and 3D DIC results.
 *
 * Responsibilities:
 * - Median Absolute Deviation (MAD) based robust outlier detection.
 * - Position-based outlier removal (points far from centroid).
 * - Displacement-based outlier removal (unphysically large displacements).
 * - Combined spatial + displacement filtering pipelines.
 *
 * Dependencies:
 * - Eigen for numerical types.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_FILTERING_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_FILTERING_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

/// --- Filtering configuration ---

struct OutlierFilterOptions {
    bool enabled = true;
    int min_points = 20;
    double position_mad_z = 8.0;
    double displacement_mad_z = 8.0;
    double max_position_radius = 0.0;       // 0 = disabled
    double max_displacement_norm = 0.0;     // 0 = disabled
};

/// --- Filter result ---

struct OutlierFilterResult {
    std::vector<bool> keep;          // final mask after filtering
    std::vector<std::uint8_t> reason;  // bitflags: 1=non-finite, 2=position, 4=displacement
    int input_count = 0;
    int output_count = 0;
    int removed = 0;
    double position_threshold = 0.0;
    double displacement_threshold = 0.0;
};

/// --- Core filter functions ---

/// Apply MAD-based outlier filtering to 3D points with displacements.
/// Modifies `valid` in-place; returns filter statistics.
OutlierFilterResult filter_displacements(
    std::vector<Eigen::Vector3d>& points,
    std::vector<Eigen::Vector3d>& displacements,
    std::vector<bool>& valid,
    const OutlierFilterOptions& options);

/// Detect outliers in a single 1D value array using MAD.
/// Returns mask where true = outlier.
std::vector<bool> mad_upper_outliers(
    const std::vector<double>& values,
    const std::vector<bool>& mask,
    double mad_z,
    double absolute_max);

/// Compute robust statistics (median, MAD, sigma) for a value array.
struct RobustStats {
    double median = 0.0;
    double mad = 0.0;
    double sigma = 0.0;  // 1.4826 * MAD
    int count = 0;
};

RobustStats compute_robust_stats(const std::vector<double>& values,
                                  const std::vector<bool>& mask);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_FILTERING_HPP
