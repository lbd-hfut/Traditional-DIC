/**
 * @file multiview_dic.hpp
 * @brief Multi-view 3D DIC orchestration for N-camera reconstruction.
 *
 * Responsibilities:
 * - Aggregate per-camera 2D DIC results into track-based observations.
 * - Triangulate 3D reference and deformed points from N >= 2 views.
 * - Apply outlier filtering on position and displacement.
 * - Optionally remove rigid body motion via SVD alignment.
 *
 * Dependencies:
 * - dic/reconstruction/shape_reconstruction.hpp.
 * - dic/calibration/camera_model.hpp.
 * - Eigen.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_MULTIVIEW_DIC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_MULTIVIEW_DIC_HPP

#include <dic/reconstruction/shape_reconstruction.hpp>
#include <dic/calibration/camera_model.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace dic {

/// --- Multi-view DIC options ---

struct MultiviewDICOptions {
    int min_views = 2;
    double min_correlation = 0.6;
    double max_reprojection_error_px = 2.0;
    double world_scale = 1.0;
    bool remove_rigid_body_motion = false;

    // Outlier filter
    bool outlier_filter_enabled = true;
    int outlier_filter_min_points = 20;
    double outlier_filter_position_mad_z = 8.0;
    double outlier_filter_displacement_mad_z = 8.0;
    double outlier_filter_max_position_radius_world = 0.0;
    double outlier_filter_max_displacement_norm_world = 0.0;
};

/// --- Per-track observation collection ---

struct TrackObservations {
    std::int64_t track_id = 0;
    std::vector<PointObservation> observations;  // one per camera that sees this point
};

/// --- Multi-view DIC orchestrator ---

class MultiviewDIC {
public:
    MultiviewDIC() = default;
    explicit MultiviewDIC(const MultiviewDICOptions& opts);

    /// Run multi-view reconstruction from track-grouped 2D DIC observations.
    ShapeReconstructionResult reconstruct(
        const std::vector<TrackObservations>& tracks,
        const std::vector<CameraModel>& cameras);

    /// Apply MAD-based outlier filtering to reconstructed points.
    /// Modifies the `valid` flag in-place. Returns the number of points removed.
    int filter_outliers(ShapeReconstructionResult& result);

private:
    MultiviewDICOptions opts_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_MULTIVIEW_DIC_HPP
