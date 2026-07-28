/**
 * @file stereo_dic.hpp
 * @brief Stereo 3D DIC orchestration integrating 2D DIC with stereo triangulation.
 *
 * Responsibilities:
 * - Accept calibrated stereo camera pair and matched 2D DIC results.
 * - Triangulate 3D reference shape and 3D deformed shape.
 * - Compute 3D displacements with optional rigid-body-motion removal.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - dic/calibration/camera_model.hpp.
 * - dic/reconstruction/shape_reconstruction.hpp.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP

#include <dic/calibration/camera_model.hpp>
#include <dic/reconstruction/shape_reconstruction.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic {

/// --- Stereo reconstruction result ---

struct StereoPointResult {
    Eigen::Vector3d point_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_def = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_ref_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_def_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement_world = Eigen::Vector3d::Zero();
    double displacement_norm_world = 0.0;
    Eigen::Vector2d uv_ref_left;
    Eigen::Vector2d uv_ref_right;
    double correlation_left = 0.0;
    double correlation_right = 0.0;
    double combined_correlation = 0.0;
    double reprojection_error_ref = 0.0;
    double reprojection_error_def = 0.0;
    bool valid = false;
};

struct StereoDICResult {
    std::vector<StereoPointResult> points;
    double world_scale = 1.0;
    int total_points = 0;
    int valid_points = 0;
    double mean_displacement_norm = 0.0;
};

/// --- Stereo DIC orchestrator ---

class StereoDIC {
public:
    /// Configuration equivalent to ShapeReconstructionOptions plus stereo specifics.
    struct Options {
        double min_correlation = 0.6;
        double max_reprojection_error_px = 2.0;
        double world_scale = 1.0;
        bool remove_rigid_body_motion = false;
    };

    StereoDIC() = default;
    explicit StereoDIC(const Options& opts);

    /// Reconstruct 3D from matched stereo point pairs.
    /// `left_obs` and `right_obs` must have the same length: element i in
    /// each vector corresponds to the same physical point.
    StereoDICResult reconstruct(
        const std::vector<PointObservation>& left_obs,
        const std::vector<PointObservation>& right_obs,
        const CameraModel& left_cam,
        const CameraModel& right_cam);

private:
    Options opts_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP
