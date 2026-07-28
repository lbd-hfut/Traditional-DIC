/**
 * @file shape_reconstruction.hpp
 * @brief 3D shape reconstruction from multi-view 2D correspondences.
 *
 * Responsibilities:
 * - Reconstruct 3D reference and deformed points via DLT triangulation.
 * - Filter results by reprojection error, correlation coefficient, and view count.
 * - Compute 3D displacement from matched reference/deformed point pairs.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - dic/calibration/camera_model.hpp for camera parameters.
 * - dic/geometry/projection.hpp for projection/distortion utilities.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_SHAPE_RECONSTRUCTION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_SHAPE_RECONSTRUCTION_HPP

#include <dic/calibration/camera_model.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

namespace dic {

/// --- Configuration ---

struct ShapeReconstructionOptions {
    int min_views = 2;
    double min_correlation = 0.6;
    double max_reprojection_error_px = 2.0;
    double world_scale = 1.0;
    bool remove_rigid_body_motion = false;
};

/// --- 2D observation ---

struct PointObservation {
    int camera_index = 0;
    Eigen::Vector2d uv_ref;    // reference image coordinate
    Eigen::Vector2d uv_def;    // deformed image coordinate
    double u_displacement = 0.0;
    double v_displacement = 0.0;
    double correlation = 0.0;
    bool dic_valid = false;
};

/// --- Reconstructed 3D point pair ---

struct ReconstructedPoint {
    std::int64_t track_id = 0;
    Eigen::Vector3d point_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_def = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_ref_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_def_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement_world = Eigen::Vector3d::Zero();
    double displacement_norm_world = 0.0;
    int num_views = 0;
    double mean_correlation = 0.0;
    double reprojection_error_ref = 0.0;
    double reprojection_error_def = 0.0;
    bool valid = false;
};

/// --- Reconstruction result container ---

struct ShapeReconstructionResult {
    std::vector<ReconstructedPoint> points;
    double world_scale = 1.0;
    int total_tracks = 0;
    int valid_tracks = 0;
};

/// --- Main reconstruction class ---

class ShapeReconstruction {
public:
    ShapeReconstruction() = default;
    explicit ShapeReconstruction(const ShapeReconstructionOptions& opts);

    /// Reconstruct 3D points from grouped 2D observations.
    /// Each entry in `tracks` is a list of observations for one track (3D point).
    ShapeReconstructionResult reconstruct(
        const std::vector<std::vector<PointObservation>>& tracks,
        const std::vector<CameraModel>& cameras);

    /// Convenience: reconstruct a single stereo pair.
    bool reconstruct_pair(
        const PointObservation& obs_a,
        const PointObservation& obs_b,
        const CameraModel& cam_a,
        const CameraModel& cam_b,
        ReconstructedPoint& out);

    /// Build 3x4 projection matrix: K * [R | t].
    static Eigen::Matrix<double, 3, 4> build_projection(const CameraModel& camera);

    /// Remove rigid body motion: align deformed points to reference via SVD.
    static Eigen::Matrix3d rigid_body_rotation(
        const std::vector<Eigen::Vector3d>& from,
        const std::vector<Eigen::Vector3d>& to,
        const std::vector<bool>& valid_mask);

private:
    ShapeReconstructionOptions opts_;

    /// DLT multi-view triangulation of a single point from normalized rays.
    bool triangulate(
        const std::vector<Eigen::Vector2d>& normalized_rays,
        const std::vector<Eigen::Matrix<double, 3, 4>>& projection_matrices,
        Eigen::Vector3d& out);

    /// Project a 3D point into a camera's pixel coordinates.
    Eigen::Vector2d project(const Eigen::Vector3d& point, const CameraModel& camera);

    /// Convert pixel coordinate to normalized ray (undistort).
    Eigen::Vector2d pixel_to_normalized(const Eigen::Vector2d& px, const CameraModel& camera);

    /// Compute reprojection error for a single observation.
    double reprojection_error(
        const Eigen::Vector3d& point,
        const Eigen::Vector2d& observation,
        const CameraModel& camera);
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_SHAPE_RECONSTRUCTION_HPP
