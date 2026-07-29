/**
 * @file stereo_dic.cpp
 * @brief Stereo 3D DIC implementation.
 */

#include <dic/reconstruction/stereo_dic.hpp>
#include <dic/reconstruction/displacement_3d.hpp>
#include <dic/reconstruction/shape_reconstruction.hpp>

namespace dic {

StereoDIC::StereoDIC(const Options& opts) : opts_(opts) {}

StereoDICResult StereoDIC::reconstruct(
    const std::vector<PointObservation>& left_obs,
    const std::vector<PointObservation>& right_obs,
    const CameraModel& left_cam,
    const CameraModel& right_cam)
{
    StereoDICResult result;
    result.world_scale = opts_.world_scale;

    if (left_obs.size() != right_obs.size()) return result;
    result.total_points = static_cast<int>(left_obs.size());

    ShapeReconstructionOptions rec_opts;
    rec_opts.min_views = 2;
    rec_opts.min_correlation = opts_.min_correlation;
    rec_opts.max_reprojection_error_px = opts_.max_reprojection_error_px;
    rec_opts.world_scale = opts_.world_scale;
    rec_opts.remove_rigid_body_motion = opts_.remove_rigid_body_motion;
    ShapeReconstruction reconstructor(rec_opts);

    // Temporary store for RBM removal
    std::vector<ReconstructedPoint> rec_pts;
    rec_pts.reserve(left_obs.size());

    for (std::size_t i = 0; i < left_obs.size(); ++i) {
        ReconstructedPoint rp;
        bool ok = reconstructor.reconstruct_pair(
            left_obs[i], right_obs[i], left_cam, right_cam, rp);
        if (ok) {
            rec_pts.push_back(rp);
        } else {
            // Insert a sentinel; will be invalid
            ReconstructedPoint invalid;
            rec_pts.push_back(invalid);
        }
    }

    // RBM removal across all valid points
    if (opts_.remove_rigid_body_motion) {
        std::vector<Eigen::Vector3d> ref_all, def_all;
        std::vector<bool> valid_mask;
        for (const auto& rp : rec_pts) {
            ref_all.push_back(rp.point_ref);
            def_all.push_back(rp.point_def);
            valid_mask.push_back(rp.valid);
        }
        const RigidBodyTransform transform = find_rigid_body_transform(
            def_all, ref_all, valid_mask);

        if (transform.valid) {
            for (auto& rp : rec_pts) {
                if (rp.valid) {
                    rp.point_def = transform.rotation * rp.point_def + transform.translation;
                    rp.point_ref_world = rp.point_ref * opts_.world_scale;
                    rp.point_def_world = rp.point_def * opts_.world_scale;
                    rp.displacement = rp.point_def - rp.point_ref;
                    rp.displacement_world = rp.point_def_world - rp.point_ref_world;
                    rp.displacement_norm_world = rp.displacement_world.norm();
                }
            }
        }
    }

    // Convert to StereoPointResult
    for (std::size_t i = 0; i < rec_pts.size(); ++i) {
        StereoPointResult sp;
        const auto& rp = rec_pts[i];

        sp.point_ref       = rp.point_ref;
        sp.point_def       = rp.point_def;
        sp.point_ref_world  = rp.point_ref_world;
        sp.point_def_world  = rp.point_def_world;
        sp.displacement      = rp.displacement;
        sp.displacement_world = rp.displacement_world;
        sp.displacement_norm_world = rp.displacement_norm_world;
        sp.reprojection_error_ref = rp.reprojection_error_ref;
        sp.reprojection_error_def = rp.reprojection_error_def;
        sp.valid = rp.valid;

        if (i < left_obs.size()) {
            sp.uv_ref_left = left_obs[i].uv_ref;
            sp.correlation_left = left_obs[i].correlation;
        }
        if (i < right_obs.size()) {
            sp.uv_ref_right = right_obs[i].uv_ref;
            sp.correlation_right = right_obs[i].correlation;
        }
        sp.combined_correlation = std::min(sp.correlation_left, sp.correlation_right);

        result.points.push_back(sp);
    }

    // Compute summary stats
    result.valid_points = 0;
    double sum_norm = 0.0;
    for (const auto& pt : result.points) {
        if (pt.valid) {
            ++result.valid_points;
            sum_norm += pt.displacement_norm_world;
        }
    }
    if (result.valid_points > 0)
        result.mean_displacement_norm = sum_norm / static_cast<double>(result.valid_points);

    return result;
}

} // namespace dic
