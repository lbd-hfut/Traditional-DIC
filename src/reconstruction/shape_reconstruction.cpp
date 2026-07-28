/**
 * @file shape_reconstruction.cpp
 * @brief 3D shape reconstruction implementation via DLT triangulation.
 */

#include <dic/reconstruction/shape_reconstruction.hpp>
#include <dic/geometry/projection.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace dic {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

ShapeReconstruction::ShapeReconstruction(const ShapeReconstructionOptions& opts)
    : opts_(opts) {}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

namespace {

double sqr(double v) { return v * v; }

Eigen::Matrix<double, 4, 4> outer_product(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    Eigen::Matrix<double, 4, 4> m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m(r, c) = a(r) * b(c);
    return m;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// DLT triangulation via eigen-decomposition of AtA
// ---------------------------------------------------------------------------

bool ShapeReconstruction::triangulate(
    const std::vector<Eigen::Vector2d>& rays,
    const std::vector<Eigen::Matrix<double, 3, 4>>& P,
    Eigen::Vector3d& out)
{
    if (rays.size() < 2) return false;

    Eigen::Matrix4d AtA = Eigen::Matrix4d::Zero();

    for (std::size_t i = 0; i < rays.size(); ++i) {
        const double x = rays[i].x();
        const double y = rays[i].y();

        Eigen::Vector4d r1(
            x * P[i](2, 0) - P[i](0, 0),
            x * P[i](2, 1) - P[i](0, 1),
            x * P[i](2, 2) - P[i](0, 2),
            x * P[i](2, 3) - P[i](0, 3));

        Eigen::Vector4d r2(
            y * P[i](2, 0) - P[i](1, 0),
            y * P[i](2, 1) - P[i](1, 1),
            y * P[i](2, 2) - P[i](1, 2),
            y * P[i](2, 3) - P[i](1, 3));

        AtA += outer_product(r1, r1);
        AtA += outer_product(r2, r2);
    }

    // Find smallest eigenvector of AtA (symmetric real => use SelfAdjointEigenSolver)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(AtA);
    if (eigensolver.info() != Eigen::Success) return false;

    const Eigen::Vector4d& homog = eigensolver.eigenvectors().col(0);
    if (std::abs(homog(3)) <= 1.0e-12) return false;

    out = Eigen::Vector3d(homog(0) / homog(3), homog(1) / homog(3), homog(2) / homog(3));
    return out.allFinite();
}

// ---------------------------------------------------------------------------
// Build projection matrix P = K * [R | t]
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 3, 4> ShapeReconstruction::build_projection(const CameraModel& camera) {
    Eigen::Matrix<double, 3, 4> P;
    P << camera.R, camera.t;
    P = camera.K * P;
    return P;
}

// ---------------------------------------------------------------------------
// Pixel to normalized ray (undistort)
// ---------------------------------------------------------------------------

Eigen::Vector2d ShapeReconstruction::pixel_to_normalized(
    const Eigen::Vector2d& px,
    const CameraModel& camera)
{
    double x = (px.x() - camera.K(0, 2)) / camera.K(0, 0);
    double y = (px.y() - camera.K(1, 2)) / camera.K(1, 1);

    double k1 = camera.distortion.size() > 0 ? camera.distortion[0] : 0.0;
    double k2 = camera.distortion.size() > 1 ? camera.distortion[1] : 0.0;

    // Iterative undistortion
    if (std::abs(k1) > 1.0e-12 || std::abs(k2) > 1.0e-12) {
        double xu = x, yu = y;
        for (int iter = 0; iter < 8; ++iter) {
            double r2 = xu * xu + yu * yu;
            double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            if (std::abs(radial) <= 1.0e-12) break;
            xu = x / radial;
            yu = y / radial;
        }
        x = xu; y = yu;
    }

    return Eigen::Vector2d(x, y);
}

// ---------------------------------------------------------------------------
// Project 3D point to pixel
// ---------------------------------------------------------------------------

Eigen::Vector2d ShapeReconstruction::project(
    const Eigen::Vector3d& point,
    const CameraModel& camera)
{
    // Camera frame
    Eigen::Vector3d cam = camera.R * point + camera.t;
    if (std::abs(cam.z()) <= 1.0e-12)
        return Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN());

    double xn = cam.x() / cam.z();
    double yn = cam.y() / cam.z();

    // Distort
    double k1 = camera.distortion.size() > 0 ? camera.distortion[0] : 0.0;
    double k2 = camera.distortion.size() > 1 ? camera.distortion[1] : 0.0;
    if (std::abs(k1) > 1.0e-12 || std::abs(k2) > 1.0e-12) {
        double r2 = xn * xn + yn * yn;
        double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        xn *= radial;
        yn *= radial;
    }

    return Eigen::Vector2d(
        camera.K(0, 0) * xn + camera.K(0, 2),
        camera.K(1, 1) * yn + camera.K(1, 2));
}

// ---------------------------------------------------------------------------
// Reprojection error for a single observation
// ---------------------------------------------------------------------------

double ShapeReconstruction::reprojection_error(
    const Eigen::Vector3d& point,
    const Eigen::Vector2d& observation,
    const CameraModel& camera)
{
    Eigen::Vector2d proj = project(point, camera);
    return std::sqrt(sqr(proj.x() - observation.x()) + sqr(proj.y() - observation.y()));
}

// ---------------------------------------------------------------------------
// Rigid body motion removal: SVD-based rotation between two point sets
// ---------------------------------------------------------------------------

Eigen::Matrix3d ShapeReconstruction::rigid_body_rotation(
    const std::vector<Eigen::Vector3d>& from,
    const std::vector<Eigen::Vector3d>& to,
    const std::vector<bool>& valid_mask)
{
    // Count valid points and compute centroids
    Eigen::Vector3d centroid_from = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid_to   = Eigen::Vector3d::Zero();
    int n = 0;
    for (std::size_t i = 0; i < from.size(); ++i) {
        if (valid_mask[i] && from[i].allFinite() && to[i].allFinite()) {
            centroid_from += from[i];
            centroid_to   += to[i];
            ++n;
        }
    }
    if (n < 3) return Eigen::Matrix3d::Identity();

    centroid_from /= static_cast<double>(n);
    centroid_to   /= static_cast<double>(n);

    // Build cross-covariance H = sum(d_from * d_to^T)
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < from.size(); ++i) {
        if (valid_mask[i] && from[i].allFinite() && to[i].allFinite()) {
            Eigen::Vector3d da = from[i] - centroid_from;
            Eigen::Vector3d db = to[i]   - centroid_to;
            H += da * db.transpose();
        }
    }

    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();

    // Ensure proper rotation (det = +1)
    if (R.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }
    return R;
}

// ---------------------------------------------------------------------------
// Single stereo pair reconstruction
// ---------------------------------------------------------------------------

bool ShapeReconstruction::reconstruct_pair(
    const PointObservation& obs_a,
    const PointObservation& obs_b,
    const CameraModel& cam_a,
    const CameraModel& cam_b,
    ReconstructedPoint& out)
{
    if (!obs_a.dic_valid || !obs_b.dic_valid) return false;

    Eigen::Vector2d ref_a(obs_a.uv_ref.x(), obs_a.uv_ref.y());
    Eigen::Vector2d ref_b(obs_b.uv_ref.x(), obs_b.uv_ref.y());
    Eigen::Vector2d def_a(obs_a.uv_ref.x() + obs_a.u_displacement,
                           obs_a.uv_ref.y() + obs_a.v_displacement);
    Eigen::Vector2d def_b(obs_b.uv_ref.x() + obs_b.u_displacement,
                           obs_b.uv_ref.y() + obs_b.v_displacement);

    auto P_a = build_projection(cam_a);
    auto P_b = build_projection(cam_b);

    std::vector<Eigen::Vector2d> rays_ref = {
        pixel_to_normalized(ref_a, cam_a),
        pixel_to_normalized(ref_b, cam_b)};
    std::vector<Eigen::Vector2d> rays_def = {
        pixel_to_normalized(def_a, cam_a),
        pixel_to_normalized(def_b, cam_b)};
    std::vector<Eigen::Matrix<double, 3, 4>> projs = {P_a, P_b};

    Eigen::Vector3d Xr, Xd;
    if (!triangulate(rays_ref, projs, Xr) || !triangulate(rays_def, projs, Xd))
        return false;

    double err_ref = 0.5 * (reprojection_error(Xr, ref_a, cam_a) +
                            reprojection_error(Xr, ref_b, cam_b));
    double err_def = 0.5 * (reprojection_error(Xd, def_a, cam_a) +
                            reprojection_error(Xd, def_b, cam_b));

    double corr_comb = std::min(obs_a.correlation, obs_b.correlation);

    out.point_ref   = Xr;
    out.point_def   = Xd;
    out.point_ref_world  = Xr * opts_.world_scale;
    out.point_def_world  = Xd * opts_.world_scale;
    out.displacement      = Xd - Xr;
    out.displacement_world = out.displacement * opts_.world_scale;
    out.displacement_norm_world = out.displacement_world.norm();
    out.num_views    = 2;
    out.mean_correlation = corr_comb;
    out.reprojection_error_ref = err_ref;
    out.reprojection_error_def = err_def;
    out.valid = (corr_comb >= opts_.min_correlation) &&
                (err_ref <= opts_.max_reprojection_error_px) &&
                (err_def <= opts_.max_reprojection_error_px);
    return true;
}

// ---------------------------------------------------------------------------
// Multi-view reconstruction from grouped tracks
// ---------------------------------------------------------------------------

ShapeReconstructionResult ShapeReconstruction::reconstruct(
    const std::vector<std::vector<PointObservation>>& tracks,
    const std::vector<CameraModel>& cameras)
{
    ShapeReconstructionResult result;
    result.world_scale = opts_.world_scale;
    result.total_tracks = static_cast<int>(tracks.size());

    // Precompute projection matrices
    std::vector<Eigen::Matrix<double, 3, 4>> proj_mats(cameras.size());
    for (std::size_t i = 0; i < cameras.size(); ++i)
        proj_mats[i] = build_projection(cameras[i]);

    // Temporary storage for RBM removal
    std::vector<Eigen::Vector3d> points_def_raw;
    std::vector<bool> rbm_valid;

    for (std::size_t t = 0; t < tracks.size(); ++t) {
        const auto& obs_list = tracks[t];
        if (static_cast<int>(obs_list.size()) < opts_.min_views) continue;

        ReconstructedPoint pt;
        pt.track_id = static_cast<std::int64_t>(t);

        std::vector<Eigen::Vector2d> rays_ref, rays_def;
        std::vector<Eigen::Matrix<double, 3, 4>> projs;
        std::vector<const PointObservation*> selected_obs;
        double corr_sum = 0.0;

        for (const auto& obs : obs_list) {
            if (!obs.dic_valid) continue;
            if (obs.correlation < opts_.min_correlation) continue;
            if (obs.camera_index < 0 ||
                static_cast<std::size_t>(obs.camera_index) >= cameras.size())
                continue;

            Eigen::Vector2d uv_ref(obs.uv_ref.x(), obs.uv_ref.y());
            Eigen::Vector2d uv_def(
                obs.uv_ref.x() + obs.u_displacement,
                obs.uv_ref.y() + obs.v_displacement);

            const auto& cam = cameras[static_cast<std::size_t>(obs.camera_index)];
            rays_ref.push_back(pixel_to_normalized(uv_ref, cam));
            rays_def.push_back(pixel_to_normalized(uv_def, cam));
            projs.push_back(proj_mats[static_cast<std::size_t>(obs.camera_index)]);
            selected_obs.push_back(&obs);
            corr_sum += obs.correlation;
        }

        if (static_cast<int>(selected_obs.size()) < opts_.min_views) continue;

        Eigen::Vector3d Xr, Xd;
        if (!triangulate(rays_ref, projs, Xr) || !triangulate(rays_def, projs, Xd))
            continue;

        // Compute mean reprojection error
        double err_ref = 0.0, err_def = 0.0;
        for (std::size_t i = 0; i < selected_obs.size(); ++i) {
            Eigen::Vector2d uvr(selected_obs[i]->uv_ref.x(), selected_obs[i]->uv_ref.y());
            Eigen::Vector2d uvd(
                selected_obs[i]->uv_ref.x() + selected_obs[i]->u_displacement,
                selected_obs[i]->uv_ref.y() + selected_obs[i]->v_displacement);
            const auto& cam = cameras[static_cast<std::size_t>(selected_obs[i]->camera_index)];
            err_ref += reprojection_error(Xr, uvr, cam);
            err_def += reprojection_error(Xd, uvd, cam);
        }
        ptrdiff_t n_views = static_cast<ptrdiff_t>(selected_obs.size());
        err_ref /= static_cast<double>(n_views);
        err_def /= static_cast<double>(n_views);

        pt.point_ref   = Xr;
        pt.point_def   = Xd;
        pt.num_views   = static_cast<int>(n_views);
        pt.mean_correlation = corr_sum / static_cast<double>(n_views);
        pt.reprojection_error_ref = err_ref;
        pt.reprojection_error_def = err_def;
        pt.valid = (err_ref <= opts_.max_reprojection_error_px) &&
                   (err_def <= opts_.max_reprojection_error_px);

        points_def_raw.push_back(Xd);
        rbm_valid.push_back(pt.valid);

        result.points.push_back(pt);
    }

    // Apply RBM removal if requested
    if (opts_.remove_rigid_body_motion) {
        std::vector<Eigen::Vector3d> ref_vec;
        for (const auto& pt : result.points)
            ref_vec.push_back(pt.point_ref);

        Eigen::Matrix3d R_rbm = rigid_body_rotation(points_def_raw, ref_vec, rbm_valid);
        // Recompute displacements with RBM removed
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            Eigen::Vector3d def_arbm = R_rbm * result.points[i].point_def;
            result.points[i].point_def = def_arbm;
            result.points[i].displacement = def_arbm - result.points[i].point_ref;
        }
    }

    // Apply world scale and compute final world-space fields
    for (auto& pt : result.points) {
        pt.point_ref_world  = pt.point_ref * opts_.world_scale;
        pt.point_def_world  = pt.point_def * opts_.world_scale;
        pt.displacement = pt.point_def - pt.point_ref;
        pt.displacement_world = pt.point_def_world - pt.point_ref_world;
        pt.displacement_norm_world = pt.displacement_world.norm();
    }

    result.valid_tracks = 0;
    for (const auto& pt : result.points) {
        if (pt.valid) ++result.valid_tracks;
    }

    return result;
}

} // namespace dic
