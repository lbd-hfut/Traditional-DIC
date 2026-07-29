#include <dic/geometry/projection.hpp>
#include <dic/postprocess/strain_3d.hpp>
#include <dic/reconstruction/stereo_dic.hpp>

#include <gtest/gtest.h>

namespace {

dic::CameraModel make_camera(const Eigen::Vector3d& center)
{
    dic::CameraModel camera;
    camera.K << 850.0, 0.0, 640.0,
                0.0, 850.0, 480.0,
                0.0, 0.0, 1.0;
    camera.t = -center;
    camera.distortion = {0.025, -0.006, 0.0005, -0.0003, 0.001};
    return camera;
}

dic::PointObservation make_observation(const Eigen::Vector3d& point_ref,
                                       const Eigen::Vector3d& point_def,
                                       const dic::CameraModel& camera,
                                       int camera_index)
{
    const Eigen::Vector2d uv_ref = dic::project_point(point_ref, camera);
    const Eigen::Vector2d uv_def = dic::project_point(point_def, camera);

    dic::PointObservation obs;
    obs.camera_index = camera_index;
    obs.uv_ref = uv_ref;
    obs.uv_def = uv_def;
    obs.u_displacement = uv_def.x() - uv_ref.x();
    obs.v_displacement = uv_def.y() - uv_ref.y();
    obs.correlation = 0.98;
    obs.dic_valid = true;
    return obs;
}

} // namespace

TEST(StereoDIC, ReconstructsReferenceShapeAndDisplacement)
{
    const dic::CameraModel left = make_camera({0.0, 0.0, 0.0});
    const dic::CameraModel right = make_camera({100.0, 0.0, 0.0});
    const Eigen::Vector3d point_ref(12.0, -6.0, 980.0);
    const Eigen::Vector3d point_def = point_ref + Eigen::Vector3d(1.5, -2.0, 4.0);

    const std::vector<dic::PointObservation> left_obs = {
        make_observation(point_ref, point_def, left, 0),
    };
    const std::vector<dic::PointObservation> right_obs = {
        make_observation(point_ref, point_def, right, 1),
    };

    dic::StereoDIC::Options options;
    options.max_reprojection_error_px = 1.0e-4;
    dic::StereoDIC dic3d(options);
    const dic::StereoDICResult result = dic3d.reconstruct(left_obs, right_obs, left, right);

    ASSERT_EQ(result.valid_points, 1);
    ASSERT_TRUE(result.points[0].valid);
    EXPECT_NEAR((result.points[0].point_ref - point_ref).norm(), 0.0, 1e-4);
    EXPECT_NEAR((result.points[0].point_def - point_def).norm(), 0.0, 1e-4);
    EXPECT_NEAR((result.points[0].displacement - (point_def - point_ref)).norm(), 0.0, 1e-4);
}

TEST(StereoDIC, RemovesRigidTranslation)
{
    const dic::CameraModel left = make_camera({0.0, 0.0, 0.0});
    const dic::CameraModel right = make_camera({100.0, 0.0, 0.0});
    const Eigen::Vector3d translation(2.0, -3.0, 5.0);
    const std::vector<Eigen::Vector3d> ref_points = {
        {0.0, 0.0, 900.0},
        {20.0, 5.0, 940.0},
        {-15.0, 18.0, 980.0},
        {8.0, -12.0, 1020.0},
    };

    std::vector<dic::PointObservation> left_obs;
    std::vector<dic::PointObservation> right_obs;
    for (const auto& ref : ref_points) {
        const Eigen::Vector3d def = ref + translation;
        left_obs.push_back(make_observation(ref, def, left, 0));
        right_obs.push_back(make_observation(ref, def, right, 1));
    }

    dic::StereoDIC::Options options;
    options.max_reprojection_error_px = 1.0e-4;
    options.remove_rigid_body_motion = true;
    dic::StereoDIC dic3d(options);
    const dic::StereoDICResult result = dic3d.reconstruct(left_obs, right_obs, left, right);

    ASSERT_EQ(result.valid_points, static_cast<int>(ref_points.size()));
    EXPECT_NEAR(result.mean_displacement_norm, 0.0, 1e-4);
    for (const auto& point : result.points) {
        EXPECT_TRUE(point.valid);
        EXPECT_NEAR(point.displacement.norm(), 0.0, 1e-4);
    }
}

TEST(Postprocess3D, ComputesSurfaceStrainForUniformStretch)
{
    const std::vector<std::array<int, 3>> faces = {{{0, 1, 2}}};
    const std::vector<Eigen::Vector3d> ref = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    const std::vector<Eigen::Vector3d> def = {
        {0.0, 0.0, 0.0},
        {1.1, 0.0, 0.0},
        {0.0, 1.2, 0.0},
    };
    const std::vector<bool> valid_faces = {true};

    const auto strain = dic::compute_surface_strain(faces, ref, def, valid_faces);

    ASSERT_EQ(strain.size(), 1);
    ASSERT_TRUE(strain[0].valid);
    EXPECT_NEAR(strain[0].Epc1, 0.105, 1e-8);
    EXPECT_NEAR(strain[0].Epc2, 0.22, 1e-8);
    EXPECT_NEAR(strain[0].Lambda1, 1.1, 1e-8);
    EXPECT_NEAR(strain[0].Lambda2, 1.2, 1e-8);
}
