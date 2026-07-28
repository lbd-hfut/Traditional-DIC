#include <dic/calibration/camera_model.hpp>
#include <dic/geometry/multiview_triangulation.hpp>
#include <dic/geometry/projection.hpp>

#include <gtest/gtest.h>

namespace {

dic::CameraModel make_camera(const Eigen::Vector3d& center)
{
    dic::CameraModel camera;
    camera.K << 900.0, 0.0, 640.0,
                0.0, 900.0, 480.0,
                0.0, 0.0, 1.0;
    camera.t = -center;
    return camera;
}

} // namespace

TEST(MultiviewTriangulation, RecoversKnownPointFromThreeViews)
{
    const std::vector<dic::CameraModel> cameras = {
        make_camera({0.0, 0.0, 0.0}),
        make_camera({120.0, 0.0, 0.0}),
        make_camera({0.0, 90.0, 0.0}),
    };
    const Eigen::Vector3d expected(25.0, -15.0, 950.0);
    std::vector<Eigen::Vector2d> observations;
    for (const auto& camera : cameras) {
        observations.push_back(dic::project_point(expected, camera));
    }

    const auto result = dic::triangulate_multiview_checked(observations, cameras);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.observations_used, 3);
    EXPECT_NEAR((result.point - expected).norm(), 0.0, 1e-8);
    EXPECT_NEAR(result.max_reprojection_error, 0.0, 1e-8);
}

TEST(MultiviewTriangulation, RejectsPointBehindCamera)
{
    const std::vector<dic::CameraModel> cameras = {
        make_camera({0.0, 0.0, 0.0}),
        make_camera({120.0, 0.0, 0.0}),
    };
    const std::vector<Eigen::Vector2d> observations = {
        dic::project_point({0.0, 0.0, -500.0}, cameras[0]),
        dic::project_point({0.0, 0.0, -500.0}, cameras[1]),
    };

    const auto result = dic::triangulate_multiview_checked(observations, cameras);

    EXPECT_FALSE(result.valid);
}
