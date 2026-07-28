#include <dic/calibration/camera_model.hpp>
#include <dic/geometry/projection.hpp>

#include <gtest/gtest.h>

namespace {

dic::CameraModel make_camera()
{
    dic::CameraModel camera;
    camera.K << 800.0, 0.0, 640.0,
                0.0, 810.0, 480.0,
                0.0, 0.0, 1.0;
    return camera;
}

} // namespace

TEST(Projection, ProjectsPinholePoint)
{
    const dic::CameraModel camera = make_camera();
    const Eigen::Vector2d pixel = dic::project_point({20.0, -10.0, 1000.0}, camera);
    EXPECT_NEAR(pixel.x(), 656.0, 1e-10);
    EXPECT_NEAR(pixel.y(), 471.9, 1e-10);
}

TEST(Projection, AppliesRadialAndTangentialDistortion)
{
    dic::CameraModel camera = make_camera();
    camera.distortion = {0.1, -0.05, 0.001, -0.002, 0.01};

    const Eigen::Vector2d normalized(0.2, -0.1);
    const Eigen::Vector2d distorted = dic::distort_normalized_point(normalized, camera.distortion);
    const Eigen::Vector2d pixel = dic::project_point({200.0, -100.0, 1000.0}, camera);

    EXPECT_NEAR(pixel.x(), 800.0 * distorted.x() + 640.0, 1e-10);
    EXPECT_NEAR(pixel.y(), 810.0 * distorted.y() + 480.0, 1e-10);
}

TEST(Projection, ComputesDepthAndReprojectionError)
{
    dic::CameraModel camera = make_camera();
    camera.t = {0.0, 0.0, -100.0};

    const Eigen::Vector3d point(0.0, 0.0, 1000.0);
    EXPECT_NEAR(dic::camera_depth(point, camera), 900.0, 1e-12);
    EXPECT_NEAR(dic::reprojection_error(point, dic::project_point(point, camera), camera), 0.0, 1e-12);
}
