#include <dic/calibration/camera_model.hpp>
#include <dic/geometry/projection.hpp>
#include <dic/geometry/stereo_triangulation.hpp>

#include <gtest/gtest.h>

namespace {

dic::CameraModel make_left_camera()
{
    dic::CameraModel camera;
    camera.K << 800.0, 0.0, 640.0,
                0.0, 800.0, 480.0,
                0.0, 0.0, 1.0;
    return camera;
}

dic::CameraModel make_right_camera()
{
    dic::CameraModel camera = make_left_camera();
    camera.t = {-100.0, 0.0, 0.0};
    return camera;
}

} // namespace

TEST(StereoTriangulation, RecoversKnownPoint)
{
    const dic::CameraModel left = make_left_camera();
    const dic::CameraModel right = make_right_camera();
    const Eigen::Vector3d expected(20.0, 5.0, 1000.0);

    const auto result = dic::triangulate_stereo_checked(dic::project_point(expected, left),
                                                       dic::project_point(expected, right),
                                                       left,
                                                       right);

    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.point.x(), expected.x(), 1e-8);
    EXPECT_NEAR(result.point.y(), expected.y(), 1e-8);
    EXPECT_NEAR(result.point.z(), expected.z(), 1e-8);
    EXPECT_NEAR(result.mean_reprojection_error, 0.0, 1e-8);
}

TEST(StereoTriangulation, RejectsLargeReprojectionError)
{
    const dic::CameraModel left = make_left_camera();
    const dic::CameraModel right = make_right_camera();
    const Eigen::Vector3d expected(20.0, 5.0, 1000.0);

    dic::TriangulationOptions options;
    options.max_reprojection_error = 0.1;
    const auto result = dic::triangulate_stereo_checked(dic::project_point(expected, left),
                                                       dic::project_point(expected, right) + Eigen::Vector2d(0.0, 8.0),
                                                       left,
                                                       right,
                                                       options);

    EXPECT_FALSE(result.valid);
    EXPECT_GT(result.max_reprojection_error, options.max_reprojection_error);
}

TEST(StereoTriangulation, ReconstructsPointBatch)
{
    const dic::CameraModel left = make_left_camera();
    const dic::CameraModel right = make_right_camera();
    const std::vector<Eigen::Vector3d> points = {
        {0.0, 0.0, 800.0},
        {20.0, 5.0, 1000.0},
        {-15.0, 12.0, 1200.0},
    };
    std::vector<Eigen::Vector2d> left_points;
    std::vector<Eigen::Vector2d> right_points;
    for (const auto& point : points) {
        left_points.push_back(dic::project_point(point, left));
        right_points.push_back(dic::project_point(point, right));
    }

    const auto result = dic::reconstruct_stereo_points(left_points, right_points, left, right);

    ASSERT_EQ(result.points.size(), points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        EXPECT_EQ(result.valid_mask[i], 1);
        EXPECT_NEAR((result.points[i] - points[i]).norm(), 0.0, 1e-8);
    }
    EXPECT_NEAR(result.mean_reprojection_error, 0.0, 1e-8);
}
