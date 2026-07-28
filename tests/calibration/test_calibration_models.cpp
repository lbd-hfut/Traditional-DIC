#include <dic/calibration/camera_model.hpp>
#include <dic/calibration/mono_calibration.hpp>
#include <dic/calibration/multiview_calibration.hpp>

#include <gtest/gtest.h>

TEST(CalibrationBoard, ChessboardObjectPoints)
{
    dic::CalibrationBoard board;
    board.type = dic::CalibrationBoardType::Chessboard;
    board.rows = 2;
    board.cols = 3;
    board.spacing = 0.5;

    const auto points = board.object_points();
    ASSERT_EQ(points.size(), 6u);
    EXPECT_DOUBLE_EQ(points.front().x(), 0.0);
    EXPECT_DOUBLE_EQ(points.front().y(), 0.0);
    EXPECT_DOUBLE_EQ(points.back().x(), 1.0);
    EXPECT_DOUBLE_EQ(points.back().y(), 0.5);
    EXPECT_DOUBLE_EQ(points.back().z(), 0.0);
}

TEST(CalibrationBoard, AsymmetricCircleObjectPoints)
{
    dic::CalibrationBoard board;
    board.type = dic::CalibrationBoardType::AsymmetricCircles;
    board.rows = 2;
    board.cols = 2;
    board.spacing = 1.0;

    const auto points = board.object_points();
    ASSERT_EQ(points.size(), 4u);
    EXPECT_DOUBLE_EQ(points[0].x(), 0.0);
    EXPECT_DOUBLE_EQ(points[1].x(), 2.0);
    EXPECT_DOUBLE_EQ(points[2].x(), 1.0);
    EXPECT_DOUBLE_EQ(points[3].x(), 3.0);
}

TEST(CameraModel, ProjectionMatrixAndCenter)
{
    dic::CameraModel camera;
    camera.K << 2.0, 0.0, 10.0, 0.0, 3.0, 20.0, 0.0, 0.0, 1.0;
    camera.t = Eigen::Vector3d(1.0, 2.0, 3.0);

    const auto P = camera.projection_matrix();
    EXPECT_DOUBLE_EQ(P(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(P(0, 3), 32.0);
    EXPECT_DOUBLE_EQ(P(1, 3), 66.0);

    const auto center = camera.camera_center();
    EXPECT_DOUBLE_EQ(center.x(), -1.0);
    EXPECT_DOUBLE_EQ(center.y(), -2.0);
    EXPECT_DOUBLE_EQ(center.z(), -3.0);
}

namespace {

Eigen::Vector2d project_for_test(const dic::CameraModel& camera, const Eigen::Vector3d& point)
{
    const Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    const Eigen::Vector3d projected = camera.projection_matrix() * homogeneous;
    return {projected.x() / projected.z(), projected.y() / projected.z()};
}

} // namespace

TEST(MultiviewScale, RecoversChessboardScale)
{
    const double sfm_to_world = 5.0;
    const double world_to_sfm = 1.0 / sfm_to_world;

    dic::CameraModel cam0;
    cam0.K << 900.0, 0.0, 640.0, 0.0, 900.0, 480.0, 0.0, 0.0, 1.0;
    cam0.R = Eigen::Matrix3d::Identity();
    cam0.t = Eigen::Vector3d(0.0, 0.0, 800.0 * world_to_sfm);

    dic::CameraModel cam1 = cam0;
    cam1.t = Eigen::Vector3d(160.0 * world_to_sfm, 0.0, 820.0 * world_to_sfm);

    std::vector<Eigen::Vector3d> board_points_sfm;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            board_points_sfm.emplace_back(20.0 * c * world_to_sfm, 20.0 * r * world_to_sfm, 0.0);
        }
    }

    dic::MultiviewScaleObservation obs0;
    obs0.camera_index = 0;
    dic::MultiviewScaleObservation obs1;
    obs1.camera_index = 1;
    for (const auto& point : board_points_sfm) {
        obs0.image_points.push_back(project_for_test(cam0, point));
        obs1.image_points.push_back(project_for_test(cam1, point));
    }

    dic::SparsePoint3D sparse;
    sparse.point = Eigen::Vector3d(2.0, 4.0, 6.0);

    dic::MultiviewScaleOptions options;
    options.board_rows = 3;
    options.board_cols = 3;
    options.square_size = 20.0;
    options.max_reprojection_error = 1e-8;
    options.trim_fraction = 0.0;
    options.min_common_corners = 4;

    const auto result = dic::estimate_multiview_chessboard_scale({cam0, cam1}, {sparse}, {obs0, obs1}, options);

    EXPECT_NEAR(result.sfm_to_world_scale, sfm_to_world, 1e-9);
    EXPECT_NEAR(result.world_to_sfm_scale, world_to_sfm, 1e-9);
    EXPECT_EQ(result.triangulated_corners, 9);
    EXPECT_EQ(result.valid_edges, 12);
    ASSERT_EQ(result.scaled_cameras.size(), 2u);
    EXPECT_NEAR(result.scaled_cameras[1].t.x(), 160.0, 1e-9);
    ASSERT_EQ(result.scaled_sparse_points.size(), 1u);
    EXPECT_NEAR(result.scaled_sparse_points[0].point.z(), 30.0, 1e-9);
}
