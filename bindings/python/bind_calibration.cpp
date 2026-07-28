#include <dic/calibration/mono_calibration.hpp>
#include <dic/calibration/multiview_calibration.hpp>
#include <dic/calibration/stereo_calibration.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_calibration(py::module_& m)
{
    auto sub = m.def_submodule("calibration");

    py::enum_<dic::CalibrationBoardType>(sub, "CalibrationBoardType")
        .value("Chessboard", dic::CalibrationBoardType::Chessboard)
        .value("SymmetricCircles", dic::CalibrationBoardType::SymmetricCircles)
        .value("AsymmetricCircles", dic::CalibrationBoardType::AsymmetricCircles);

    py::class_<dic::CameraModel>(sub, "CameraModel")
        .def(py::init<>())
        .def_readwrite("K", &dic::CameraModel::K)
        .def_readwrite("distortion", &dic::CameraModel::distortion)
        .def_readwrite("R", &dic::CameraModel::R)
        .def_readwrite("t", &dic::CameraModel::t)
        .def_readwrite("image_width", &dic::CameraModel::image_width)
        .def_readwrite("image_height", &dic::CameraModel::image_height)
        .def_readwrite("rms_error", &dic::CameraModel::rms_error)
        .def_readwrite("label", &dic::CameraModel::label)
        .def("projection_matrix", &dic::CameraModel::projection_matrix)
        .def("camera_center", &dic::CameraModel::camera_center);

    py::class_<dic::CalibrationBoard>(sub, "CalibrationBoard")
        .def(py::init<>())
        .def_readwrite("type", &dic::CalibrationBoard::type)
        .def_readwrite("rows", &dic::CalibrationBoard::rows)
        .def_readwrite("cols", &dic::CalibrationBoard::cols)
        .def_readwrite("spacing", &dic::CalibrationBoard::spacing)
        .def("point_count", &dic::CalibrationBoard::point_count)
        .def("object_points", &dic::CalibrationBoard::object_points);

    py::class_<dic::BoardDetectionOptions>(sub, "BoardDetectionOptions")
        .def(py::init<>())
        .def_readwrite("refine_corners", &dic::BoardDetectionOptions::refine_corners)
        .def_readwrite("normalize_image", &dic::BoardDetectionOptions::normalize_image)
        .def_readwrite("max_iterations", &dic::BoardDetectionOptions::max_iterations)
        .def_readwrite("epsilon", &dic::BoardDetectionOptions::epsilon);

    py::class_<dic::CalibrationDetection>(sub, "CalibrationDetection")
        .def(py::init<>())
        .def_readwrite("found", &dic::CalibrationDetection::found)
        .def_readwrite("image_path", &dic::CalibrationDetection::image_path)
        .def_readwrite("image_width", &dic::CalibrationDetection::image_width)
        .def_readwrite("image_height", &dic::CalibrationDetection::image_height)
        .def_readwrite("image_points", &dic::CalibrationDetection::image_points);

    py::class_<dic::MonoCalibrationOptions>(sub, "MonoCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("detection", &dic::MonoCalibrationOptions::detection)
        .def_readwrite("estimate_tangential_distortion", &dic::MonoCalibrationOptions::estimate_tangential_distortion)
        .def_readwrite("estimate_k3", &dic::MonoCalibrationOptions::estimate_k3)
        .def_readwrite("max_iterations", &dic::MonoCalibrationOptions::max_iterations)
        .def_readwrite("epsilon", &dic::MonoCalibrationOptions::epsilon);

    py::class_<dic::MonoCalibrationResult>(sub, "MonoCalibrationResult")
        .def(py::init<>())
        .def_readwrite("camera", &dic::MonoCalibrationResult::camera)
        .def_readwrite("board_rotations", &dic::MonoCalibrationResult::board_rotations)
        .def_readwrite("board_translations", &dic::MonoCalibrationResult::board_translations)
        .def_readwrite("per_view_errors", &dic::MonoCalibrationResult::per_view_errors)
        .def_readwrite("detections", &dic::MonoCalibrationResult::detections)
        .def_readwrite("rms_error", &dic::MonoCalibrationResult::rms_error);

    py::class_<dic::StereoCalibrationOptions>(sub, "StereoCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("detection", &dic::StereoCalibrationOptions::detection)
        .def_readwrite("fix_intrinsics", &dic::StereoCalibrationOptions::fix_intrinsics)
        .def_readwrite("estimate_tangential_distortion", &dic::StereoCalibrationOptions::estimate_tangential_distortion)
        .def_readwrite("estimate_k3", &dic::StereoCalibrationOptions::estimate_k3)
        .def_readwrite("max_iterations", &dic::StereoCalibrationOptions::max_iterations)
        .def_readwrite("epsilon", &dic::StereoCalibrationOptions::epsilon);

    py::class_<dic::StereoCalibrationResult>(sub, "StereoCalibrationResult")
        .def(py::init<>())
        .def_readwrite("left", &dic::StereoCalibrationResult::left)
        .def_readwrite("right", &dic::StereoCalibrationResult::right)
        .def_readwrite("R_lr", &dic::StereoCalibrationResult::R_lr)
        .def_readwrite("t_lr", &dic::StereoCalibrationResult::t_lr)
        .def_readwrite("essential", &dic::StereoCalibrationResult::essential)
        .def_readwrite("fundamental", &dic::StereoCalibrationResult::fundamental)
        .def_readwrite("per_pair_errors", &dic::StereoCalibrationResult::per_pair_errors)
        .def_readwrite("left_detections", &dic::StereoCalibrationResult::left_detections)
        .def_readwrite("right_detections", &dic::StereoCalibrationResult::right_detections)
        .def_readwrite("rms_error", &dic::StereoCalibrationResult::rms_error);

    py::class_<dic::FeatureTrackObservation>(sub, "FeatureTrackObservation")
        .def(py::init<>())
        .def_readwrite("image_index", &dic::FeatureTrackObservation::image_index)
        .def_readwrite("point", &dic::FeatureTrackObservation::point);

    py::class_<dic::SparsePoint3D>(sub, "SparsePoint3D")
        .def(py::init<>())
        .def_readwrite("point", &dic::SparsePoint3D::point)
        .def_readwrite("observations", &dic::SparsePoint3D::observations)
        .def_readwrite("reprojection_error", &dic::SparsePoint3D::reprojection_error);

    py::class_<dic::MultiviewCalibrationOptions>(sub, "MultiviewCalibrationOptions")
        .def(py::init<>())
        .def_readwrite("max_features", &dic::MultiviewCalibrationOptions::max_features)
        .def_readwrite("match_ratio", &dic::MultiviewCalibrationOptions::match_ratio)
        .def_readwrite("ransac_reprojection_threshold", &dic::MultiviewCalibrationOptions::ransac_reprojection_threshold)
        .def_readwrite("min_triangulation_angle_degrees", &dic::MultiviewCalibrationOptions::min_triangulation_angle_degrees)
        .def_readwrite("min_inlier_matches", &dic::MultiviewCalibrationOptions::min_inlier_matches)
        .def_readwrite("refine_bundle", &dic::MultiviewCalibrationOptions::refine_bundle)
        .def_readwrite("initial_cameras", &dic::MultiviewCalibrationOptions::initial_cameras);

    py::class_<dic::MultiviewCalibrationResult>(sub, "MultiviewCalibrationResult")
        .def(py::init<>())
        .def_readwrite("cameras", &dic::MultiviewCalibrationResult::cameras)
        .def_readwrite("sparse_points", &dic::MultiviewCalibrationResult::sparse_points)
        .def_readwrite("inlier_match_counts", &dic::MultiviewCalibrationResult::inlier_match_counts)
        .def_readwrite("mean_reprojection_error", &dic::MultiviewCalibrationResult::mean_reprojection_error);

    py::class_<dic::MultiviewScaleObservation>(sub, "MultiviewScaleObservation")
        .def(py::init<>())
        .def_readwrite("camera_index", &dic::MultiviewScaleObservation::camera_index)
        .def_readwrite("image_points", &dic::MultiviewScaleObservation::image_points);

    py::class_<dic::MultiviewScaleOptions>(sub, "MultiviewScaleOptions")
        .def(py::init<>())
        .def_readwrite("board_rows", &dic::MultiviewScaleOptions::board_rows)
        .def_readwrite("board_cols", &dic::MultiviewScaleOptions::board_cols)
        .def_readwrite("square_size", &dic::MultiviewScaleOptions::square_size)
        .def_readwrite("max_reprojection_error", &dic::MultiviewScaleOptions::max_reprojection_error)
        .def_readwrite("trim_fraction", &dic::MultiviewScaleOptions::trim_fraction)
        .def_readwrite("min_common_corners", &dic::MultiviewScaleOptions::min_common_corners);

    py::class_<dic::MultiviewScaleResult>(sub, "MultiviewScaleResult")
        .def(py::init<>())
        .def_readwrite("sfm_to_world_scale", &dic::MultiviewScaleResult::sfm_to_world_scale)
        .def_readwrite("world_to_sfm_scale", &dic::MultiviewScaleResult::world_to_sfm_scale)
        .def_readwrite("sfm_square_size_mean", &dic::MultiviewScaleResult::sfm_square_size_mean)
        .def_readwrite("sfm_square_size_median", &dic::MultiviewScaleResult::sfm_square_size_median)
        .def_readwrite("sfm_square_size_std", &dic::MultiviewScaleResult::sfm_square_size_std)
        .def_readwrite("edge_cv", &dic::MultiviewScaleResult::edge_cv)
        .def_readwrite("triangulated_corners", &dic::MultiviewScaleResult::triangulated_corners)
        .def_readwrite("valid_edges", &dic::MultiviewScaleResult::valid_edges)
        .def_readwrite("triangulated_board_points_sfm", &dic::MultiviewScaleResult::triangulated_board_points_sfm)
        .def_readwrite("edge_lengths_sfm", &dic::MultiviewScaleResult::edge_lengths_sfm)
        .def_readwrite("scaled_cameras", &dic::MultiviewScaleResult::scaled_cameras)
        .def_readwrite("scaled_sparse_points", &dic::MultiviewScaleResult::scaled_sparse_points);

    sub.def("detect_calibration_board", &dic::detect_calibration_board, py::arg("image_path"), py::arg("board"), py::arg("options") = dic::BoardDetectionOptions{});
    sub.def("calibrate_mono_zhang", &dic::calibrate_mono_zhang, py::arg("image_paths"), py::arg("board"), py::arg("options") = dic::MonoCalibrationOptions{});
    sub.def("calibrate_mono_from_points", &dic::calibrate_mono_from_points, py::arg("object_points"), py::arg("image_points"), py::arg("image_width"), py::arg("image_height"), py::arg("options") = dic::MonoCalibrationOptions{});
    sub.def("calibrate_stereo_zhang", &dic::calibrate_stereo_zhang, py::arg("left_image_paths"), py::arg("right_image_paths"), py::arg("board"), py::arg("options") = dic::StereoCalibrationOptions{});
    sub.def("calibrate_stereo_from_points", &dic::calibrate_stereo_from_points, py::arg("object_points"), py::arg("left_image_points"), py::arg("right_image_points"), py::arg("image_width"), py::arg("image_height"), py::arg("options") = dic::StereoCalibrationOptions{});
    sub.def("calibrate_multiview_colmap_like", &dic::calibrate_multiview_colmap_like, py::arg("image_paths"), py::arg("options") = dic::MultiviewCalibrationOptions{});
    sub.def("estimate_multiview_chessboard_scale", &dic::estimate_multiview_chessboard_scale, py::arg("cameras"), py::arg("sparse_points"), py::arg("observations"), py::arg("options"));
}
