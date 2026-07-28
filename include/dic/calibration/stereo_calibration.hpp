/**
 * @file stereo_calibration.hpp
 * @brief Stereo calibration result and entry point.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_STEREO_CALIBRATION_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_STEREO_CALIBRATION_HPP

#include <dic/calibration/camera_model.hpp>
#include <dic/calibration/mono_calibration.hpp>

namespace dic {

struct StereoCalibrationOptions {
    BoardDetectionOptions detection;
    bool fix_intrinsics = false;
    bool estimate_tangential_distortion = true;
    bool estimate_k3 = true;
    int max_iterations = 100;
    double epsilon = 1e-9;
};

struct StereoCalibrationResult {
    CameraModel left;
    CameraModel right;
    Eigen::Matrix3d R_lr = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_lr = Eigen::Vector3d::Zero();
    Eigen::Matrix3d essential = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
    std::vector<double> per_pair_errors;
    std::vector<CalibrationDetection> left_detections;
    std::vector<CalibrationDetection> right_detections;
    double rms_error = 0.0;
};

StereoCalibrationResult calibrate_stereo_zhang(const std::vector<std::string>& left_image_paths,
                                               const std::vector<std::string>& right_image_paths,
                                               const CalibrationBoard& board,
                                               const StereoCalibrationOptions& options = {});

StereoCalibrationResult calibrate_stereo_from_points(
    const std::vector<std::vector<Eigen::Vector3d>>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& left_image_points,
    const std::vector<std::vector<Eigen::Vector2d>>& right_image_points,
    int image_width,
    int image_height,
    const StereoCalibrationOptions& options = {});

StereoCalibrationResult calibrate_stereo(int image_pair_count);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_STEREO_CALIBRATION_HPP
