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

namespace dic {

struct StereoCalibrationResult { CameraModel left; CameraModel right; Eigen::Matrix3d R_lr = Eigen::Matrix3d::Identity(); Eigen::Vector3d t_lr = Eigen::Vector3d::Zero(); };
StereoCalibrationResult calibrate_stereo(int image_pair_count);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_STEREO_CALIBRATION_HPP
