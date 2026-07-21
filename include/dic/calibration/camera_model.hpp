/**
 * @file camera_model.hpp
 * @brief Pinhole camera model with projection matrix.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_CAMERA_MODEL_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_CAMERA_MODEL_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

struct CameraModel { Eigen::Matrix3d K = Eigen::Matrix3d::Identity(); std::vector<double> distortion; Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); Eigen::Vector3d t = Eigen::Vector3d::Zero(); Eigen::Matrix<double, 3, 4> projection_matrix() const; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CALIBRATION_CAMERA_MODEL_HPP
