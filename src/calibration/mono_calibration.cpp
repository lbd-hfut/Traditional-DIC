/**
 * @file mono_calibration.cpp
 * @brief Minimal implementation placeholder for mono calibration.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/calibration/mono_calibration.hpp>
#include <stdexcept>

namespace dic {

CameraModel calibrate_mono(int image_count) { (void)image_count; throw std::runtime_error("Not implemented yet."); }

} // namespace dic
