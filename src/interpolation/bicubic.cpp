/**
 * @file bicubic.cpp
 * @brief Minimal implementation placeholder for bicubic.
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

#include <dic/interpolation/bicubic.hpp>
#include <stdexcept>

namespace dic {

BicubicInterpolator::BicubicInterpolator(const Image& image) : image_(&image) { (void)image; }
double BicubicInterpolator::value(double x, double y) const { (void)x; (void)y; throw std::runtime_error("Not implemented yet."); }
Eigen::Vector2d BicubicInterpolator::gradient(double x, double y) const { (void)x; (void)y; throw std::runtime_error("Not implemented yet."); }

} // namespace dic
