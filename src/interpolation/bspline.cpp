/**
 * @file bspline.cpp
 * @brief Minimal implementation placeholder for bspline.
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

#include <dic/interpolation/bspline.hpp>
#include <stdexcept>

namespace dic {

BSplineInterpolator::BSplineInterpolator(const Image& image) : coefficients_() { (void)image; }
void BSplineInterpolator::precompute() { coefficients_.resize(0, 0); }
double BSplineInterpolator::value(double x, double y) const { (void)x; (void)y; throw std::runtime_error("Not implemented yet."); }
Eigen::Vector2d BSplineInterpolator::gradient(double x, double y) const { (void)x; (void)y; throw std::runtime_error("Not implemented yet."); }

} // namespace dic
