/**
 * @file bspline.hpp
 * @brief BSplineInterpolator interpolation placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP

#include <dic/core/image.hpp>
#include <dic/interpolation/interpolator.hpp>
#include <Eigen/Dense>

namespace dic {

class BSplineInterpolator : public Interpolator {
public:
    explicit BSplineInterpolator(const Image& image);
    void precompute();
    double value(double x, double y) const override;
    Eigen::Vector2d gradient(double x, double y) const override;
private:
    // TODO: Implement cubic B-spline coefficient precomputation.
    // TODO: Add Ncorr-like image pre-fitting, subpixel intensity evaluation,
    // gradient evaluation, memory optimization, and later SIMD/OpenMP paths.
    Eigen::MatrixXd coefficients_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP
