/**
 * @file bspline.hpp
 * @brief B-spline interpolation and image precomputation interfaces.
 *
 * Responsibilities:
 * - Support first-, third-, and fifth-degree tensor-product B-spline interpolation.
 * - Precompute reference image B-spline coefficients for DIC solvers.
 * - Build per-integer-pixel local polynomial coefficient blocks and gradients.
 *
 * Inputs:
 * - Grayscale Image data and B-spline degree/preprocessing configuration.
 *
 * Outputs:
 * - Subpixel intensity values, gradients, coefficient images, local blocks, and gradient images.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Replace the first-pass edge-padded coefficient approximation with validated FFT
 *   or recursive B-spline prefiltering for degree 3/5.
 * - Add numerical equivalence tests against the reference Python/JAX pipeline.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP

#include <dic/core/image.hpp>
#include <dic/interpolation/interpolator.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace dic {

enum class BSplineDegree {
    Linear = 1,
    Cubic = 3,
    Quintic = 5
};

struct BSplinePrecomputeConfig {
    BSplineDegree degree{BSplineDegree::Quintic};
    int border{3};

    // TODO: Enable exact FFT/recursive coefficient prefiltering once the
    // numerical backend is selected. The current default keeps a usable,
    // deterministic edge-padded coefficient image for downstream integration.
    bool use_exact_prefilter{false};
};

struct BSplinePrecomputedImage {
    int width{0};
    int height{0};
    BSplinePrecomputeConfig config;

    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd qk;
    std::vector<Eigen::MatrixXd> local_polynomial_blocks;
    Eigen::MatrixXd gradient_x;
    Eigen::MatrixXd gradient_y;

    bool empty() const;
    const Eigen::MatrixXd& local_block(int x, int y) const;
};

class BSplineImagePreprocessor {
public:
    explicit BSplineImagePreprocessor(BSplinePrecomputeConfig config = {});

    BSplinePrecomputedImage compute(const Image& image) const;

    static Eigen::MatrixXd build_qk(BSplineDegree degree);
    static double basis(double x, int derivative_order, BSplineDegree degree);

private:
    Eigen::MatrixXd form_coefficients(const Image& image) const;
    Eigen::MatrixXd extract_coefficient_block(
        const Eigen::MatrixXd& coefficients,
        int x,
        int y
    ) const;
    Eigen::MatrixXd build_local_polynomial_block(
        const Eigen::MatrixXd& coefficient_block,
        const Eigen::MatrixXd& qk
    ) const;

    BSplinePrecomputeConfig config_;
};

class BSplineInterpolator : public Interpolator {
public:
    explicit BSplineInterpolator(const Image& image);
    explicit BSplineInterpolator(
        const Image& image,
        BSplinePrecomputeConfig config
    );
    explicit BSplineInterpolator(BSplinePrecomputedImage precomputed);

    void precompute();
    double value(double x, double y) const override;
    Eigen::Vector2d gradient(double x, double y) const override;

    const BSplinePrecomputedImage& precomputed() const;

private:
    BSplinePrecomputeConfig config_;
    Image image_;
    BSplinePrecomputedImage precomputed_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INTERPOLATION_BSPLINE_HPP
