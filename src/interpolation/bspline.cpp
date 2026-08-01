/**
 * @file bspline.cpp
 * @brief B-spline image preprocessing and interpolation implementation.
 *
 * Responsibilities:
 * - Build B-spline coefficient images for degree 1/3/5 interpolation.
 * - Build local polynomial coefficient blocks for each integer image pixel.
 * - Provide subpixel intensity and gradient queries for Subset-DIC and Mesh-DIC.
 *
 * Inputs:
 * - Grayscale Image data and BSplinePrecomputeConfig.
 *
 * Outputs:
 * - BSplinePrecomputedImage, interpolated intensities, and image gradients.
 *
 * Dependencies:
 * - Image core container and Eigen matrix types.
 *
 * TODO:
 * - Add SIMD/OpenMP acceleration for per-pixel local block construction.
 * - Add numerical regression tests against more reference datasets.
 */

#include <dic/interpolation/bspline.hpp>

#include <unsupported/Eigen/FFT>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

namespace dic {
namespace {

int degree_value(BSplineDegree degree)
{
    return static_cast<int>(degree);
}

void validate_degree(BSplineDegree degree)
{
    const int d = degree_value(degree);
    if (d != 1 && d != 3 && d != 5) {
        throw std::invalid_argument("Only B-spline degree 1, 3, and 5 are supported.");
    }
}

int factorial(int n)
{
    int value = 1;
    for (int i = 2; i <= n; ++i) {
        value *= i;
    }
    return value;
}

int combination(int n, int k)
{
    if (k < 0 || k > n) {
        return 0;
    }
    return factorial(n) / (factorial(k) * factorial(n - k));
}

double plus_power(double x, int power)
{
    return x > 0.0 ? std::pow(x, power) : 0.0;
}

int clamp_index(int value, int lower, int upper)
{
    return std::max(lower, std::min(value, upper));
}

int symmetric_index(int value, int size)
{
    if (size <= 0) {
        return 0;
    }
    const int period = 2 * size;
    int wrapped = value % period;
    if (wrapped < 0) {
        wrapped += period;
    }
    return wrapped < size ? wrapped : period - 1 - wrapped;
}

std::size_t flat_index(int x, int y, int width)
{
    return static_cast<std::size_t>(y * width + x);
}

double evaluate_polynomial(
    const Eigen::MatrixXd& block,
    double dx,
    double dy
)
{
    double value = 0.0;
    double y_power = 1.0;
    for (Eigen::Index row = 0; row < block.rows(); ++row) {
        double x_power = 1.0;
        for (Eigen::Index col = 0; col < block.cols(); ++col) {
            value += block(row, col) * y_power * x_power;
            x_power *= dx;
        }
        y_power *= dy;
    }
    return value;
}

double evaluate_polynomial_dx(
    const Eigen::MatrixXd& block,
    double dx,
    double dy
)
{
    double value = 0.0;
    double y_power = 1.0;
    for (Eigen::Index row = 0; row < block.rows(); ++row) {
        double x_power = 1.0;
        for (Eigen::Index col = 1; col < block.cols(); ++col) {
            value += static_cast<double>(col) * block(row, col) * y_power * x_power;
            x_power *= dx;
        }
        y_power *= dy;
    }
    return value;
}

double evaluate_polynomial_dy(
    const Eigen::MatrixXd& block,
    double dx,
    double dy
)
{
    double value = 0.0;
    double y_power = 1.0;
    for (Eigen::Index row = 1; row < block.rows(); ++row) {
        double x_power = 1.0;
        for (Eigen::Index col = 0; col < block.cols(); ++col) {
            value += static_cast<double>(row) * block(row, col) * y_power * x_power;
            x_power *= dx;
        }
        y_power *= dy;
    }
    return value;
}

Eigen::MatrixXd build_local_block_on_demand(
    const BSplinePrecomputedImage& precomputed,
    int x,
    int y
)
{
    const int d = degree_value(precomputed.config.degree);
    const int n = d + 1;
    const int offset = d / 2;
    const int top = y + precomputed.config.border - offset;
    const int left = x + precomputed.config.border - offset;

    Eigen::MatrixXd coefficient_block(n, n);
    for (int row = 0; row < n; ++row) {
        const int source_y = clamp_index(top + row, 0, static_cast<int>(precomputed.coefficients.rows()) - 1);
        for (int col = 0; col < n; ++col) {
            const int source_x = clamp_index(left + col, 0, static_cast<int>(precomputed.coefficients.cols()) - 1);
            coefficient_block(row, col) = precomputed.coefficients(source_y, source_x);
        }
    }

    return precomputed.qk * coefficient_block * precomputed.qk.transpose();
}

std::vector<double> build_prefilter_kernel(int length, BSplineDegree degree)
{
    const int d = degree_value(degree);
    const int radius = d / 2;
    std::vector<double> kernel(static_cast<std::size_t>(length), 0.0);
    kernel[0] = BSplineImagePreprocessor::basis(0.0, 0, degree);
    for (int offset = 1; offset <= radius; ++offset) {
        const double value = BSplineImagePreprocessor::basis(static_cast<double>(offset), 0, degree);
        kernel[static_cast<std::size_t>(offset)] = value;
        kernel[static_cast<std::size_t>(length - offset)] = value;
    }
    return kernel;
}

std::vector<double> deconvolve_periodic_signal(
    const std::vector<double>& signal,
    const std::vector<double>& kernel
)
{
    constexpr double epsilon = 1e-14;
    Eigen::FFT<double> fft;
    std::vector<std::complex<double>> signal_spectrum;
    std::vector<std::complex<double>> kernel_spectrum;
    fft.fwd(signal_spectrum, signal);
    fft.fwd(kernel_spectrum, kernel);

    for (std::size_t i = 0; i < signal_spectrum.size(); ++i) {
        const double denom = std::norm(kernel_spectrum[i]);
        if (denom <= epsilon) {
            throw std::runtime_error("B-spline prefilter kernel has a near-zero FFT coefficient.");
        }
        signal_spectrum[i] /= kernel_spectrum[i];
    }

    std::vector<double> output;
    fft.inv(output, signal_spectrum);
    return output;
}

void deconvolve_rows(Eigen::MatrixXd& coefficients, BSplineDegree degree)
{
    const auto kernel = build_prefilter_kernel(static_cast<int>(coefficients.cols()), degree);
    for (Eigen::Index y = 0; y < coefficients.rows(); ++y) {
        std::vector<double> signal(static_cast<std::size_t>(coefficients.cols()));
        for (Eigen::Index x = 0; x < coefficients.cols(); ++x) {
            signal[static_cast<std::size_t>(x)] = coefficients(y, x);
        }
        const auto output = deconvolve_periodic_signal(signal, kernel);
        for (Eigen::Index x = 0; x < coefficients.cols(); ++x) {
            coefficients(y, x) = output[static_cast<std::size_t>(x)];
        }
    }
}

void deconvolve_columns(Eigen::MatrixXd& coefficients, BSplineDegree degree)
{
    const auto kernel = build_prefilter_kernel(static_cast<int>(coefficients.rows()), degree);
    for (Eigen::Index x = 0; x < coefficients.cols(); ++x) {
        std::vector<double> signal(static_cast<std::size_t>(coefficients.rows()));
        for (Eigen::Index y = 0; y < coefficients.rows(); ++y) {
            signal[static_cast<std::size_t>(y)] = coefficients(y, x);
        }
        const auto output = deconvolve_periodic_signal(signal, kernel);
        for (Eigen::Index y = 0; y < coefficients.rows(); ++y) {
            coefficients(y, x) = output[static_cast<std::size_t>(y)];
        }
    }
}

} // namespace

bool BSplinePrecomputedImage::empty() const
{
    return width <= 0 || height <= 0 || coefficients.size() == 0;
}

const Eigen::MatrixXd& BSplinePrecomputedImage::local_block(int x, int y) const
{
    if (x < 0 || y < 0 || x >= width || y >= height) {
        throw std::out_of_range("B-spline local block coordinate out of bounds.");
    }
    return local_polynomial_blocks[flat_index(x, y, width)];
}

BSplineImagePreprocessor::BSplineImagePreprocessor(BSplinePrecomputeConfig config)
    : config_(config)
{
    validate_degree(config_.degree);
    if (config_.border < 3) {
        config_.border = 3;
    }
}

BSplinePrecomputedImage BSplineImagePreprocessor::compute(const Image& image) const
{
    if (image.empty()) {
        throw std::invalid_argument("Cannot precompute B-spline data for an empty image.");
    }

    BSplinePrecomputedImage result;
    result.width = image.width();
    result.height = image.height();
    result.config = config_;
    result.qk = build_qk(config_.degree);
    result.coefficients = form_coefficients(image);
    result.gradient_x = Eigen::MatrixXd::Zero(result.height, result.width);
    result.gradient_y = Eigen::MatrixXd::Zero(result.height, result.width);
    if (config_.precompute_local_blocks) {
        result.local_polynomial_blocks.reserve(static_cast<std::size_t>(result.width * result.height));
        for (int y = 0; y < result.height; ++y) {
            for (int x = 0; x < result.width; ++x) {
                const auto coefficient_block = extract_coefficient_block(result.coefficients, x, y);
                auto local_block = build_local_polynomial_block(coefficient_block, result.qk);
                if (local_block.rows() > 1 && local_block.cols() > 1) {
                    result.gradient_x(y, x) = evaluate_polynomial_dx(local_block, 0.5, 0.5);
                    result.gradient_y(y, x) = evaluate_polynomial_dy(local_block, 0.5, 0.5);
                }
                result.local_polynomial_blocks.push_back(std::move(local_block));
            }
        }
    }

    return result;
}

BSplinePrecomputedImage BSplineImagePreprocessor::compute_lazy(const Image& image) const
{
    if (image.empty()) {
        throw std::invalid_argument("Cannot precompute B-spline data for an empty image.");
    }

    BSplinePrecomputedImage result;
    result.width = image.width();
    result.height = image.height();
    result.config = config_;
    result.config.precompute_local_blocks = false;
    result.qk = build_qk(config_.degree);
    result.coefficients = form_coefficients(image);
    result.gradient_x = Eigen::MatrixXd::Zero(result.height, result.width);
    result.gradient_y = Eigen::MatrixXd::Zero(result.height, result.width);
    return result;
}

Eigen::MatrixXd BSplineImagePreprocessor::build_qk(BSplineDegree degree)
{
    validate_degree(degree);

    const int d = degree_value(degree);
    const int n = d + 1;
    const int offset = d / 2;
    Eigen::MatrixXd qk(n, n);

    for (int derivative_order = 0; derivative_order < n; ++derivative_order) {
        for (int col = 0; col < n; ++col) {
            const double x = static_cast<double>(-offset + col);
            const double sign = derivative_order % 2 == 0 ? 1.0 : -1.0;
            qk(derivative_order, col) =
                sign * basis(x, derivative_order, degree) /
                static_cast<double>(factorial(derivative_order));
        }
    }

    return qk;
}

double BSplineImagePreprocessor::basis(
    double x,
    int derivative_order,
    BSplineDegree degree
)
{
    validate_degree(degree);

    const int d = degree_value(degree);
    if (derivative_order < 0 || derivative_order > d) {
        return 0.0;
    }

    const int derivative_factor = factorial(d) / factorial(d - derivative_order);
    double sum = 0.0;
    for (int k = 0; k <= d + 1; ++k) {
        const double sign = k % 2 == 0 ? 1.0 : -1.0;
        const double coefficient = sign * static_cast<double>(combination(d + 1, k));
        const double shift = (static_cast<double>(d + 1) / 2.0) - static_cast<double>(k);
        sum += coefficient *
            static_cast<double>(derivative_factor) *
            plus_power(x + shift, d - derivative_order);
    }

    return sum / static_cast<double>(factorial(d));
}

Eigen::MatrixXd BSplineImagePreprocessor::form_coefficients(const Image& image) const
{
    const int border = config_.border;
    Eigen::MatrixXd coefficients(image.height() + 2 * border, image.width() + 2 * border);

    for (int y = 0; y < coefficients.rows(); ++y) {
        const int source_y = config_.use_exact_prefilter
            ? symmetric_index(y - border, image.height())
            : clamp_index(y - border, 0, image.height() - 1);
        for (int x = 0; x < coefficients.cols(); ++x) {
            const int source_x = config_.use_exact_prefilter
                ? symmetric_index(x - border, image.width())
                : clamp_index(x - border, 0, image.width() - 1);
            coefficients(y, x) = static_cast<double>(image.at(source_x, source_y));
        }
    }

    if (config_.use_exact_prefilter) {
        deconvolve_rows(coefficients, config_.degree);
        deconvolve_columns(coefficients, config_.degree);
    }

    return coefficients;
}

Eigen::MatrixXd BSplineImagePreprocessor::extract_coefficient_block(
    const Eigen::MatrixXd& coefficients,
    int x,
    int y
) const
{
    const int d = degree_value(config_.degree);
    const int n = d + 1;
    const int offset = d / 2;
    const int top = y + config_.border - offset;
    const int left = x + config_.border - offset;

    Eigen::MatrixXd block(n, n);
    for (int row = 0; row < n; ++row) {
        const int source_y = clamp_index(top + row, 0, static_cast<int>(coefficients.rows()) - 1);
        for (int col = 0; col < n; ++col) {
            const int source_x = clamp_index(left + col, 0, static_cast<int>(coefficients.cols()) - 1);
            block(row, col) = coefficients(source_y, source_x);
        }
    }

    return block;
}

Eigen::MatrixXd BSplineImagePreprocessor::build_local_polynomial_block(
    const Eigen::MatrixXd& coefficient_block,
    const Eigen::MatrixXd& qk
) const
{
    return qk * coefficient_block * qk.transpose();
}

BSplineInterpolator::BSplineInterpolator(const Image& image)
    : BSplineInterpolator(image, BSplinePrecomputeConfig{})
{
}

BSplineInterpolator::BSplineInterpolator(
    const Image& image,
    BSplinePrecomputeConfig config
)
    : config_(config), image_(image)
{
    precompute();
}

BSplineInterpolator::BSplineInterpolator(BSplinePrecomputedImage precomputed)
    : config_(precomputed.config), precomputed_(std::move(precomputed))
{
}

BSplineInterpolator::BSplineInterpolator(const BSplinePrecomputedImage* precomputed)
    : config_(precomputed != nullptr ? precomputed->config : BSplinePrecomputeConfig{}),
      external_precomputed_(precomputed)
{
}

void BSplineInterpolator::precompute()
{
    BSplineImagePreprocessor preprocessor(config_);
    precomputed_ = preprocessor.compute(image_);
    external_precomputed_ = nullptr;
}

double BSplineInterpolator::value(double x, double y) const
{
    const auto& precomputed = external_precomputed_ != nullptr ? *external_precomputed_ : precomputed_;
    if (precomputed.empty()) {
        throw std::runtime_error("B-spline interpolator has no precomputed image data.");
    }

    const int ix = clamp_index(static_cast<int>(std::floor(x)), 0, precomputed.width - 1);
    const int iy = clamp_index(static_cast<int>(std::floor(y)), 0, precomputed.height - 1);
    const double dx = x - static_cast<double>(ix);
    const double dy = y - static_cast<double>(iy);
    if (!precomputed.local_polynomial_blocks.empty()) {
        return evaluate_polynomial(precomputed.local_block(ix, iy), dx, dy);
    }
    const auto local_block = build_local_block_on_demand(precomputed, ix, iy);
    return evaluate_polynomial(local_block, dx, dy);
}

Eigen::Vector2d BSplineInterpolator::gradient(double x, double y) const
{
    const auto& precomputed = external_precomputed_ != nullptr ? *external_precomputed_ : precomputed_;
    if (precomputed.empty()) {
        throw std::runtime_error("B-spline interpolator has no precomputed image data.");
    }

    const int ix = clamp_index(static_cast<int>(std::floor(x)), 0, precomputed.width - 1);
    const int iy = clamp_index(static_cast<int>(std::floor(y)), 0, precomputed.height - 1);
    const double dx = x - static_cast<double>(ix);
    const double dy = y - static_cast<double>(iy);
    if (!precomputed.local_polynomial_blocks.empty()) {
        const auto& block = precomputed.local_block(ix, iy);
        return {
            evaluate_polynomial_dx(block, dx, dy),
            evaluate_polynomial_dy(block, dx, dy)
        };
    }

    const auto block = build_local_block_on_demand(precomputed, ix, iy);
    return {
        evaluate_polynomial_dx(block, dx, dy),
        evaluate_polynomial_dy(block, dx, dy)
    };
}

const BSplinePrecomputedImage& BSplineInterpolator::precomputed() const
{
    return external_precomputed_ != nullptr ? *external_precomputed_ : precomputed_;
}

} // namespace dic
