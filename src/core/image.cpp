/**
 * @file image.cpp
 * @brief Minimal implementation placeholder for image container.
 *
 * Responsibilities:
 * - Provide memory-backed image construction and pixel access.
 * - Load grayscale float images from disk when OpenCV is enabled.
 * - Apply shared intensity scaling and mean/std normalization policies.
 *
 * Inputs:
 * - Image path or caller-provided image dimensions and float data.
 *
 * Outputs:
 * - Validated image container state and bounds-checked pixel access.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Add tests for OpenCV and non-OpenCV builds.
 */

#include <dic/core/image.hpp>

#include <dic/core/mask.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace dic {
namespace {

void validate_not_empty(const Image& image)
{
    if (image.empty()) {
        throw std::invalid_argument("Image must not be empty.");
    }
}

void validate_matching_mask(const Image& image, const Mask& roi)
{
    if (roi.empty() || roi.width() != image.width() || roi.height() != image.height()) {
        throw std::invalid_argument("ROI mask dimensions must match image dimensions.");
    }
}

Image normalize_from_stats(const Image& image, double mean, double stddev)
{
    constexpr double epsilon = 1e-12;
    std::vector<float> normalized;
    normalized.reserve(image.size());
    if (stddev <= epsilon) {
        normalized.assign(image.size(), 0.0F);
    } else {
        for (float value : image.data()) {
            normalized.push_back(static_cast<float>((static_cast<double>(value) - mean) / stddev));
        }
    }
    return Image(image.width(), image.height(), std::move(normalized));
}

} // namespace

Image::Image() = default;

Image::Image(const std::string& path)
    : Image(path, ImageLoadOptions{})
{
}

Image::Image(const std::string& path, ImageLoadOptions options)
{
#if defined(TRADITIONAL_DIC_HAS_OPENCV)
    cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    cv::Mat grayscale;
    if (options.color_mode == ImageColorMode::Unchanged && input.channels() != 1) {
        throw std::invalid_argument("Unchanged image loading only supports single-channel images.");
    }
    if (input.channels() == 1) {
        grayscale = input;
    } else {
        cv::cvtColor(input, grayscale, cv::COLOR_BGR2GRAY);
    }

    cv::Mat float_image;
    grayscale.convertTo(float_image, CV_32F);

    if (options.intensity_scale == ImageIntensityScale::Unit) {
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(float_image, &min_value, &max_value);
        if (max_value > 0.0) {
            float_image.convertTo(float_image, CV_32F, 1.0 / max_value);
        } else {
            float_image.setTo(0.0F);
        }
    }

    width_ = float_image.cols;
    height_ = float_image.rows;
    data_.resize(static_cast<std::size_t>(width_ * height_));

    for (int y = 0; y < height_; ++y) {
        const auto* row = float_image.ptr<float>(y);
        for (int x = 0; x < width_; ++x) {
            data_[static_cast<std::size_t>(y * width_ + x)] = row[x];
        }
    }
#else
    (void)options;
    (void)path;
    throw std::runtime_error("Image file loading requires OpenCV support.");
#endif
}

Image::Image(int width, int height, std::vector<float> data)
    : width_(width), height_(height), data_(std::move(data))
{
    if (width_ < 0 || height_ < 0) {
        throw std::invalid_argument("Image dimensions must be non-negative.");
    }
    const auto expected = static_cast<std::size_t>(width_ * height_);
    if (data_.size() != expected) {
        throw std::invalid_argument("Image data size does not match dimensions.");
    }
}

int Image::width() const { return width_; }
int Image::height() const { return height_; }
bool Image::empty() const { return data_.empty(); }

bool Image::contains(int x, int y) const
{
    return x >= 0 && y >= 0 && x < width_ && y < height_;
}

std::size_t Image::size() const
{
    return data_.size();
}

float Image::at(int x, int y) const
{
    if (!contains(x, y)) {
        throw std::out_of_range("Image coordinate out of bounds.");
    }
    return data_[static_cast<std::size_t>(y * width_ + x)];
}

void Image::set(int x, int y, float value)
{
    if (!contains(x, y)) {
        throw std::out_of_range("Image coordinate out of bounds.");
    }
    data_[static_cast<std::size_t>(y * width_ + x)] = value;
}

const std::vector<float>& Image::data() const
{
    return data_;
}

std::vector<float>& Image::data()
{
    return data_;
}

Image normalize_image(
    const Image& image,
    ImageNormalization normalization,
    const Mask* roi
)
{
    switch (normalization) {
    case ImageNormalization::None:
        return Image(image.width(), image.height(), image.data());
    case ImageNormalization::MaxIntensity:
        return normalize_max_intensity(image);
    case ImageNormalization::GlobalMeanStd:
        return normalize_global_mean_std(image);
    case ImageNormalization::RoiMeanStd:
        if (roi == nullptr) {
            throw std::invalid_argument("ROI normalization requires a mask.");
        }
        return normalize_roi_mean_std(image, *roi);
    }

    throw std::invalid_argument("Unsupported image normalization policy.");
}

Image normalize_max_intensity(const Image& image)
{
    validate_not_empty(image);

    const auto max_it = std::max_element(image.data().begin(), image.data().end());
    const float max_value = max_it != image.data().end() ? *max_it : 0.0F;

    std::vector<float> normalized;
    normalized.reserve(image.size());
    if (max_value <= 0.0F) {
        normalized.assign(image.size(), 0.0F);
    } else {
        for (float value : image.data()) {
            normalized.push_back(value / max_value);
        }
    }
    return Image(image.width(), image.height(), std::move(normalized));
}

Image normalize_global_mean_std(const Image& image)
{
    validate_not_empty(image);

    double sum = 0.0;
    for (float value : image.data()) {
        sum += static_cast<double>(value);
    }
    const double mean = sum / static_cast<double>(image.size());

    double variance = 0.0;
    for (float value : image.data()) {
        const double delta = static_cast<double>(value) - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(image.size());

    return normalize_from_stats(image, mean, std::sqrt(variance));
}

Image normalize_roi_mean_std(const Image& image, const Mask& roi)
{
    validate_not_empty(image);
    validate_matching_mask(image, roi);

    double sum = 0.0;
    std::size_t count = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (roi.valid(x, y)) {
                sum += static_cast<double>(image.at(x, y));
                ++count;
            }
        }
    }
    if (count == 0U) {
        throw std::invalid_argument("ROI normalization requires at least one valid pixel.");
    }

    const double mean = sum / static_cast<double>(count);
    double variance = 0.0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (roi.valid(x, y)) {
                const double delta = static_cast<double>(image.at(x, y)) - mean;
                variance += delta * delta;
            }
        }
    }
    variance /= static_cast<double>(count);

    return normalize_from_stats(image, mean, std::sqrt(variance));
}

} // namespace dic
