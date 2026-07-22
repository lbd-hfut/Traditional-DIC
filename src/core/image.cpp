/**
 * @file image.cpp
 * @brief Minimal implementation placeholder for image container.
 *
 * Responsibilities:
 * - Provide memory-backed image construction and pixel access.
 * - Load grayscale float images from disk when OpenCV is enabled.
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
 * - Add configurable normalization and image metadata handling.
 * - Add tests for OpenCV and non-OpenCV builds.
 */

#include <dic/core/image.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace dic {

Image::Image() = default;

Image::Image(const std::string& path)
{
#if defined(TRADITIONAL_DIC_HAS_OPENCV)
    cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    cv::Mat grayscale;
    if (input.channels() == 1) {
        grayscale = input;
    } else {
        cv::cvtColor(input, grayscale, cv::COLOR_BGR2GRAY);
    }

    cv::Mat float_image;
    grayscale.convertTo(float_image, CV_32F);

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

} // namespace dic
