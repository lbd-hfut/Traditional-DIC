/**
 * @file mask.cpp
 * @brief Minimal implementation placeholder for mask.
 *
 * Responsibilities:
 * - Provide binary ROI/mask storage and validity queries.
 * - Load user-supplied ROI mask images when OpenCV is enabled.
 *
 * Inputs:
 * - Dimensions, boolean mask data, or mask image path.
 *
 * Outputs:
 * - Bounds-checked pixel validity state.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Add mask image threshold configuration if needed.
 * - Add tests for mask loading and invalid dimension handling.
 */

#include <dic/core/mask.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace dic {

Mask::Mask(int width, int height)
    : width_(width), height_(height)
{
    if (width_ < 0 || height_ < 0) {
        throw std::invalid_argument("Mask dimensions must be non-negative.");
    }
    data_.assign(static_cast<std::size_t>(width_ * height_), true);
}

Mask::Mask(int width, int height, std::vector<bool> data)
    : width_(width), height_(height), data_(std::move(data))
{
    if (width_ < 0 || height_ < 0) {
        throw std::invalid_argument("Mask dimensions must be non-negative.");
    }
    const auto expected = static_cast<std::size_t>(width_ * height_);
    if (data_.size() != expected) {
        throw std::invalid_argument("Mask data size does not match dimensions.");
    }
}

Mask::Mask(const std::string& path)
{
#if defined(TRADITIONAL_DIC_HAS_OPENCV)
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        throw std::runtime_error("Failed to load mask image: " + path);
    }

    width_ = input.cols;
    height_ = input.rows;
    data_.resize(static_cast<std::size_t>(width_ * height_));
    for (int y = 0; y < height_; ++y) {
        const auto* row = input.ptr<unsigned char>(y);
        for (int x = 0; x < width_; ++x) {
            data_[static_cast<std::size_t>(y * width_ + x)] = row[x] > 0;
        }
    }
#else
    (void)path;
    throw std::runtime_error("Mask image loading requires OpenCV support.");
#endif
}

int Mask::width() const { return width_; }
int Mask::height() const { return height_; }
bool Mask::empty() const { return data_.empty(); }

bool Mask::contains(int x, int y) const
{
    return x >= 0 && y >= 0 && x < width_ && y < height_;
}

std::size_t Mask::size() const
{
    return data_.size();
}

bool Mask::valid(int x, int y) const
{
    return contains(x, y) && data_[static_cast<std::size_t>(y * width_ + x)];
}

void Mask::set(int x, int y, bool valid_value)
{
    if (!contains(x, y)) {
        throw std::out_of_range("Mask coordinate out of bounds.");
    }
    data_[static_cast<std::size_t>(y * width_ + x)] = valid_value;
}

void Mask::fill(bool valid_value)
{
    std::fill(data_.begin(), data_.end(), valid_value);
}

} // namespace dic
