/**
 * @file image.hpp
 * @brief Grayscale floating-point image container.
 *
 * Responsibilities:
 * - Store grayscale image data as row-major float pixels.
 * - Provide path-based loading and shared DIC preprocessing helpers.
 * - Provide memory-based construction for tests and Python bindings.
 *
 * Inputs:
 * - Image file path or caller-provided width/height/pixel buffer.
 *
 * Outputs:
 * - Image dimensions, pixel values, and immutable/raw data access.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add OpenCV-backed loading tests for common image formats.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace dic {

class Mask;

enum class ImageColorMode {
    Unchanged,
    Grayscale
};

enum class ImageIntensityScale {
    Preserve,
    Unit
};

enum class ImageNormalization {
    None,
    MaxIntensity,
    GlobalMeanStd,
    RoiMeanStd
};

struct ImageLoadOptions {
    ImageColorMode color_mode{ImageColorMode::Grayscale};
    ImageIntensityScale intensity_scale{ImageIntensityScale::Unit};
};

class Image {
public:
    Image();
    explicit Image(const std::string& path);
    Image(const std::string& path, ImageLoadOptions options);
    Image(int width, int height, std::vector<float> data);

    int width() const;
    int height() const;
    bool empty() const;
    bool contains(int x, int y) const;
    std::size_t size() const;

    float at(int x, int y) const;
    void set(int x, int y, float value);

    const std::vector<float>& data() const;
    std::vector<float>& data();

private:
    int width_{0};
    int height_{0};
    std::vector<float> data_;
};

Image normalize_image(
    const Image& image,
    ImageNormalization normalization,
    const Mask* roi = nullptr
);

Image normalize_global_mean_std(const Image& image);
Image normalize_max_intensity(const Image& image);
Image normalize_roi_mean_std(const Image& image, const Mask& roi);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_IMAGE_HPP
