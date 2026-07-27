#include <dic/config/yaml_parser.hpp>
#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/subset/seed/seed_selector.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::uint16_t read_u16(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}

std::uint32_t read_u32(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset] |
                                      (bytes[offset + 1] << 8) |
                                      (bytes[offset + 2] << 16) |
                                      (bytes[offset + 3] << 24));
}

std::int32_t read_i32(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}

std::vector<unsigned char> read_file_bytes(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open image: " + path);
    }
    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()
    );
}

dic::Image read_bmp_image(const std::string& path)
{
    const auto bytes = read_file_bytes(path);
    if (bytes.size() < 54U || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::runtime_error("Only BMP images are supported by this diagnostic tool: " + path);
    }

    const auto data_offset = read_u32(bytes, 10);
    const auto dib_size = read_u32(bytes, 14);
    if (dib_size < 40U) {
        throw std::runtime_error("Unsupported BMP DIB header: " + path);
    }

    const int width = read_i32(bytes, 18);
    const int signed_height = read_i32(bytes, 22);
    const int height = std::abs(signed_height);
    const bool top_down = signed_height < 0;
    const auto planes = read_u16(bytes, 26);
    const auto bits_per_pixel = read_u16(bytes, 28);
    const auto compression = read_u32(bytes, 30);
    if (width <= 0 || height <= 0 || planes != 1 || compression != 0U) {
        throw std::runtime_error("Unsupported BMP layout: " + path);
    }
    if (bits_per_pixel != 8U && bits_per_pixel != 24U && bits_per_pixel != 32U) {
        throw std::runtime_error("Only 8/24/32-bit BMP images are supported: " + path);
    }

    const int bytes_per_pixel = static_cast<int>(bits_per_pixel / 8U);
    const int row_stride = ((width * static_cast<int>(bits_per_pixel) + 31) / 32) * 4;
    if (static_cast<std::size_t>(data_offset) + static_cast<std::size_t>(row_stride * height) > bytes.size()) {
        throw std::runtime_error("BMP pixel data is truncated: " + path);
    }

    std::vector<float> pixels(static_cast<std::size_t>(width * height), 0.0F);
    for (int y = 0; y < height; ++y) {
        const int source_y = top_down ? y : height - 1 - y;
        const auto row_offset = static_cast<std::size_t>(data_offset + source_y * row_stride);
        for (int x = 0; x < width; ++x) {
            const auto pixel_offset = row_offset + static_cast<std::size_t>(x * bytes_per_pixel);
            double gray = 0.0;
            if (bits_per_pixel == 8U) {
                gray = static_cast<double>(bytes[pixel_offset]);
            } else {
                const double b = static_cast<double>(bytes[pixel_offset]);
                const double g = static_cast<double>(bytes[pixel_offset + 1U]);
                const double r = static_cast<double>(bytes[pixel_offset + 2U]);
                gray = 0.114 * b + 0.587 * g + 0.299 * r;
            }
            pixels[static_cast<std::size_t>(y * width + x)] = static_cast<float>(gray / 255.0);
        }
    }
    return dic::Image(width, height, std::move(pixels));
}

dic::Mask read_bmp_mask(const std::string& path)
{
    const auto image = read_bmp_image(path);
    dic::Mask mask(image.width(), image.height());
    mask.fill(false);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            mask.set(x, y, image.at(x, y) > 0.0F);
        }
    }
    return mask;
}

void write_csv(const std::string& path, const dic::SeedSelectionResult& result)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV output: " + path);
    }

    out << "index,x,y,u,v,matched_x,matched_y,quality,disp_norm,quality_passed,valid,best\n";
    out << std::setprecision(17);
    for (std::size_t i = 0; i < result.candidates.size(); ++i) {
        const auto& candidate = result.candidates[i];
        const bool is_best = result.found &&
                             candidate.valid &&
                             candidate.point.x() == result.best_seed.point.x() &&
                             candidate.point.y() == result.best_seed.point.y();
        out << i << ','
            << candidate.point.x() << ','
            << candidate.point.y() << ','
            << candidate.displacement.u << ','
            << candidate.displacement.v << ','
            << candidate.point.x() + candidate.displacement.u << ','
            << candidate.point.y() + candidate.displacement.v << ','
            << candidate.quality << ','
            << candidate.displacement_norm << ','
            << (candidate.quality_passed ? 1 : 0) << ','
            << (candidate.valid ? 1 : 0) << ','
            << (is_best ? 1 : 0) << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 6) {
            std::cerr << "Usage: seed_matching_diagnostic <reference> <deformed> <roi> <out_csv> <config_yaml>\n";
            return 2;
        }

        const std::string reference_path = argv[1];
        const std::string deformed_path = argv[2];
        const std::string roi_path = argv[3];
        const std::string csv_path = argv[4];
        const std::string config_path = argv[5];

        const auto reference = read_bmp_image(reference_path);
        const auto deformed = read_bmp_image(deformed_path);
        const auto roi = read_bmp_mask(roi_path);

        const auto config = dic::load_subset_config_from_yaml(config_path);
        const dic::SeedSelector selector(config);
        const auto result = selector.select_best_seed(reference, deformed, roi);
        write_csv(csv_path, result);

        int passed = 0;
        int valid = 0;
        for (const auto& candidate : result.candidates) {
            if (candidate.valid) {
                ++valid;
            }
            if (candidate.valid && candidate.quality_passed) {
                ++passed;
            }
        }

        std::cout << "candidate_count=" << result.candidates.size() << '\n'
                  << "valid_count=" << valid << '\n'
                  << "quality_passed_count=" << passed << '\n'
                  << "found=" << (result.found ? 1 : 0) << '\n';
        if (result.found) {
            std::cout << std::setprecision(17)
                      << "best_x=" << result.best_seed.point.x() << '\n'
                      << "best_y=" << result.best_seed.point.y() << '\n'
                      << "best_u=" << result.best_seed.displacement.u << '\n'
                      << "best_v=" << result.best_seed.displacement.v << '\n'
                      << "best_quality=" << result.best_seed.quality << '\n'
                      << "best_disp_norm=" << result.best_seed.displacement_norm << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
