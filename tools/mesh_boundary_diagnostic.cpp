#include <dic/core/mask.hpp>
#include <dic/mesh/generation/boundary_exporter.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint16_t read_u16(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1U] << 8));
}

std::uint32_t read_u32(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset] | (bytes[offset + 1U] << 8) |
                                      (bytes[offset + 2U] << 16) | (bytes[offset + 3U] << 24));
}

std::int32_t read_i32(const std::vector<unsigned char>& bytes, std::size_t offset)
{
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}

dic::Mask read_bmp_mask(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open mask image: " + path.string());
    }
    std::vector<unsigned char> bytes;
    char byte = 0;
    while (in.get(byte)) {
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    if (bytes.size() < 54U || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::runtime_error("Only BMP masks are supported by this diagnostic without OpenCV.");
    }

    const auto data_offset = read_u32(bytes, 10);
    const auto dib_size = read_u32(bytes, 14);
    if (dib_size < 40U) {
        throw std::runtime_error("Unsupported BMP DIB header.");
    }

    const int width = read_i32(bytes, 18);
    const int signed_height = read_i32(bytes, 22);
    const int height = std::abs(signed_height);
    const bool top_down = signed_height < 0;
    const auto bpp = read_u16(bytes, 28);
    const auto compression = read_u32(bytes, 30);
    if (width <= 0 || height <= 0 || compression != 0U) {
        throw std::runtime_error("Unsupported BMP mask layout.");
    }
    if (bpp != 8U && bpp != 24U && bpp != 32U) {
        throw std::runtime_error("Only 8/24/32-bit BMP masks are supported.");
    }

    const int bytes_per_pixel = static_cast<int>(bpp / 8U);
    const int row_stride = ((width * static_cast<int>(bpp) + 31) / 32) * 4;
    std::vector<bool> valid(static_cast<std::size_t>(width * height), false);
    for (int y = 0; y < height; ++y) {
        const int src_y = top_down ? y : height - 1 - y;
        const auto row_offset = static_cast<std::size_t>(data_offset + src_y * row_stride);
        for (int x = 0; x < width; ++x) {
            const auto pixel_offset = row_offset + static_cast<std::size_t>(x * bytes_per_pixel);
            unsigned char gray = 0;
            if (bpp == 8U) {
                gray = bytes[pixel_offset];
            } else {
                const double b = static_cast<double>(bytes[pixel_offset]);
                const double g = static_cast<double>(bytes[pixel_offset + 1U]);
                const double r = static_cast<double>(bytes[pixel_offset + 2U]);
                gray = static_cast<unsigned char>(0.114 * b + 0.587 * g + 0.299 * r);
            }
            valid[static_cast<std::size_t>(y * width + x)] = gray > 0U;
        }
    }

    return dic::Mask(width, height, std::move(valid));
}

void write_boundary_csv(
    const std::filesystem::path& path,
    const std::vector<dic::mesh::BoundaryLoop>& loops)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "loop,point,x,y\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t loop_id = 0; loop_id < loops.size(); ++loop_id) {
        const auto& points = loops[loop_id].points;
        for (std::size_t i = 0; i < points.size(); ++i) {
            out << loop_id << "," << i << ","
                << points[i].x() << "," << points[i].y() << "\n";
        }
    }
}

void write_loop_txt_files(
    const std::filesystem::path& out_dir,
    const std::vector<dic::mesh::BoundaryLoop>& loops)
{
    for (std::size_t loop_id = 0; loop_id < loops.size(); ++loop_id) {
        const auto path = out_dir / ("boundary_loop_" + std::to_string(loop_id) + ".txt");
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Failed to write " + path.string());
        }
        out << "# x, y\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& point : loops[loop_id].points) {
            out << point.x() << ", " << point.y() << "\n";
        }
    }
}

void write_svg(
    const std::filesystem::path& path,
    const dic::Mask& mask,
    const std::vector<dic::mesh::BoundaryLoop>& loops)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << mask.width()
        << "\" height=\"" << mask.height() << "\" viewBox=\"0 0 " << mask.width()
        << " " << mask.height() << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << mask.width()
        << "\" height=\"" << mask.height()
        << "\" fill=\"none\" stroke=\"#cccccc\" stroke-width=\"1\"/>\n";

    static const char* colors[] = {"#d62728", "#1f77b4", "#2ca02c", "#9467bd"};
    for (std::size_t loop_id = 0; loop_id < loops.size(); ++loop_id) {
        const auto& points = loops[loop_id].points;
        if (points.empty()) {
            continue;
        }
        out << "<polyline fill=\"none\" stroke=\"" << colors[loop_id % 4]
            << "\" stroke-width=\"1.5\" points=\"";
        for (const auto& point : points) {
            out << point.x() << "," << point.y() << " ";
        }
        out << points.front().x() << "," << points.front().y()
            << "\"/>\n";
    }
    out << "</svg>\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: mesh_boundary_diagnostic <roi_mask.bmp> <out_dir>\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path mask_path = argv[1];
        const std::filesystem::path out_dir = argv[2];
        std::filesystem::create_directories(out_dir);

        dic::Mask mask = read_bmp_mask(mask_path);
        auto loops = dic::mesh::extract_boundary_loops(mask);
        write_boundary_csv(out_dir / "boundary_loops.csv", loops);
        write_loop_txt_files(out_dir, loops);
        write_svg(out_dir / "boundary_preview.svg", mask, loops);

        std::ofstream summary(out_dir / "summary.txt");
        summary << "mask_width=" << mask.width() << "\n";
        summary << "mask_height=" << mask.height() << "\n";
        summary << "loops=" << loops.size() << "\n";
        for (std::size_t i = 0; i < loops.size(); ++i) {
            summary << "loop_" << i << "_points=" << loops[i].points.size() << "\n";
        }

        std::cout << "Wrote boundary diagnostics to " << out_dir.string() << "\n";
        std::cout << "Loops: " << loops.size() << "\n";
        for (std::size_t i = 0; i < loops.size(); ++i) {
            std::cout << "  loop " << i << ": " << loops[i].points.size() << " points\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_boundary_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
