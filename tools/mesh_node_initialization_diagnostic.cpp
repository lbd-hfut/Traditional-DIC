#include <dic/core/image.hpp>
#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/subset_initializer.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/subset/solver/icgn.hpp>
#include <dic/subset/subset_config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct NodeRecord {
    int id{0};
    double x{0.0};
    double y{0.0};
};

struct InitRecord {
    int id{0};
    double x{0.0};
    double y{0.0};
    bool integer_valid{false};
    double integer_u{0.0};
    double integer_v{0.0};
    double integer_confidence{0.0};
    bool refined_valid{false};
    double refined_u{0.0};
    double refined_v{0.0};
    double refined_confidence{0.0};
};

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

std::vector<unsigned char> read_file_bytes(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open: " + path.string());
    }
    std::vector<unsigned char> bytes;
    char byte = 0;
    while (in.get(byte)) {
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    return bytes;
}

dic::Image read_bmp_image(const std::filesystem::path& path)
{
    const auto bytes = read_file_bytes(path);
    if (bytes.size() < 54U || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::runtime_error("Only BMP images are supported by this diagnostic.");
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
        throw std::runtime_error("Unsupported BMP layout.");
    }
    if (bpp != 8U && bpp != 24U && bpp != 32U) {
        throw std::runtime_error("Only 8/24/32-bit BMP images are supported.");
    }

    const int bytes_per_pixel = static_cast<int>(bpp / 8U);
    const int row_stride = ((width * static_cast<int>(bpp) + 31) / 32) * 4;
    std::vector<float> pixels(static_cast<std::size_t>(width * height), 0.0F);
    for (int y = 0; y < height; ++y) {
        const int src_y = top_down ? y : height - 1 - y;
        const auto row_offset = static_cast<std::size_t>(data_offset + src_y * row_stride);
        for (int x = 0; x < width; ++x) {
            const auto pixel_offset = row_offset + static_cast<std::size_t>(x * bytes_per_pixel);
            double gray = 0.0;
            if (bpp == 8U) {
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

std::vector<NodeRecord> read_nodes(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open nodes file: " + path.string());
    }

    std::vector<NodeRecord> nodes;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        NodeRecord node;
        if (iss >> node.id >> node.x >> node.y) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

void write_csv(const std::filesystem::path& path, const std::vector<InitRecord>& records)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "node,x,y,integer_valid,integer_u,integer_v,integer_confidence,"
        << "refined_valid,refined_u,refined_v,refined_confidence\n";
    out << std::setprecision(12);
    for (const auto& r : records) {
        out << r.id << "," << r.x << "," << r.y << ","
            << (r.integer_valid ? 1 : 0) << "," << r.integer_u << "," << r.integer_v << ","
            << r.integer_confidence << ","
            << (r.refined_valid ? 1 : 0) << "," << r.refined_u << "," << r.refined_v << ","
            << r.refined_confidence << "\n";
    }
}

void write_summary(
    const std::filesystem::path& path,
    const std::vector<InitRecord>& records,
    bool run_subpixel,
    const std::string& refine_mode,
    const std::string& precompute_mode)
{
    int integer_valid = 0;
    int refined_valid = 0;
    double sum_u = 0.0;
    double sum_v = 0.0;
    double max_mag = 0.0;
    for (const auto& r : records) {
        if (r.integer_valid) {
            ++integer_valid;
        }
        if (r.refined_valid) {
            ++refined_valid;
            sum_u += r.refined_u;
            sum_v += r.refined_v;
            max_mag = std::max(max_mag, std::sqrt(r.refined_u * r.refined_u + r.refined_v * r.refined_v));
        }
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "nodes=" << records.size() << "\n";
    out << "subpixel_enabled=" << (run_subpixel ? 1 : 0) << "\n";
    out << "refine_mode=" << refine_mode << "\n";
    out << "precompute_mode=" << precompute_mode << "\n";
    if (!run_subpixel) {
        out << "note=refined fields mirror integer results because subpixel refinement is disabled\n";
    }
    out << "integer_valid=" << integer_valid << "\n";
    out << "integer_invalid=" << static_cast<int>(records.size()) - integer_valid << "\n";
    out << "refined_valid=" << refined_valid << "\n";
    out << "refined_invalid=" << static_cast<int>(records.size()) - refined_valid << "\n";
    if (refined_valid > 0) {
        out << "mean_refined_u=" << sum_u / refined_valid << "\n";
        out << "mean_refined_v=" << sum_v / refined_valid << "\n";
    }
    out << "max_refined_magnitude=" << max_mag << "\n";
}

void write_svg(
    const std::filesystem::path& path,
    const std::vector<InitRecord>& records,
    int width,
    int height)
{
    double max_mag = 0.0;
    for (const auto& r : records) {
        if (r.refined_valid) {
            max_mag = std::max(max_mag, std::sqrt(r.refined_u * r.refined_u + r.refined_v * r.refined_v));
        }
    }
    const double arrow_scale = max_mag > 0.0 ? 25.0 / max_mag : 1.0;

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" "
        << "orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,6 L7,3 z\" fill=\"#1f77b4\"/></marker></defs>\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"none\" stroke=\"#cccccc\"/>\n";
    out << "<g stroke-width=\"1.2\" stroke=\"#1f77b4\" marker-end=\"url(#arrow)\">\n";
    for (const auto& r : records) {
        if (!r.refined_valid) {
            continue;
        }
        const double x2 = r.x + r.refined_u * arrow_scale;
        const double y2 = r.y + r.refined_v * arrow_scale;
        out << "<line x1=\"" << r.x << "\" y1=\"" << r.y
            << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g>\n";
    for (const auto& r : records) {
        if (r.refined_valid) {
            out << "<circle cx=\"" << r.x << "\" cy=\"" << r.y
                << "\" r=\"2.2\" fill=\"#2ca02c\"/>\n";
        } else if (r.integer_valid) {
            out << "<circle cx=\"" << r.x << "\" cy=\"" << r.y
                << "\" r=\"2.2\" fill=\"#ffbf00\"/>\n";
        } else {
            out << "<circle cx=\"" << r.x << "\" cy=\"" << r.y
                << "\" r=\"3\" fill=\"#d62728\"/>\n";
        }
    }
    out << "</g>\n</svg>\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::cerr << "Usage: mesh_node_initialization_diagnostic <ref.bmp> <def.bmp>"
                  << " <nodes.txt> <out_dir> <label> [subset_radius=10] [search_radius=30]"
                  << " [subpixel_radius=15] [run_subpixel=1] [max_nodes=0]"
                  << " [refine_mode=subset_initializer|direct_icgn]"
                  << " [precompute_mode=lazy|full]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path ref_path = argv[1];
        const std::filesystem::path def_path = argv[2];
        const std::filesystem::path nodes_path = argv[3];
        const std::filesystem::path out_dir = argv[4];
        const std::string label = argv[5];
        const int subset_radius = argc >= 7 ? std::atoi(argv[6]) : 10;
        const int search_radius = argc >= 8 ? std::atoi(argv[7]) : 30;
        const int subpixel_radius = argc >= 9 ? std::atoi(argv[8]) : 15;
        const bool run_subpixel = argc >= 10 ? std::atoi(argv[9]) != 0 : true;
        const int max_nodes = argc >= 11 ? std::atoi(argv[10]) : 0;
        const std::string refine_mode = argc >= 12 ? argv[11] : "subset_initializer";
        const std::string precompute_mode = argc >= 13 ? argv[12] : "lazy";
        if (refine_mode != "subset_initializer" && refine_mode != "direct_icgn") {
            throw std::runtime_error("refine_mode must be subset_initializer or direct_icgn.");
        }
        if (precompute_mode != "lazy" && precompute_mode != "full") {
            throw std::runtime_error("precompute_mode must be lazy or full.");
        }

        std::filesystem::create_directories(out_dir);
        const auto reference = read_bmp_image(ref_path);
        const auto deformed = read_bmp_image(def_path);
        const auto nodes = read_nodes(nodes_path);

        dic::SubsetConfig config;
        config.image_precompute = dic::BSplinePrecomputeConfig(dic::BSplineDegree::Quintic, 3, false);
        config.seed_initialization.integer_search.subset_radius = subset_radius;
        config.seed_initialization.integer_search.search_radius = search_radius;
        config.seed_initialization.integer_search.pyramid_enabled = true;
        config.seed_initialization.subpixel.enabled = true;
        config.seed_initialization.subpixel.subset_radius = subpixel_radius;
        config.seed_initialization.subpixel.max_iterations = 30;
        config.seed_initialization.subpixel.convergence_threshold = 1e-3;
        config.seed_initialization.subpixel.optimizer = dic::SubsetOptimizationMethod::ICGN;
        config.seed_initialization.subpixel.shape_function = dic::SubsetShapeFunctionMethod::FirstOrder;

        dic::BSplineInterpolator reference_interpolator(nullptr);
        dic::BSplinePrecomputedImage deformed_precomputed;
        if (run_subpixel) {
            auto precompute_config = config.image_precompute;
            precompute_config.precompute_local_blocks = precompute_mode == "full";
            dic::BSplineImagePreprocessor preprocessor(precompute_config);
            deformed_precomputed = precompute_mode == "full"
                ? preprocessor.compute(deformed)
                : preprocessor.compute_lazy(deformed);
        }
        dic::BSplineInterpolator deformed_interpolator(
            run_subpixel ? &deformed_precomputed : nullptr);

        const dic::IntegerSearchInitializer integer_search(config.seed_initialization, config.image_precompute);
        const dic::SubsetInitializer subset_initializer(config);
        dic::SubsetConfig solver_config = config;
        solver_config.subset_radius = config.seed_initialization.subpixel.subset_radius;
        solver_config.search_radius = config.seed_initialization.integer_search.search_radius;
        solver_config.convergence_threshold = config.seed_initialization.subpixel.convergence_threshold;
        solver_config.max_iterations = config.seed_initialization.subpixel.max_iterations;
        solver_config.shape_function = config.seed_initialization.subpixel.shape_function;
        solver_config.optimizer = config.seed_initialization.subpixel.optimizer;
        solver_config.use_second_order =
            config.seed_initialization.subpixel.shape_function == dic::SubsetShapeFunctionMethod::SecondOrder;
        const dic::ICGNSolver direct_solver(solver_config);

        std::vector<InitRecord> records;
        records.reserve(nodes.size());
        const std::size_t n_limit = max_nodes > 0
            ? std::min(nodes.size(), static_cast<std::size_t>(max_nodes))
            : nodes.size();
        for (std::size_t node_index = 0; node_index < n_limit; ++node_index) {
            const auto& node = nodes[node_index];
            const Eigen::Vector2d point{node.x, node.y};
            const auto integer = integer_search.estimate(reference, deformed, point);
            dic::InitialDisplacement refined = integer;
            if (run_subpixel && integer.valid) {
                if (refine_mode == "direct_icgn") {
                    const auto solved = direct_solver.solve_with_interpolators(
                        reference, deformed, point, integer, reference_interpolator, deformed_interpolator);
                    if (solved.valid && solved.status == dic::SolverStatus::Success) {
                        refined = {solved.u, solved.v,
                                   solved.du_dx, solved.du_dy,
                                   solved.dv_dx, solved.dv_dy,
                                   solved.correlation, true};
                    }
                } else {
                    refined = subset_initializer.estimate_with_interpolators(
                        reference, deformed, point, reference_interpolator, deformed_interpolator);
                }
            }
            records.push_back(InitRecord{
                node.id, node.x, node.y,
                integer.valid, integer.u, integer.v, integer.confidence,
                refined.valid, refined.u, refined.v, refined.confidence
            });
        }

        write_csv(out_dir / (label + "_node_initialization.csv"), records);
        write_summary(out_dir / (label + "_summary.txt"), records, run_subpixel, refine_mode, precompute_mode);
        write_svg(out_dir / (label + "_initialization_quiver.svg"), records,
                  reference.width(), reference.height());

        std::cout << "Wrote node initialization diagnostics to " << out_dir.string() << "\n";
        std::cout << "Nodes: " << records.size() << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_node_initialization_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
