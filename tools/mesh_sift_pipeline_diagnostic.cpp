#include <dic/core/image.hpp>
#include <dic/initialization/feature_matcher.hpp>
#include <dic/initialization/sift_initializer.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <dic/mesh/mesh_generation_config.hpp>

#include "coordinate/g2l_internal.hpp"
#include "element/shape_func_internal.hpp"
#include "generation/inform_builder.hpp"
#include "solver/fem_assembler.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class SolverKind {
    ICGN,
    ForwardGN
};

enum class InitKind {
    SIFT,
    Zero
};

struct Stats {
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double stddev{0.0};
};

dic::mesh::MeshElementType parse_element_type(const std::string& value)
{
    if (value == "T3" || value == "t3") return dic::mesh::MeshElementType::T3;
    if (value == "Q8" || value == "q8") return dic::mesh::MeshElementType::Q8;
    return dic::mesh::MeshElementType::Q4;
}

SolverKind parse_solver_kind(const std::string& value)
{
    if (value == "forward_gn" || value == "forward_gauss_newton" || value == "fgn") {
        return SolverKind::ForwardGN;
    }
    return SolverKind::ICGN;
}

std::string solver_name(SolverKind solver)
{
    return solver == SolverKind::ForwardGN ? "forward_gauss_newton" : "global_icgn";
}

InitKind parse_init_kind(const std::string& value)
{
    if (value == "zero" || value == "none" || value == "0") {
        return InitKind::Zero;
    }
    return InitKind::SIFT;
}

std::string init_name(InitKind init)
{
    return init == InitKind::Zero ? "zero" : "sift";
}

int element_stride(dic::mesh::MeshElementType type)
{
    return type == dic::mesh::MeshElementType::Q8 ? 9 : dic::mesh::internal::nodes_per_element(type);
}

void read_nodes(const std::filesystem::path& path, std::vector<double>& coords)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open nodes file: " + path.string());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int id = 0;
        double x = 0.0;
        double y = 0.0;
        if (iss >> id >> x >> y) {
            coords.push_back(x);
            coords.push_back(y);
        }
    }
}

void read_elements(const std::filesystem::path& path,
                   dic::mesh::MeshElementType type,
                   std::vector<int>& elements,
                   int& n_elements)
{
    const int nn = dic::mesh::internal::nodes_per_element(type);
    const int stride = element_stride(type);
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open elements file: " + path.string());
    std::string line;
    n_elements = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int id = 0;
        iss >> id;
        for (int i = 0; i < nn; ++i) {
            int node = 0;
            iss >> node;
            elements.push_back(node);
        }
        for (int i = nn; i < stride; ++i) {
            elements.push_back(0);
        }
        ++n_elements;
    }
}

Stats compute_stats(const std::vector<double>& values)
{
    Stats stats;
    if (values.empty()) return stats;
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double var = 0.0;
    for (double v : values) {
        const double d = v - stats.mean;
        var += d * d;
    }
    stats.stddev = std::sqrt(var / static_cast<double>(values.size()));
    return stats;
}

std::vector<double> image_values(const dic::Image& image)
{
    std::vector<double> values;
    values.reserve(image.size());
    for (float value : image.data()) values.push_back(static_cast<double>(value));
    return values;
}

std::vector<double> matrix_values(const Eigen::MatrixXd& matrix)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(matrix.size()));
    for (int r = 0; r < matrix.rows(); ++r) {
        for (int c = 0; c < matrix.cols(); ++c) values.push_back(matrix(r, c));
    }
    return values;
}

void write_stats(std::ofstream& out, const std::string& name, const Stats& stats)
{
    out << name << "_min=" << stats.min << "\n";
    out << name << "_max=" << stats.max << "\n";
    out << name << "_mean=" << stats.mean << "\n";
    out << name << "_std=" << stats.stddev << "\n";
}

void write_u_csv(const std::filesystem::path& path,
                 const std::vector<double>& nodes,
                 const Eigen::VectorXd& U)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "node,x,y,u,v,mag\n";
    out << std::setprecision(12);
    const int n = static_cast<int>(nodes.size() / 2U);
    for (int i = 0; i < n; ++i) {
        const double u = U(2 * i);
        const double v = U(2 * i + 1);
        out << (i + 1) << "," << nodes[2 * i] << "," << nodes[2 * i + 1] << ","
            << u << "," << v << "," << std::hypot(u, v) << "\n";
    }
}

void write_quiver_svg(const std::filesystem::path& path,
                      const std::vector<double>& nodes,
                      const Eigen::VectorXd& U,
                      int width,
                      int height)
{
    double max_mag = 0.0;
    const int n = static_cast<int>(nodes.size() / 2U);
    for (int i = 0; i < n; ++i) {
        max_mag = std::max(max_mag, std::hypot(U(2 * i), U(2 * i + 1)));
    }
    const double scale = max_mag > 0.0 ? 25.0 / max_mag : 1.0;

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" "
        << "orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,6 L7,3 z\" fill=\"#1f77b4\"/></marker></defs>\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"none\" stroke=\"#cccccc\"/>\n";
    out << "<g stroke=\"#1f77b4\" stroke-width=\"1.1\" marker-end=\"url(#arrow)\">\n";
    for (int i = 0; i < n; ++i) {
        const double x = nodes[2 * i];
        const double y = nodes[2 * i + 1];
        out << "<line x1=\"" << x << "\" y1=\"" << y
            << "\" x2=\"" << x + U(2 * i) * scale
            << "\" y2=\"" << y + U(2 * i + 1) * scale << "\"/>\n";
    }
    out << "</g><g fill=\"#2ca02c\">\n";
    for (int i = 0; i < n; ++i) {
        out << "<circle cx=\"" << nodes[2 * i] << "\" cy=\"" << nodes[2 * i + 1] << "\" r=\"2\"/>\n";
    }
    out << "</g></svg>\n";
}

void write_element_debug(const std::filesystem::path& path,
                         const dic::mesh::internal::G2LOutput& g2l,
                         dic::mesh::MeshElementType type,
                         int max_samples)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "x,y,elem_id,xi,eta,N_sum,N_min,N_max,J_det\n";
    out << std::setprecision(12);
    const int nn = dic::mesh::internal::nodes_per_element(type);
    std::vector<double> N(static_cast<std::size_t>(nn), 0.0);
    std::vector<double> dxi(static_cast<std::size_t>(nn), 0.0);
    std::vector<double> deta(static_cast<std::size_t>(nn), 0.0);
    int written = 0;
    for (int y = 0; y < g2l.img_h && written < max_samples; ++y) {
        for (int x = 0; x < g2l.img_w && written < max_samples; ++x) {
            const int idx = y * g2l.img_w + x;
            if (!g2l.valid[static_cast<std::size_t>(idx)]) continue;
            dic::mesh::internal::shape_functions(
                type,
                g2l.xi[static_cast<std::size_t>(idx)],
                g2l.eta[static_cast<std::size_t>(idx)],
                N.data(), dxi.data(), deta.data());
            const double n_sum = std::accumulate(N.begin(), N.end(), 0.0);
            const auto [n_min, n_max] = std::minmax_element(N.begin(), N.end());
            const double det = g2l.J11[idx] * g2l.J22[idx] - g2l.J12[idx] * g2l.J21[idx];
            out << x << "," << y << "," << g2l.elem_id[idx] << ","
                << g2l.xi[idx] << "," << g2l.eta[idx] << ","
                << n_sum << "," << *n_min << "," << *n_max << "," << det << "\n";
            ++written;
        }
    }
}

void write_jacobian_stats(const std::filesystem::path& path,
                          const dic::mesh::internal::G2LOutput& g2l)
{
    std::vector<double> dets;
    for (std::size_t i = 0; i < g2l.valid.size(); ++i) {
        if (!g2l.valid[i]) continue;
        dets.push_back(g2l.J11[i] * g2l.J22[i] - g2l.J12[i] * g2l.J21[i]);
    }
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "valid_pixels=" << dets.size() << "\n";
    write_stats(out, "jacobian_det", compute_stats(dets));
}

Eigen::VectorXd compute_sift_initial_u(const dic::Image& reference,
                                       const dic::Image& deformed,
                                       const std::vector<double>& nodes)
{
    dic::FeatureMatcherConfig matcher_config;
    matcher_config.max_features = 4000;
    matcher_config.ratio_threshold = 0.75;
    matcher_config.robust_mad_factor = 5.0;
    const dic::FeatureMatcher matcher(matcher_config);
    const auto matches = matcher.match(reference, deformed);

    dic::SIFTInitializerConfig init_config;
    init_config.matcher = matcher_config;
    init_config.interpolation_neighbors = 8;
    init_config.interpolation_radius = 180.0;
    const dic::SIFTInitializer initializer(init_config);

    const int n = static_cast<int>(nodes.size() / 2U);
    Eigen::VectorXd U = Eigen::VectorXd::Zero(2 * n);
    for (int i = 0; i < n; ++i) {
        const auto initial = initializer.estimate_from_matches(matches, {nodes[2 * i], nodes[2 * i + 1]});
        if (initial.valid) {
            U(2 * i) = initial.u;
            U(2 * i + 1) = initial.v;
        }
    }
    return U;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr << "Usage: mesh_sift_pipeline_diagnostic <ref.bmp> <def.bmp>"
                  << " <nodes.txt> <elements.txt> <out_dir> <element_type>"
                  << " [max_iterations=5] [pixel_stride=1] [solver=icgn|forward_gn]"
                  << " [init=sift|zero] [tolerance=1e-3]\n";
        return EXIT_FAILURE;
    }
    try {
        const std::filesystem::path ref_path = argv[1];
        const std::filesystem::path def_path = argv[2];
        const std::filesystem::path nodes_path = argv[3];
        const std::filesystem::path elements_path = argv[4];
        const std::filesystem::path out_dir = argv[5];
        const auto element_type = parse_element_type(argv[6]);
        std::filesystem::create_directories(out_dir);

        const dic::Image reference(ref_path.string());
        const dic::Image deformed(def_path.string());
        if (reference.width() != deformed.width() || reference.height() != deformed.height()) {
            throw std::runtime_error("Image dimensions do not match.");
        }
        const int img_w = reference.width();
        const int img_h = reference.height();

        std::vector<double> nodes;
        std::vector<int> elements;
        int n_elements = 0;
        read_nodes(nodes_path, nodes);
        read_elements(elements_path, element_type, elements, n_elements);
        const int n_nodes = static_cast<int>(nodes.size() / 2U);

        dic::BSplinePrecomputeConfig precompute(dic::BSplineDegree::Quintic, 3, false, false);
        dic::BSplineImagePreprocessor preprocessor(precompute);
        auto ref_precomp = preprocessor.compute_lazy(reference);
        auto def_precomp = preprocessor.compute_lazy(deformed);
        dic::BSplineInterpolator def_interp(&def_precomp);

        ref_precomp.gradient_x = Eigen::MatrixXd::Zero(img_h, img_w);
        ref_precomp.gradient_y = Eigen::MatrixXd::Zero(img_h, img_w);
        for (int y = 0; y < img_h; ++y) {
            for (int x = 0; x < img_w; ++x) {
                if (x > 0 && x < img_w - 1)
                    ref_precomp.gradient_x(y, x) = 0.5 * (reference.at(x + 1, y) - reference.at(x - 1, y));
                if (y > 0 && y < img_h - 1)
                    ref_precomp.gradient_y(y, x) = 0.5 * (reference.at(x, y + 1) - reference.at(x, y - 1));
            }
        }

        std::vector<double> ref_flat(static_cast<std::size_t>(img_w * img_h));
        std::vector<double> fx_flat(static_cast<std::size_t>(img_w * img_h));
        std::vector<double> fy_flat(static_cast<std::size_t>(img_w * img_h));
        for (int y = 0; y < img_h; ++y) {
            for (int x = 0; x < img_w; ++x) {
                const int idx = y * img_w + x;
                ref_flat[idx] = reference.at(x, y);
                fx_flat[idx] = ref_precomp.gradient_x(y, x);
                fy_flat[idx] = ref_precomp.gradient_y(y, x);
            }
        }

        auto inform = dic::mesh::build_inform(
            nodes.data(), n_nodes, elements.data(), n_elements, element_type, img_h, img_w);
        const int n_pixels = static_cast<int>(inform.size() / 3U);

        dic::mesh::internal::G2LParams g2l_params;
        g2l_params.max_iter = 200;
        auto g2l = dic::mesh::internal::compute_global_to_local(
            inform.data(), n_pixels, nodes.data(), n_nodes,
            elements.data(), n_elements, img_h, img_w, element_type, g2l_params);

        const int max_iterations = argc >= 8 ? std::atoi(argv[7]) : 5;
        const int pixel_stride = argc >= 9 ? std::max(1, std::atoi(argv[8])) : 1;
        const SolverKind solver_kind = argc >= 10 ? parse_solver_kind(argv[9]) : SolverKind::ICGN;
        const InitKind init_kind = argc >= 11 ? parse_init_kind(argv[10]) : InitKind::SIFT;
        const double tolerance = argc >= 12 ? std::atof(argv[11]) : 1e-3;
        if (pixel_stride > 1) {
            for (int y = 0; y < g2l.img_h; ++y) {
                for (int x = 0; x < g2l.img_w; ++x) {
                    if ((x % pixel_stride) != 0 || (y % pixel_stride) != 0) {
                        const int idx = y * g2l.img_w + x;
                        g2l.valid[static_cast<std::size_t>(idx)] = 0;
                    }
                }
            }
        }

        auto cache = dic::mesh::internal::assemble_stiffness(
            g2l, img_h, img_w, fx_flat.data(), fy_flat.data(),
            n_nodes, elements.data(), n_elements, element_type, 0.0);

        Eigen::VectorXd U_initial = init_kind == InitKind::Zero
            ? Eigen::VectorXd::Zero(2 * n_nodes)
            : compute_sift_initial_u(reference, deformed, nodes);
        Eigen::VectorXd U_final = U_initial;
        const int iterations = solver_kind == SolverKind::ForwardGN
            ? dic::mesh::internal::global_forward_gn(
                cache, g2l, ref_flat.data(), img_h, img_w,
                elements.data(), n_elements, U_final, &def_interp, 0.0, tolerance, max_iterations)
            : dic::mesh::internal::global_icgn(
                cache, g2l, ref_flat.data(), img_h, img_w,
                elements.data(), n_elements, U_final, &def_interp, 0.0, tolerance, max_iterations);

        write_u_csv(out_dir / "initial_U.csv", nodes, U_initial);
        write_u_csv(out_dir / "final_U.csv", nodes, U_final);
        write_u_csv(out_dir / "delta_U.csv", nodes, U_final - U_initial);
        write_quiver_svg(out_dir / "initial_U_quiver.svg", nodes, U_initial, img_w, img_h);
        write_quiver_svg(out_dir / "final_U_quiver.svg", nodes, U_final, img_w, img_h);
        write_quiver_svg(out_dir / "delta_U_quiver.svg", nodes, U_final - U_initial, img_w, img_h);

        {
            std::ofstream out(out_dir / "image_preprocess_summary.txt");
            out << "image_width=" << img_w << "\n";
            out << "image_height=" << img_h << "\n";
            write_stats(out, "reference", compute_stats(image_values(reference)));
            write_stats(out, "deformed", compute_stats(image_values(deformed)));
        }
        {
            std::ofstream out(out_dir / "gradient_stats.txt");
            write_stats(out, "fx", compute_stats(matrix_values(ref_precomp.gradient_x)));
            write_stats(out, "fy", compute_stats(matrix_values(ref_precomp.gradient_y)));
        }
        {
            std::ofstream out(out_dir / "interpolation_config.txt");
            out << "bspline_degree=5\n";
            out << "border=3\n";
            out << "use_exact_prefilter=0\n";
            out << "precompute_local_blocks=0\n";
            out << "deformed_interpolator=BSplineInterpolator(compute_lazy)\n";
            out << "reference_gradient=center_difference\n";
        }
        write_element_debug(out_dir / "element_debug_samples.csv", g2l, element_type, 2000);
        write_jacobian_stats(out_dir / "jacobian_stats.txt", g2l);
        {
            std::ofstream out(out_dir / "solver_summary.txt");
            out << "nodes=" << n_nodes << "\n";
            out << "elements=" << n_elements << "\n";
            out << "inform_pixels=" << n_pixels << "\n";
            int valid_solver_pixels = 0;
            for (auto flag : g2l.valid) valid_solver_pixels += flag ? 1 : 0;
            out << "solver=" << solver_name(solver_kind) << "\n";
            out << "initialization=" << init_name(init_kind) << "\n";
            out << "solver_valid_pixels=" << valid_solver_pixels << "\n";
            out << "pixel_stride=" << pixel_stride << "\n";
            out << "iterations=" << iterations << "\n";
            out << "tolerance=" << tolerance << "\n";
            out << "max_iterations=" << max_iterations << "\n";
            out << "regularization_alpha=0\n";
        }

        std::cout << "Wrote mesh SIFT pipeline diagnostics to " << out_dir.string() << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_sift_pipeline_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
