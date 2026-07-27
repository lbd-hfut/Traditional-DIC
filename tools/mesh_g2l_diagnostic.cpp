#include <dic/mesh/mesh_generation_config.hpp>

#include "coordinate/g2l_internal.hpp"
#include "element/shape_func_internal.hpp"

#include <algorithm>
#include <cmath>
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

struct SampleRecord {
    int element_id{0};
    double xi_true{0.0};
    double eta_true{0.0};
    double x{0.0};
    double y{0.0};
    bool converged{false};
    double xi_solved{0.0};
    double eta_solved{0.0};
    double residual{0.0};
    double local_error{0.0};
    double det_j{0.0};
};

struct Summary {
    int elements{0};
    int samples{0};
    int failed{0};
    double max_residual{0.0};
    double mean_residual{0.0};
    double max_local_error{0.0};
    double min_abs_det_j{std::numeric_limits<double>::infinity()};
    double max_abs_det_j{0.0};
};

dic::mesh::MeshElementType parse_element_type(const std::string& value)
{
    if (value == "T3" || value == "t3") return dic::mesh::MeshElementType::T3;
    if (value == "Q8" || value == "q8") return dic::mesh::MeshElementType::Q8;
    return dic::mesh::MeshElementType::Q4;
}

std::string element_type_name(dic::mesh::MeshElementType type)
{
    switch (type) {
    case dic::mesh::MeshElementType::T3: return "T3";
    case dic::mesh::MeshElementType::Q4: return "Q4";
    case dic::mesh::MeshElementType::Q8: return "Q8";
    }
    return "Q4";
}

int nodes_per_element(dic::mesh::MeshElementType type)
{
    return dic::mesh::internal::nodes_per_element(type);
}

int stride_for(dic::mesh::MeshElementType type)
{
    return type == dic::mesh::MeshElementType::Q8 ? 9 : nodes_per_element(type);
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
        double x = 0.0, y = 0.0;
        if (iss >> id >> x >> y) {
            coords.push_back(x);
            coords.push_back(y);
        }
    }
}

void read_elements(
    const std::filesystem::path& path,
    dic::mesh::MeshElementType type,
    std::vector<int>& elements,
    int& n_elements)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open elements file: " + path.string());
    const int nn = nodes_per_element(type);
    const int stride = stride_for(type);
    std::string line;
    n_elements = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int eid = 0;
        if (!(iss >> eid)) continue;
        for (int i = 0; i < nn; ++i) {
            int nid = 0;
            if (!(iss >> nid)) {
                throw std::runtime_error("Element row has too few node ids: " + line);
            }
            elements.push_back(nid);
        }
        if (stride > nn) {
            elements.push_back(0);
        }
        ++n_elements;
    }
}

std::vector<std::pair<double, double>> sample_points(dic::mesh::MeshElementType type)
{
    std::vector<std::pair<double, double>> samples;
    if (type == dic::mesh::MeshElementType::T3) {
        const double vals[] = {0.1, 0.3, 0.5, 0.7};
        for (double xi : vals) {
            for (double eta : vals) {
                if (xi + eta < 0.95) {
                    samples.push_back({xi, eta});
                }
            }
        }
        samples.push_back({1.0 / 3.0, 1.0 / 3.0});
        return samples;
    }

    const double vals[] = {-0.8, -0.4, 0.0, 0.4, 0.8};
    for (double xi : vals) {
        for (double eta : vals) {
            samples.push_back({xi, eta});
        }
    }
    return samples;
}

void element_nodes(
    const std::vector<double>& coords,
    const std::vector<int>& elements,
    int e,
    dic::mesh::MeshElementType type,
    std::vector<double>& elem_nodes)
{
    const int nn = nodes_per_element(type);
    const int stride = stride_for(type);
    elem_nodes.assign(static_cast<std::size_t>(2 * nn), 0.0);
    for (int i = 0; i < nn; ++i) {
        const int nid = elements[e * stride + i] - 1;
        elem_nodes[2 * i] = coords[2 * nid];
        elem_nodes[2 * i + 1] = coords[2 * nid + 1];
    }
}

void local_to_global_and_jacobian(
    dic::mesh::MeshElementType type,
    const std::vector<double>& elem_nodes,
    double xi,
    double eta,
    double& x,
    double& y,
    double& j11,
    double& j12,
    double& j21,
    double& j22)
{
    const int nn = nodes_per_element(type);
    std::vector<double> n(static_cast<std::size_t>(nn));
    std::vector<double> dxi(static_cast<std::size_t>(nn));
    std::vector<double> deta(static_cast<std::size_t>(nn));
    dic::mesh::internal::shape_functions(type, xi, eta, n.data(), dxi.data(), deta.data());

    x = 0.0; y = 0.0;
    j11 = 0.0; j12 = 0.0; j21 = 0.0; j22 = 0.0;
    for (int i = 0; i < nn; ++i) {
        const double nx = elem_nodes[2 * i];
        const double ny = elem_nodes[2 * i + 1];
        x += n[i] * nx;
        y += n[i] * ny;
        j11 += dxi[i] * nx;
        j12 += deta[i] * nx;
        j21 += dxi[i] * ny;
        j22 += deta[i] * ny;
    }
}

bool solve_global_to_local(
    dic::mesh::MeshElementType type,
    const std::vector<double>& elem_nodes,
    double x,
    double y,
    double& xi,
    double& eta,
    double& j11,
    double& j12,
    double& j21,
    double& j22)
{
    using namespace dic::mesh::internal;
    if (type == dic::mesh::MeshElementType::T3) {
        return solve_point_t3(x, y, elem_nodes.data(), xi, eta, j11, j12, j21, j22);
    }
    if (type == dic::mesh::MeshElementType::Q4) {
        return solve_point_q4(x, y, elem_nodes.data(), xi, eta, j11, j12, j21, j22, 200);
    }

    double xi0 = 0.0, eta0 = 0.0;
    double q4_j11 = 0.0, q4_j12 = 0.0, q4_j21 = 0.0, q4_j22 = 0.0;
    const bool seed_ok = solve_point_q4(
        x, y, elem_nodes.data(), xi0, eta0, q4_j11, q4_j12, q4_j21, q4_j22, 200);
    if (!seed_ok) {
        xi0 = 0.0;
        eta0 = 0.0;
    }
    bool ok = solve_point_q8(
        x, y, elem_nodes.data(), xi0, eta0, xi, eta, j11, j12, j21, j22, 1e-8, 200);
    if (!ok) {
        ok = solve_point_q8_fallback(
            x, y, elem_nodes.data(), xi0, eta0, xi, eta, j11, j12, j21, j22,
            1e-8, 1e-10, 200);
    }
    return ok;
}

void write_samples_csv(const std::filesystem::path& path, const std::vector<SampleRecord>& samples)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "element,xi_true,eta_true,x,y,converged,xi_solved,eta_solved,residual,local_error,det_j\n";
    out << std::setprecision(12);
    for (const auto& s : samples) {
        out << s.element_id << "," << s.xi_true << "," << s.eta_true << ","
            << s.x << "," << s.y << "," << (s.converged ? 1 : 0) << ","
            << s.xi_solved << "," << s.eta_solved << "," << s.residual << ","
            << s.local_error << "," << s.det_j << "\n";
    }
}

void write_summary(const std::filesystem::path& path, const Summary& summary)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "elements=" << summary.elements << "\n";
    out << "samples=" << summary.samples << "\n";
    out << "failed=" << summary.failed << "\n";
    out << "max_residual=" << summary.max_residual << "\n";
    out << "mean_residual=" << summary.mean_residual << "\n";
    out << "max_local_error=" << summary.max_local_error << "\n";
    out << "min_abs_det_j=" << summary.min_abs_det_j << "\n";
    out << "max_abs_det_j=" << summary.max_abs_det_j << "\n";
}

const char* residual_color(double residual, bool ok)
{
    if (!ok) return "#d62728";
    if (residual < 1e-9) return "#2ca02c";
    if (residual < 1e-6) return "#ffbf00";
    return "#ff7f0e";
}

void write_svg(
    const std::filesystem::path& path,
    const std::vector<double>& coords,
    const std::vector<int>& elements,
    int n_elements,
    dic::mesh::MeshElementType type,
    const std::vector<SampleRecord>& samples,
    int width,
    int height)
{
    const int corner_count = type == dic::mesh::MeshElementType::T3 ? 3 : 4;
    const int stride = stride_for(type);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write " + path.string());
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"none\" stroke=\"#cccccc\"/>\n";
    out << "<g fill=\"none\" stroke=\"#777777\" stroke-width=\"0.8\" opacity=\"0.75\">\n";
    for (int e = 0; e < n_elements; ++e) {
        out << "<polyline points=\"";
        for (int k = 0; k < corner_count; ++k) {
            const int nid = elements[e * stride + k] - 1;
            out << coords[2 * nid] << "," << coords[2 * nid + 1] << " ";
        }
        const int nid0 = elements[e * stride] - 1;
        out << coords[2 * nid0] << "," << coords[2 * nid0 + 1] << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g stroke=\"#333333\" stroke-width=\"0.3\">\n";
    for (const auto& s : samples) {
        out << "<circle cx=\"" << s.x << "\" cy=\"" << s.y
            << "\" r=\"1.8\" fill=\"" << residual_color(s.residual, s.converged) << "\"/>\n";
    }
    out << "</g>\n";
    out << "</svg>\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: mesh_g2l_diagnostic <nodes.txt> <elements.txt> <out_dir> <element_type>"
                  << " [image_width=1280] [image_height=1280]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path nodes_path = argv[1];
        const std::filesystem::path elements_path = argv[2];
        const std::filesystem::path out_dir = argv[3];
        const auto type = parse_element_type(argv[4]);
        const int width = argc >= 6 ? std::atoi(argv[5]) : 1280;
        const int height = argc >= 7 ? std::atoi(argv[6]) : 1280;
        std::filesystem::create_directories(out_dir);

        std::vector<double> coords;
        std::vector<int> elements;
        int n_elements = 0;
        read_nodes(nodes_path, coords);
        read_elements(elements_path, type, elements, n_elements);

        const auto local_samples = sample_points(type);
        std::vector<SampleRecord> records;
        records.reserve(static_cast<std::size_t>(n_elements) * local_samples.size());

        Summary summary;
        summary.elements = n_elements;
        std::vector<double> elem_nodes;
        for (int e = 0; e < n_elements; ++e) {
            element_nodes(coords, elements, e, type, elem_nodes);
            for (const auto& [xi_true, eta_true] : local_samples) {
                double x = 0.0, y = 0.0;
                double j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
                local_to_global_and_jacobian(type, elem_nodes, xi_true, eta_true, x, y, j11, j12, j21, j22);

                double xi = 0.0, eta = 0.0;
                double sj11 = 0.0, sj12 = 0.0, sj21 = 0.0, sj22 = 0.0;
                const bool ok = solve_global_to_local(type, elem_nodes, x, y, xi, eta, sj11, sj12, sj21, sj22);

                double xr = 0.0, yr = 0.0;
                double rj11 = 0.0, rj12 = 0.0, rj21 = 0.0, rj22 = 0.0;
                local_to_global_and_jacobian(type, elem_nodes, xi, eta, xr, yr, rj11, rj12, rj21, rj22);
                const double residual = std::sqrt((x - xr) * (x - xr) + (y - yr) * (y - yr));
                const double local_error = std::sqrt((xi_true - xi) * (xi_true - xi) +
                                                     (eta_true - eta) * (eta_true - eta));
                const double det_j = sj11 * sj22 - sj12 * sj21;

                records.push_back(SampleRecord{
                    e + 1, xi_true, eta_true, x, y, ok, xi, eta, residual, local_error, det_j
                });

                ++summary.samples;
                if (!ok || !std::isfinite(residual)) {
                    ++summary.failed;
                }
                if (std::isfinite(residual)) {
                    summary.max_residual = std::max(summary.max_residual, residual);
                    summary.mean_residual += residual;
                }
                summary.max_local_error = std::max(summary.max_local_error, local_error);
                const double abs_det = std::abs(det_j);
                summary.min_abs_det_j = std::min(summary.min_abs_det_j, abs_det);
                summary.max_abs_det_j = std::max(summary.max_abs_det_j, abs_det);
            }
        }
        if (summary.samples > 0) {
            summary.mean_residual /= static_cast<double>(summary.samples);
        }

        write_samples_csv(out_dir / "samples.csv", records);
        write_summary(out_dir / "summary.txt", summary);
        write_svg(out_dir / "g2l_error_preview.svg", coords, elements, n_elements, type, records, width, height);

        std::cout << "Wrote G2L diagnostics to " << out_dir.string() << "\n";
        std::cout << element_type_name(type) << ": samples=" << summary.samples
                  << ", failed=" << summary.failed
                  << ", max_residual=" << summary.max_residual
                  << ", max_local_error=" << summary.max_local_error << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_g2l_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
