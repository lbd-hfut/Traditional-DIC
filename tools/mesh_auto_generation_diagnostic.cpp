#include <dic/core/mask.hpp>
#include <dic/mesh/generation/boundary_exporter.hpp>
#include <dic/mesh/mesh_generation_config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct NodeRecord {
    int id{0};
    double x{0.0};
    double y{0.0};
};

struct ElementRecord {
    int id{0};
    std::vector<int> node_ids;
};

struct MeshData {
    std::string element_type;
    std::vector<NodeRecord> nodes;
    std::vector<ElementRecord> elements;
};

struct CircleEstimate {
    double cx{0.0};
    double cy{0.0};
    double radius{0.0};
};

struct QualitySummary {
    double min_area{std::numeric_limits<double>::infinity()};
    double min_edge{std::numeric_limits<double>::infinity()};
    double max_edge{0.0};
    double max_aspect_ratio{0.0};
    int degenerate{0};
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
        throw std::runtime_error("Only BMP masks are supported by this diagnostic.");
    }

    const auto data_offset = read_u32(bytes, 10);
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

CircleEstimate estimate_circle(const dic::mesh::BoundaryLoop& loop)
{
    CircleEstimate circle;
    for (const auto& p : loop.points) {
        circle.cx += p.x();
        circle.cy += p.y();
    }
    circle.cx /= static_cast<double>(loop.points.size());
    circle.cy /= static_cast<double>(loop.points.size());
    for (const auto& p : loop.points) {
        const double dx = p.x() - circle.cx;
        const double dy = p.y() - circle.cy;
        circle.radius += std::sqrt(dx * dx + dy * dy);
    }
    circle.radius /= static_cast<double>(loop.points.size());
    return circle;
}

int add_node(MeshData& mesh, double x, double y)
{
    const int id = static_cast<int>(mesh.nodes.size()) + 1;
    mesh.nodes.push_back(NodeRecord{id, x, y});
    return id;
}

double dist(const NodeRecord& a, const NodeRecord& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double polygon_area(const std::vector<NodeRecord>& poly)
{
    double area = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1U) % poly.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5 * area;
}

MeshData generate_annulus_q4(
    const CircleEstimate& center,
    double r_inner,
    double r_outer,
    int nr,
    int nt)
{
    MeshData mesh;
    mesh.element_type = "Q4";
    auto node_id = [nt](int ir, int it) { return ir * nt + (it % nt) + 1; };
    for (int ir = 0; ir <= nr; ++ir) {
        const double r = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
        for (int it = 0; it < nt; ++it) {
            const double theta = 2.0 * kPi * static_cast<double>(it) / nt;
            add_node(mesh, center.cx + r * std::cos(theta), center.cy + r * std::sin(theta));
        }
    }
    int eid = 1;
    for (int ir = 0; ir < nr; ++ir) {
        for (int it = 0; it < nt; ++it) {
            mesh.elements.push_back(ElementRecord{
                eid++,
                {node_id(ir, it), node_id(ir, it + 1), node_id(ir + 1, it + 1), node_id(ir + 1, it)}
            });
        }
    }
    return mesh;
}

MeshData generate_annulus_t3(
    const CircleEstimate& center,
    double r_inner,
    double r_outer,
    int nr,
    int nt)
{
    MeshData mesh = generate_annulus_q4(center, r_inner, r_outer, nr, nt);
    mesh.element_type = "T3";
    std::vector<ElementRecord> triangles;
    triangles.reserve(mesh.elements.size() * 2U);
    int eid = 1;
    for (const auto& q : mesh.elements) {
        triangles.push_back(ElementRecord{eid++, {q.node_ids[0], q.node_ids[1], q.node_ids[2]}});
        triangles.push_back(ElementRecord{eid++, {q.node_ids[0], q.node_ids[2], q.node_ids[3]}});
    }
    mesh.elements = std::move(triangles);
    return mesh;
}

MeshData generate_annulus_q8(
    const CircleEstimate& center,
    double r_inner,
    double r_outer,
    int nr,
    int nt)
{
    MeshData mesh;
    mesh.element_type = "Q8";
    auto corner_key = [nt](int ir, int it) { return ir * nt + (it % nt); };
    std::vector<int> corner_ids(static_cast<std::size_t>((nr + 1) * nt), 0);
    for (int ir = 0; ir <= nr; ++ir) {
        const double r = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
        for (int it = 0; it < nt; ++it) {
            const double theta = 2.0 * kPi * static_cast<double>(it) / nt;
            corner_ids[static_cast<std::size_t>(corner_key(ir, it))] =
                add_node(mesh, center.cx + r * std::cos(theta), center.cy + r * std::sin(theta));
        }
    }

    std::map<std::pair<int, int>, int> edge_mid_ids;
    auto midpoint_id = [&](int a, int b) {
        const auto key = std::minmax(a, b);
        const auto found = edge_mid_ids.find(key);
        if (found != edge_mid_ids.end()) {
            return found->second;
        }
        const auto& na = mesh.nodes[static_cast<std::size_t>(a - 1)];
        const auto& nb = mesh.nodes[static_cast<std::size_t>(b - 1)];
        const double mx = 0.5 * (na.x + nb.x);
        const double my = 0.5 * (na.y + nb.y);
        const int id = add_node(mesh, mx, my);
        edge_mid_ids[key] = id;
        return id;
    };
    auto arc_midpoint_id = [&](int a, int b, double radius, int it) {
        const auto key = std::minmax(a, b);
        const auto found = edge_mid_ids.find(key);
        if (found != edge_mid_ids.end()) {
            return found->second;
        }
        const double theta = 2.0 * kPi * (static_cast<double>(it) + 0.5) / nt;
        const int id = add_node(mesh,
                                center.cx + radius * std::cos(theta),
                                center.cy + radius * std::sin(theta));
        edge_mid_ids[key] = id;
        return id;
    };

    int eid = 1;
    for (int ir = 0; ir < nr; ++ir) {
        for (int it = 0; it < nt; ++it) {
            const int n0 = corner_ids[static_cast<std::size_t>(corner_key(ir, it))];
            const int n1 = corner_ids[static_cast<std::size_t>(corner_key(ir, it + 1))];
            const int n2 = corner_ids[static_cast<std::size_t>(corner_key(ir + 1, it + 1))];
            const int n3 = corner_ids[static_cast<std::size_t>(corner_key(ir + 1, it))];
            const double r0 = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
            const double r1 = r_inner + (r_outer - r_inner) * static_cast<double>(ir + 1) / nr;
            const int n4 = (ir == 0) ? arc_midpoint_id(n0, n1, r0, it) : midpoint_id(n0, n1);
            const int n5 = midpoint_id(n1, n2);
            const int n6 = (ir + 1 == nr) ? arc_midpoint_id(n2, n3, r1, it) : midpoint_id(n2, n3);
            const int n7 = midpoint_id(n3, n0);
            mesh.elements.push_back(ElementRecord{eid++, {n0, n1, n2, n3, n4, n5, n6, n7}});
        }
    }
    return mesh;
}

QualitySummary compute_quality(const MeshData& mesh)
{
    QualitySummary q;
    for (const auto& e : mesh.elements) {
        const int corner_count = mesh.element_type == "T3" ? 3 : 4;
        std::vector<NodeRecord> corners;
        corners.reserve(static_cast<std::size_t>(corner_count));
        for (int i = 0; i < corner_count; ++i) {
            corners.push_back(mesh.nodes[static_cast<std::size_t>(e.node_ids[i] - 1)]);
        }
        const double area = polygon_area(corners);
        if (std::abs(area) < 1e-12) {
            ++q.degenerate;
        }
        q.min_area = std::min(q.min_area, std::abs(area));
        double min_edge = std::numeric_limits<double>::infinity();
        double max_edge = 0.0;
        for (std::size_t i = 0; i < corners.size(); ++i) {
            const double edge = dist(corners[i], corners[(i + 1U) % corners.size()]);
            min_edge = std::min(min_edge, edge);
            max_edge = std::max(max_edge, edge);
        }
        q.min_edge = std::min(q.min_edge, min_edge);
        q.max_edge = std::max(q.max_edge, max_edge);
        if (min_edge > 0.0) {
            q.max_aspect_ratio = std::max(q.max_aspect_ratio, max_edge / min_edge);
        }
    }
    return q;
}

void write_mesh_files(const std::filesystem::path& out_dir, const MeshData& mesh)
{
    std::filesystem::create_directories(out_dir);
    {
        std::ofstream nodes(out_dir / ("nodes_" + mesh.element_type + ".txt"));
        nodes << std::fixed << std::setprecision(6);
        for (const auto& n : mesh.nodes) {
            nodes << n.id << ", " << n.x << ", " << n.y << "\n";
        }
    }
    {
        std::ofstream elements(out_dir / ("elements_" + mesh.element_type + ".txt"));
        for (const auto& e : mesh.elements) {
            elements << e.id;
            for (int nid : e.node_ids) {
                elements << ", " << nid;
            }
            elements << "\n";
        }
    }
}

void write_svg(const std::filesystem::path& path, const MeshData& mesh, int width, int height)
{
    std::ofstream out(path);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"none\" stroke=\"#cccccc\"/>\n";
    out << "<g fill=\"none\" stroke=\"#555555\" stroke-width=\"1\">\n";
    for (const auto& e : mesh.elements) {
        const int corner_count = mesh.element_type == "T3" ? 3 : 4;
        out << "<polyline points=\"";
        for (int i = 0; i < corner_count; ++i) {
            const auto& n = mesh.nodes[static_cast<std::size_t>(e.node_ids[i] - 1)];
            out << n.x << "," << n.y << " ";
        }
        const auto& n0 = mesh.nodes[static_cast<std::size_t>(e.node_ids[0] - 1)];
        out << n0.x << "," << n0.y << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g fill=\"#d62728\">\n";
    for (const auto& n : mesh.nodes) {
        out << "<circle cx=\"" << n.x << "\" cy=\"" << n.y << "\" r=\"1.8\"/>\n";
    }
    out << "</g>\n</svg>\n";
}

void write_summary(
    const std::filesystem::path& path,
    const MeshData& mesh,
    const QualitySummary& q,
    int nr,
    int nt,
    double target,
    double min_size,
    double max_size)
{
    std::ofstream out(path);
    out << "element_type=" << mesh.element_type << "\n";
    out << "target_element_size=" << target << "\n";
    out << "min_element_size=" << min_size << "\n";
    out << "max_element_size=" << max_size << "\n";
    out << "radial_divisions=" << nr << "\n";
    out << "circumferential_divisions=" << nt << "\n";
    out << "nodes=" << mesh.nodes.size() << "\n";
    out << "elements=" << mesh.elements.size() << "\n";
    out << "min_area=" << q.min_area << "\n";
    out << "min_edge=" << q.min_edge << "\n";
    out << "max_edge=" << q.max_edge << "\n";
    out << "max_aspect_ratio=" << q.max_aspect_ratio << "\n";
    out << "degenerate_elements=" << q.degenerate << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: mesh_auto_generation_diagnostic <roi_mask.bmp> <out_dir>"
                  << " [target=35] [min=18] [max=55]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path mask_path = argv[1];
        const std::filesystem::path out_dir = argv[2];
        const double target = argc >= 4 ? std::atof(argv[3]) : 35.0;
        const double min_size = argc >= 5 ? std::atof(argv[4]) : 18.0;
        const double max_size = argc >= 6 ? std::atof(argv[5]) : 55.0;

        const dic::Mask mask = read_bmp_mask(mask_path);
        auto loops = dic::mesh::extract_boundary_loops(mask);
        if (loops.size() < 2U) {
            throw std::runtime_error("This diagnostic currently expects an annulus mask with one hole.");
        }

        const auto outer = estimate_circle(loops[0]);
        const auto inner = estimate_circle(loops[1]);
        CircleEstimate center;
        center.cx = 0.5 * (outer.cx + inner.cx);
        center.cy = 0.5 * (outer.cy + inner.cy);
        const double r_inner = std::min(outer.radius, inner.radius);
        const double r_outer = std::max(outer.radius, inner.radius);
        const double thickness = r_outer - r_inner;

        const int nr = std::max(2, static_cast<int>(std::ceil(thickness / target)));
        int nt = std::max(12, static_cast<int>(std::ceil(2.0 * kPi * r_outer / target)));
        if (nt % 4 != 0) {
            nt += 4 - nt % 4;
        }

        std::filesystem::create_directories(out_dir);
        const auto q4 = generate_annulus_q4(center, r_inner, r_outer, nr, nt);
        const auto t3 = generate_annulus_t3(center, r_inner, r_outer, nr, nt);
        const auto q8 = generate_annulus_q8(center, r_inner, r_outer, nr, nt);
        const std::vector<MeshData> meshes = {t3, q4, q8};

        std::ofstream index(out_dir / "summary.txt");
        index << "mask_width=" << mask.width() << "\n";
        index << "mask_height=" << mask.height() << "\n";
        index << "target_element_size=" << target << "\n";
        index << "min_element_size=" << min_size << "\n";
        index << "max_element_size=" << max_size << "\n";
        index << "estimated_center=" << center.cx << "," << center.cy << "\n";
        index << "estimated_inner_radius=" << r_inner << "\n";
        index << "estimated_outer_radius=" << r_outer << "\n";
        index << "radial_divisions=" << nr << "\n";
        index << "circumferential_divisions=" << nt << "\n";

        for (const auto& mesh : meshes) {
            const auto mesh_dir = out_dir / mesh.element_type;
            const auto q = compute_quality(mesh);
            write_mesh_files(mesh_dir, mesh);
            write_svg(mesh_dir / "mesh_preview.svg", mesh, mask.width(), mask.height());
            write_summary(mesh_dir / "quality_summary.txt", mesh, q, nr, nt, target, min_size, max_size);
            index << mesh.element_type << "_nodes=" << mesh.nodes.size() << "\n";
            index << mesh.element_type << "_elements=" << mesh.elements.size() << "\n";
            index << mesh.element_type << "_max_aspect_ratio=" << q.max_aspect_ratio << "\n";
            index << mesh.element_type << "_min_edge=" << q.min_edge << "\n";
            index << mesh.element_type << "_max_edge=" << q.max_edge << "\n";
            index << mesh.element_type << "_degenerate_elements=" << q.degenerate << "\n";
        }

        std::cout << "Wrote auto mesh generation diagnostics to " << out_dir.string() << "\n";
        std::cout << "Radial divisions: " << nr << ", circumferential divisions: " << nt << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_auto_generation_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
