#include <dic/mesh/generation/annulus_mesh_generator.hpp>

#include <dic/mesh/generation/boundary_exporter.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace dic::mesh {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct CircleEstimate {
    double cx{0.0};
    double cy{0.0};
    double radius{0.0};
};

CircleEstimate estimate_circle(const BoundaryLoop& loop)
{
    if (loop.points.empty()) {
        throw std::runtime_error("Cannot estimate a circle from an empty boundary loop.");
    }
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

std::size_t add_node(Mesh& mesh, double x, double y)
{
    const std::size_t id = mesh.nodes().size();
    Node node;
    node.id = id;
    node.coordinate = Eigen::Vector2d{x, y};
    mesh.add_node(node);
    return id;
}

std::size_t corner_key(int nt, int ir, int it)
{
    return static_cast<std::size_t>(ir * nt + (it % nt));
}

Mesh make_annulus_q4(const CircleEstimate& center, double r_inner, double r_outer, int nr, int nt)
{
    Mesh mesh;
    auto node_id = [nt](int ir, int it) {
        return static_cast<std::size_t>(ir * nt + (it % nt));
    };

    for (int ir = 0; ir <= nr; ++ir) {
        const double r = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
        for (int it = 0; it < nt; ++it) {
            const double theta = 2.0 * kPi * static_cast<double>(it) / nt;
            add_node(mesh, center.cx + r * std::cos(theta), center.cy + r * std::sin(theta));
        }
    }

    for (int ir = 0; ir < nr; ++ir) {
        for (int it = 0; it < nt; ++it) {
            MeshElementConnectivity elem;
            elem.type = MeshElementType::Q4;
            elem.node_ids = {
                node_id(ir, it),
                node_id(ir, it + 1),
                node_id(ir + 1, it + 1),
                node_id(ir + 1, it),
            };
            mesh.add_element(elem);
        }
    }
    return mesh;
}

Mesh make_annulus_t3(const CircleEstimate& center, double r_inner, double r_outer, int nr, int nt)
{
    Mesh q4 = make_annulus_q4(center, r_inner, r_outer, nr, nt);
    Mesh mesh;
    for (const auto& node : q4.nodes()) {
        mesh.add_node(node);
    }
    for (const auto& q : q4.elements()) {
        MeshElementConnectivity a;
        a.type = MeshElementType::T3;
        a.node_ids = {q.node_ids[0], q.node_ids[1], q.node_ids[2]};
        mesh.add_element(a);

        MeshElementConnectivity b;
        b.type = MeshElementType::T3;
        b.node_ids = {q.node_ids[0], q.node_ids[2], q.node_ids[3]};
        mesh.add_element(b);
    }
    return mesh;
}

Mesh make_annulus_q8(const CircleEstimate& center, double r_inner, double r_outer, int nr, int nt)
{
    Mesh mesh;
    std::vector<std::size_t> corner_ids(static_cast<std::size_t>((nr + 1) * nt), 0);
    for (int ir = 0; ir <= nr; ++ir) {
        const double r = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
        for (int it = 0; it < nt; ++it) {
            const double theta = 2.0 * kPi * static_cast<double>(it) / nt;
            corner_ids[corner_key(nt, ir, it)] =
                add_node(mesh, center.cx + r * std::cos(theta), center.cy + r * std::sin(theta));
        }
    }

    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edge_mid_ids;
    auto midpoint_id = [&](std::size_t a, std::size_t b) {
        const auto key = std::minmax(a, b);
        const auto found = edge_mid_ids.find(key);
        if (found != edge_mid_ids.end()) {
            return found->second;
        }
        const auto& na = mesh.nodes()[a];
        const auto& nb = mesh.nodes()[b];
        const std::size_t id = add_node(mesh,
                                        0.5 * (na.coordinate.x() + nb.coordinate.x()),
                                        0.5 * (na.coordinate.y() + nb.coordinate.y()));
        edge_mid_ids[key] = id;
        return id;
    };
    auto arc_midpoint_id = [&](std::size_t a, std::size_t b, double radius, int it) {
        const auto key = std::minmax(a, b);
        const auto found = edge_mid_ids.find(key);
        if (found != edge_mid_ids.end()) {
            return found->second;
        }
        const double theta = 2.0 * kPi * (static_cast<double>(it) + 0.5) / nt;
        const std::size_t id = add_node(mesh,
                                        center.cx + radius * std::cos(theta),
                                        center.cy + radius * std::sin(theta));
        edge_mid_ids[key] = id;
        return id;
    };

    for (int ir = 0; ir < nr; ++ir) {
        for (int it = 0; it < nt; ++it) {
            const std::size_t n0 = corner_ids[corner_key(nt, ir, it)];
            const std::size_t n1 = corner_ids[corner_key(nt, ir, it + 1)];
            const std::size_t n2 = corner_ids[corner_key(nt, ir + 1, it + 1)];
            const std::size_t n3 = corner_ids[corner_key(nt, ir + 1, it)];
            const double r0 = r_inner + (r_outer - r_inner) * static_cast<double>(ir) / nr;
            const double r1 = r_inner + (r_outer - r_inner) * static_cast<double>(ir + 1) / nr;
            const std::size_t n4 = (ir == 0) ? arc_midpoint_id(n0, n1, r0, it) : midpoint_id(n0, n1);
            const std::size_t n5 = midpoint_id(n1, n2);
            const std::size_t n6 = (ir + 1 == nr) ? arc_midpoint_id(n2, n3, r1, it) : midpoint_id(n2, n3);
            const std::size_t n7 = midpoint_id(n3, n0);

            MeshElementConnectivity elem;
            elem.type = MeshElementType::Q8;
            elem.node_ids = {n0, n1, n2, n3, n4, n5, n6, n7};
            mesh.add_element(elem);
        }
    }
    return mesh;
}

double positive_or_default(double value, double fallback)
{
    return value > 0.0 ? value : fallback;
}

} // namespace

AnnulusMeshGenerationResult generate_annulus_meshes_from_mask(
    const Mask& mask,
    const MeshGenerationConfig& config)
{
    auto loops = extract_boundary_loops(mask);
    if (loops.size() < 2U) {
        throw std::runtime_error("Annulus mesh generation expects an ROI mask with one hole.");
    }

    const auto outer = estimate_circle(loops[0]);
    const auto inner = estimate_circle(loops[1]);
    CircleEstimate center;
    center.cx = 0.5 * (outer.cx + inner.cx);
    center.cy = 0.5 * (outer.cy + inner.cy);
    const double r_inner = std::min(outer.radius, inner.radius);
    const double r_outer = std::max(outer.radius, inner.radius);
    const double thickness = r_outer - r_inner;
    if (thickness <= 0.0) {
        throw std::runtime_error("Annulus ROI has non-positive thickness.");
    }

    const double target = positive_or_default(config.target_element_size, 35.0);
    int nr = std::max(2, static_cast<int>(std::ceil(thickness / target)));
    int nt = std::max(12, static_cast<int>(std::ceil(2.0 * kPi * r_outer / target)));
    if (nt % 4 != 0) {
        nt += 4 - nt % 4;
    }

    AnnulusMeshGenerationResult result;
    result.summary.center_x = center.cx;
    result.summary.center_y = center.cy;
    result.summary.inner_radius = r_inner;
    result.summary.outer_radius = r_outer;
    result.summary.radial_divisions = nr;
    result.summary.circumferential_divisions = nt;
    result.summary.target_element_size = target;
    result.summary.min_element_size = config.min_element_size;
    result.summary.max_element_size = config.max_element_size;
    result.q4 = make_annulus_q4(center, r_inner, r_outer, nr, nt);
    result.t3 = make_annulus_t3(center, r_inner, r_outer, nr, nt);
    result.q8 = make_annulus_q8(center, r_inner, r_outer, nr, nt);
    return result;
}

Mesh generate_annulus_mesh_from_mask(
    const Mask& mask,
    const MeshGenerationConfig& config)
{
    auto result = generate_annulus_meshes_from_mask(mask, config);
    switch (config.element_type) {
    case MeshElementType::T3: return result.t3;
    case MeshElementType::Q4: return result.q4;
    case MeshElementType::Q8: return result.q8;
    }
    return result.q4;
}

} // namespace dic::mesh
