#include "inform_builder.hpp"
#include "../element/shape_func_internal.hpp"

#include <algorithm>
#include <cmath>

namespace dic::mesh {

// ============================================================
// Point-in-polygon test (ray casting algorithm)
// ============================================================
static bool point_in_polygon(const double* poly, int n_vertices,
                             double px, double py)
{
    bool inside = false;
    for (int i = 0, j = n_vertices - 1; i < n_vertices; j = i++) {
        double xi = poly[2 * i], yi = poly[2 * i + 1];
        double xj = poly[2 * j], yj = poly[2 * j + 1];
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

// ============================================================
// Build boundary polygon for one element
//
// T3: [0, 1, 2]
// Q4: [0, 1, 2, 3]
// Q8: [0, 4, 1, 5, 2, 6, 3, 7] + edge subdivision (16 vertices)
// ============================================================
static std::vector<double> build_element_polygon(
    const double* elem_nodes, int nn, MeshElementType type)
{
    if (type == MeshElementType::T3) {
        // T3: 3-vertex polygon
        return {elem_nodes[0], elem_nodes[1],
                elem_nodes[2], elem_nodes[3],
                elem_nodes[4], elem_nodes[5]};
    }

    if (type == MeshElementType::Q4) {
        // Q4: 4-vertex polygon (corners)
        return {elem_nodes[0], elem_nodes[1],
                elem_nodes[2], elem_nodes[3],
                elem_nodes[4], elem_nodes[5],
                elem_nodes[6], elem_nodes[7]};
    }

    // Q8: 8 boundary nodes, subdivide → 16 vertices
    // Node order: corners 0,1,2,3, edge mids 4,5,6,7
    // Boundary order: 0,4,1,5,2,6,3,7
    const int boundary_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<double> poly;
    poly.reserve(32); // 16 vertices × 2 coords
    for (int i = 0; i < 8; ++i) {
        int curr = boundary_order[i];
        int next = boundary_order[(i + 1) % 8];
        // Current vertex
        poly.push_back(elem_nodes[2 * curr]);
        poly.push_back(elem_nodes[2 * curr + 1]);
        // Midpoint
        poly.push_back(0.5 * (elem_nodes[2 * curr] + elem_nodes[2 * next]));
        poly.push_back(0.5 * (elem_nodes[2 * curr + 1] + elem_nodes[2 * next + 1]));
    }
    return poly;
}

// ============================================================
// build_inform
// ============================================================
std::vector<double> build_inform(
    const double* nodes_coord, int n_nodes,
    const int* elements, int n_elements,
    MeshElementType element_type,
    int img_h, int img_w,
    const unsigned char* roi_mask)
{
    using namespace internal;
    int nn = nodes_per_element(element_type);
    int elem_stride = (element_type == MeshElementType::Q8) ? 9 : nn;

    // Pre-extract element node coordinates
    std::vector<std::vector<double>> elem_nodes(n_elements);
    std::vector<std::vector<double>> elem_polygons(n_elements);
    for (int e = 0; e < n_elements; ++e) {
        elem_nodes[e].resize(2 * nn);
        for (int k = 0; k < nn; ++k) {
            int nid = elements[e * elem_stride + k] - 1;
            elem_nodes[e][2 * k]     = nodes_coord[2 * nid];
            elem_nodes[e][2 * k + 1] = nodes_coord[2 * nid + 1];
        }
        elem_polygons[e] = build_element_polygon(elem_nodes[e].data(), nn, element_type);
    }

    // Collect inform triples [gx, gy, elem_id]
    std::vector<double> inform;

    for (int e = 0; e < n_elements; ++e) {
        const auto& poly = elem_polygons[e];
        int n_verts = static_cast<int>(poly.size() / 2);

        // Compute bounding box
        double xmin = poly[0], xmax = poly[0];
        double ymin = poly[1], ymax = poly[1];
        for (int v = 1; v < n_verts; ++v) {
            xmin = std::min(xmin, poly[2 * v]);
            xmax = std::max(xmax, poly[2 * v]);
            ymin = std::min(ymin, poly[2 * v + 1]);
            ymax = std::max(ymax, poly[2 * v + 1]);
        }

        int ixmin = std::max(0,          static_cast<int>(std::floor(xmin)));
        int ixmax = std::min(img_w - 1,  static_cast<int>(std::ceil(xmax)));
        int iymin = std::max(0,          static_cast<int>(std::floor(ymin)));
        int iymax = std::min(img_h - 1,  static_cast<int>(std::ceil(ymax)));

        for (int py = iymin; py <= iymax; ++py) {
            for (int px = ixmin; px <= ixmax; ++px) {
                // ROI mask check (if provided)
                if (roi_mask && !roi_mask[py * img_w + px]) continue;

                if (point_in_polygon(poly.data(), n_verts,
                                     static_cast<double>(px),
                                     static_cast<double>(py))) {
                    inform.push_back(static_cast<double>(px));
                    inform.push_back(static_cast<double>(py));
                    inform.push_back(static_cast<double>(e + 1));  // 1-based elem_id
                }
            }
        }
    }

    return inform;
}

} // namespace dic::mesh
