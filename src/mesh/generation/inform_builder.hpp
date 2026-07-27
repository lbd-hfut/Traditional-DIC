#pragma once

#include <dic/mesh/mesh_generation_config.hpp>

#include <vector>

namespace dic::mesh {

// ============================================================
// Build the inform array: for each ROI pixel, determine which
// element it belongs to.
//
// Returns flat array: [gx, gy, elem_id], where elem_id is 1-based.
//
// For Q8 elements, edges are subdivided (inserting midpoints)
// to better approximate the curved quadratic boundary.
//
// If roi_mask is provided (nullptr = entire image), only pixels
// inside the ROI are included.
// ============================================================

std::vector<double> build_inform(
    const double* nodes_coord,    // [2 * n_nodes]
    int n_nodes,
    const int*    elements,        // [n_elements * stride] (stride = nn or 9 for Q8)
    int n_elements,
    MeshElementType element_type,
    int img_h, int img_w,
    const unsigned char* roi_mask = nullptr);

} // namespace dic::mesh
