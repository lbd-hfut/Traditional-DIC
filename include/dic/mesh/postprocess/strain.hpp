#pragma once

#include <dic/mesh/mesh_generation_config.hpp>

#include <vector>

namespace dic {

void compute_strain(
    mesh::MeshElementType type,
    const double* U, int n_nodes,
    const double* nodes_coord,
    const int* elements, int n_elements,
    std::vector<double>& Exx,
    std::vector<double>& Eyy,
    std::vector<double>& Exy);

} // namespace dic
