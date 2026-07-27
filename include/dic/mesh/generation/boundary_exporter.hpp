#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_BOUNDARY_EXPORTER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_BOUNDARY_EXPORTER_HPP

#include <dic/core/mask.hpp>
#include <Eigen/Dense>
#include <vector>

namespace dic::mesh {

struct BoundaryLoop {
    std::vector<Eigen::Vector2d> points;
};

std::vector<BoundaryLoop> extract_boundary_loops(const Mask& mask);

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_GENERATION_BOUNDARY_EXPORTER_HPP
