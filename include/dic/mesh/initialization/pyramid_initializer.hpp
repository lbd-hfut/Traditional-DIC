#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_PYRAMID_INITIALIZER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_PYRAMID_INITIALIZER_HPP

#include <dic/core/image.hpp>
#include <dic/initialization/initializer.hpp>
#include <dic/mesh/mesh_config.hpp>

#include <Eigen/Core>
#include <vector>

namespace dic::mesh {

using PyramidInitializationConfig = dic::MeshConfig::PyramidInitializationConfig;

std::vector<InitialDisplacement> estimate_pyramid_initial_displacements(
    const Image& reference,
    const Image& deformed,
    const std::vector<Eigen::Vector2d>& points,
    const PyramidInitializationConfig& cfg);

} // namespace dic::mesh

#endif
