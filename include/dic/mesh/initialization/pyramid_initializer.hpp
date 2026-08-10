/**
 * @file pyramid_initializer.hpp
 * @brief True coarse-to-fine image-pyramid nodal initialization for Mesh-DIC.
 *
 * Responsibilities:
 * - Estimate a per-node coarse displacement field through an image pyramid.
 * - Feed those displacements to the FFT initializer as search-center offsets
 *   so large disparities are covered by a small radius at each level.
 *
 * Inputs:
 * - Reference / deformed images, node coordinates, pyramid configuration.
 *
 * Outputs:
 * - Per-node full-resolution initial displacements with a validity mask.
 *
 * Dependencies:
 * - OpenCV imgproc for pyramid downsampling (cv::resize) when available.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_PYRAMID_INITIALIZER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_PYRAMID_INITIALIZER_HPP

#include <dic/core/image.hpp>
#include <dic/initialization/initializer.hpp>
#include <dic/mesh/mesh_config.hpp>

#include <Eigen/Core>
#include <vector>

namespace dic::mesh {

// PyramidInitializationConfig is a nested type of dic::MeshConfig; expose a
// short alias inside dic::mesh so the public function signature stays terse.
using PyramidInitializationConfig = dic::MeshConfig::PyramidInitializationConfig;

// Estimate a per-node coarse displacement field through a true image pyramid.
// Returns full-resolution displacements; valid=false marks nodes whose
// pyramid chain failed (the caller falls back to a blind full-resolution
// search on those nodes).
std::vector<InitialDisplacement> estimate_pyramid_initial_displacements(
    const Image& reference,
    const Image& deformed,
    const std::vector<Eigen::Vector2d>& points,
    const PyramidInitializationConfig& cfg);

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_PYRAMID_INITIALIZER_HPP
