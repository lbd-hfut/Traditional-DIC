/**
 * @file observation_mask.hpp
 * @brief ROI mask generation from per-camera sparse 2D observations.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_OBSERVATION_MASK_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_OBSERVATION_MASK_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

struct ObservationMaskOptions {
    int outlier_k = 6;
    double outlier_knn_scale = 4.0;
    double component_radius_scale = 8.0;
    double edge_scale = 8.0;
    double radius_scale = 6.0;
    int min_hole_area = 500;
    int tiny_hole_fill_area = 3000;
};

struct ObservationMaskResult {
    int camera_index = -1;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> mask;
    std::vector<unsigned char> hull_mask;
    std::vector<unsigned char> supported_mask;
    std::vector<unsigned char> rejected_hole_mask;
    std::vector<Eigen::Vector2d> observations;
    std::vector<Eigen::Vector2d> clean_observations;
    int n_triangles_raw = 0;
    int n_triangles_valid = 0;
    int n_holes_detected = 0;
    int n_holes_filled_as_speckle = 0;
    int n_holes_rejected = 0;
};

ObservationMaskResult build_observation_mask(
    int camera_index,
    int width,
    int height,
    const std::vector<Eigen::Vector2d>& observations,
    const ObservationMaskOptions& options = {});

std::vector<ObservationMaskResult> build_observation_masks(
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& camera_indices,
    const std::vector<Eigen::Vector2d>& observation_uv,
    const ObservationMaskOptions& options = {});

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_OBSERVATION_MASK_HPP
