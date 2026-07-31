#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_SURFACE_OUTLIER_CLEANING_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_SURFACE_OUTLIER_CLEANING_HPP

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <vector>

namespace dic {

struct SurfaceOutlierCleaningOptions {
    int neighbor_count = 8;
    double distance_sigma = 6.0;
    double displacement_sigma = 6.0;
    double face_edge_scale = 4.0;
};

struct SurfaceOutlierCleaningResult {
    std::vector<std::uint8_t> valid_points;
    std::vector<std::uint8_t> valid_faces;
    std::size_t removed_points = 0;
    std::size_t removed_faces = 0;
};

SurfaceOutlierCleaningResult clean_surface_outliers(
    const std::vector<Eigen::Vector3d>& reference,
    const std::vector<Eigen::Vector3d>& deformed,
    const std::vector<std::array<std::int64_t, 3>>& faces,
    const std::vector<std::uint8_t>& initial_valid,
    const SurfaceOutlierCleaningOptions& options = {});

} // namespace dic

#endif
