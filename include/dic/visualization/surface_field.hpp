#ifndef TRADITIONAL_DIC_INCLUDE_DIC_VISUALIZATION_SURFACE_FIELD_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_VISUALIZATION_SURFACE_FIELD_HPP

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <vector>

namespace dic::visualization {

struct SurfaceFieldData {
    std::vector<std::array<std::int64_t, 3>> faces;
    std::vector<Eigen::Vector3d> face_centers;
    std::vector<double> face_values;
    std::vector<std::uint8_t> valid_faces;
};

SurfaceFieldData prepare_surface_field(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<std::array<std::int64_t, 3>>& faces,
    const std::vector<double>& point_values);

} // namespace dic::visualization

#endif // TRADITIONAL_DIC_INCLUDE_DIC_VISUALIZATION_SURFACE_FIELD_HPP
