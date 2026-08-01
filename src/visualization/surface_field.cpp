#include <dic/visualization/surface_field.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dic::visualization {

SurfaceFieldData prepare_surface_field(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<std::array<std::int64_t, 3>>& faces,
    const std::vector<double>& point_values)
{
    if (points.size() != point_values.size()) {
        throw std::invalid_argument("points and point_values must have the same length");
    }

    SurfaceFieldData result;
    result.faces.reserve(faces.size());
    result.face_centers.reserve(faces.size());
    result.face_values.reserve(faces.size());
    result.valid_faces.reserve(faces.size());

    const auto point_count = static_cast<std::int64_t>(points.size());
    for (const auto& face : faces) {
        bool valid = true;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        double value = 0.0;
        for (const auto raw_id : face) {
            if (raw_id < 0 || raw_id >= point_count) {
                valid = false;
                break;
            }
            const auto id = static_cast<std::size_t>(raw_id);
            const auto& point = points[id];
            const double point_value = point_values[id];
            if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z()) ||
                !std::isfinite(point_value)) {
                valid = false;
                break;
            }
            center += point;
            value += point_value;
        }
        if (valid) {
            center /= 3.0;
            value /= 3.0;
        } else {
            center = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
            value = std::numeric_limits<double>::quiet_NaN();
        }
        result.faces.push_back(face);
        result.face_centers.push_back(center);
        result.face_values.push_back(value);
        result.valid_faces.push_back(valid ? 1 : 0);
    }

    return result;
}

} // namespace dic::visualization
