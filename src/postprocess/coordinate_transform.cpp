/**
 * @file coordinate_transform.cpp
 * @brief Coordinate transform implementation: scaling, rotation, translation.
 */

#include <dic/postprocess/coordinate_transform.hpp>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dic {

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

Eigen::Vector3d camera_to_world(const Eigen::Vector3d& point, double scale)
{
    return point * scale;
}

void scale_points(std::vector<Eigen::Vector3d>& points, double scale)
{
    for (auto& pt : points)
        pt *= scale;
}

// ---------------------------------------------------------------------------
// Rigid transform: R * point + t
// ---------------------------------------------------------------------------

Eigen::Vector3d transform_point(const Eigen::Vector3d& point,
                                 const Eigen::Matrix3d& rotation,
                                 const Eigen::Vector3d& translation)
{
    return rotation * point + translation;
}

void transform_points(std::vector<Eigen::Vector3d>& points,
                       const Eigen::Matrix3d& rotation,
                       const Eigen::Vector3d& translation)
{
    for (auto& pt : points)
        pt = rotation * pt + translation;
}

// ---------------------------------------------------------------------------
// Align one vector to another via axis-angle
// ---------------------------------------------------------------------------

Eigen::Matrix3d align_vector(const Eigen::Vector3d& from,
                               const Eigen::Vector3d& to)
{
    Eigen::Vector3d f = from.normalized();
    Eigen::Vector3d t = to.normalized();

    double dot_product = f.dot(t);

    // Vectors are already aligned
    if (dot_product > 1.0 - 1.0e-12)
        return Eigen::Matrix3d::Identity();

    // Vectors are anti-parallel
    if (dot_product < -1.0 + 1.0e-12) {
        // Pick an orthogonal vector
        Eigen::Vector3d axis;
        if (std::abs(f.x()) < 0.9)
            axis = Eigen::Vector3d::UnitX().cross(f).normalized();
        else
            axis = Eigen::Vector3d::UnitY().cross(f).normalized();
        // 180-degree rotation around axis
        return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
    }

    // General case: rotation axis = cross(f, t)
    Eigen::Vector3d axis = f.cross(t).normalized();
    double angle = std::acos(dot_product);
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

// ---------------------------------------------------------------------------
// Centroid computation
// ---------------------------------------------------------------------------

Eigen::Vector3d compute_centroid(const std::vector<Eigen::Vector3d>& points,
                                  const std::vector<bool>& mask)
{
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;

    if (mask.empty()) {
        for (const auto& pt : points) {
            if (pt.allFinite()) {
                sum += pt;
                ++count;
            }
        }
    } else {
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (mask[i] && points[i].allFinite()) {
                sum += points[i];
                ++count;
            }
        }
    }

    if (count == 0) return Eigen::Vector3d::Zero();
    return sum / static_cast<double>(count);
}

Eigen::Vector3d center_points(std::vector<Eigen::Vector3d>& points,
                               const std::vector<bool>& mask)
{
    Eigen::Vector3d centroid = compute_centroid(points, mask);

    if (mask.empty()) {
        for (auto& pt : points)
            pt -= centroid;
    } else {
        for (std::size_t i = 0; i < points.size(); ++i)
            if (mask[i] && points[i].allFinite())
                points[i] -= centroid;
    }
    return centroid;
}

} // namespace dic
