/**
 * @file roi.hpp
 * @brief Region-of-interest abstractions.
 *
 * Responsibilities:
 * - Represent user-provided ROI data for image-domain DIC workflows.
 * - Support mask-image ROI as the primary non-GUI ROI input path.
 * - Keep ROI membership tests independent from GUI drawing tools.
 *
 * Inputs:
 * - Rectangle parameters, polygon vertices, or a user-provided Mask.
 *
 * Outputs:
 * - Point-in-ROI membership checks in image pixel coordinates.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add ROI image metadata validation and mask/image dimension checks.
 * - Add polygon edge-case tests for boundary points and self-intersections.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP

#include <dic/core/mask.hpp>
#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

namespace dic {

enum class ROIType {
    All,
    Rectangle,
    Polygon,
    Mask
};

struct RectangleROI {
    int x{0};
    int y{0};
    int width{0};
    int height{0};
};

struct PolygonROI {
    std::vector<Eigen::Vector2d> vertices;
};

class ROI {
public:
    ROI();
    explicit ROI(const RectangleROI& rectangle);
    explicit ROI(PolygonROI polygon);
    explicit ROI(Mask mask);

    static ROI all();
    static ROI from_mask_image(const std::string& path);

    ROIType type() const;
    bool empty() const;
    bool contains(const Eigen::Vector2d& point) const;
    bool contains(double x, double y) const;

    const std::optional<RectangleROI>& rectangle() const;
    const std::optional<PolygonROI>& polygon() const;
    const std::optional<Mask>& mask() const;

private:
    ROIType type_{ROIType::All};
    std::optional<RectangleROI> rectangle_;
    std::optional<PolygonROI> polygon_;
    std::optional<Mask> mask_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP
