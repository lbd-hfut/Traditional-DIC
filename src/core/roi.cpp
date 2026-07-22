/**
 * @file roi.cpp
 * @brief Minimal implementation placeholder for ROI.
 *
 * Responsibilities:
 * - Provide ROI membership tests for rectangle, polygon, and mask-image ROIs.
 * - Keep ROI representation independent from any GUI drawing workflow.
 *
 * Inputs:
 * - Rectangle, polygon, mask data, or point coordinates.
 *
 * Outputs:
 * - Boolean membership in image pixel coordinates.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Add robust polygon boundary handling and ROI image metadata validation.
 * - Add tests for mask ROI and polygon edge cases.
 */

#include <dic/core/roi.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace dic {

ROI::ROI() = default;

ROI::ROI(const RectangleROI& rectangle)
    : type_(ROIType::Rectangle), rectangle_(rectangle)
{
    if (rectangle.width < 0 || rectangle.height < 0) {
        throw std::invalid_argument("Rectangle ROI dimensions must be non-negative.");
    }
}

ROI::ROI(PolygonROI polygon)
    : type_(ROIType::Polygon), polygon_(std::move(polygon))
{
}

ROI::ROI(Mask mask)
    : type_(ROIType::Mask), mask_(std::move(mask))
{
}

ROI ROI::all()
{
    return ROI{};
}

ROI ROI::from_mask_image(const std::string& path)
{
    return ROI{Mask{path}};
}

ROIType ROI::type() const
{
    return type_;
}

bool ROI::empty() const
{
    switch (type_) {
    case ROIType::All:
        return false;
    case ROIType::Rectangle:
        return !rectangle_ || rectangle_->width == 0 || rectangle_->height == 0;
    case ROIType::Polygon:
        return !polygon_ || polygon_->vertices.size() < 3;
    case ROIType::Mask:
        return !mask_ || mask_->empty();
    }
    return true;
}

bool ROI::contains(const Eigen::Vector2d& point) const
{
    return contains(point.x(), point.y());
}

bool ROI::contains(double x, double y) const
{
    switch (type_) {
    case ROIType::All:
        return true;
    case ROIType::Rectangle:
        if (!rectangle_) {
            return false;
        }
        return x >= rectangle_->x &&
            y >= rectangle_->y &&
            x < rectangle_->x + rectangle_->width &&
            y < rectangle_->y + rectangle_->height;
    case ROIType::Polygon:
        if (!polygon_ || polygon_->vertices.size() < 3) {
            return false;
        }
        {
            bool inside = false;
            const auto& vertices = polygon_->vertices;
            for (std::size_t i = 0, j = vertices.size() - 1; i < vertices.size(); j = i++) {
                const auto& pi = vertices[i];
                const auto& pj = vertices[j];
                const bool intersects = ((pi.y() > y) != (pj.y() > y)) &&
                    (x < (pj.x() - pi.x()) * (y - pi.y()) / (pj.y() - pi.y()) + pi.x());
                if (intersects) {
                    inside = !inside;
                }
            }
            return inside;
        }
    case ROIType::Mask:
        if (!mask_) {
            return false;
        }
        return mask_->valid(
            static_cast<int>(std::floor(x)),
            static_cast<int>(std::floor(y))
        );
    }
    return false;
}

const std::optional<RectangleROI>& ROI::rectangle() const
{
    return rectangle_;
}

const std::optional<PolygonROI>& ROI::polygon() const
{
    return polygon_;
}

const std::optional<Mask>& ROI::mask() const
{
    return mask_;
}

} // namespace dic
