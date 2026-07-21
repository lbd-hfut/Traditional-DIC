/**
 * @file roi.hpp
 * @brief Region-of-interest abstractions.
 *
 * Responsibilities:
 * - Define the public interface and data structures for this module.
 * - Keep dependencies explicit and module coupling low for future development.
 *
 * Inputs:
 * - Images, coordinates, parameters, configuration, or calibration data relevant to this module.
 *
 * Outputs:
 * - Typed results, numerical values, solver state, or placeholder exceptions.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Implement validated numerical algorithms.
 * - Add input validation, edge-case handling, and regression tests.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP

#include <Eigen/Dense>
#include <vector>

namespace dic {

struct RectangleROI { int x{0}; int y{0}; int width{0}; int height{0}; };
struct PolygonROI { std::vector<Eigen::Vector2d> vertices; };
class ROI { public: bool contains(const Eigen::Vector2d& point) const; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_ROI_HPP
