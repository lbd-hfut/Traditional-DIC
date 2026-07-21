/**
 * @file element.hpp
 * @brief Abstract finite element interface for 2D Mesh-DIC.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_ELEMENT_ELEMENT_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_ELEMENT_ELEMENT_HPP

#include <Eigen/Dense>

namespace dic {

class Element {
public:
    virtual ~Element() = default;
    virtual int node_count() const = 0;
    virtual Eigen::VectorXd shape_functions(double xi, double eta) const = 0;
    virtual Eigen::MatrixXd shape_function_derivatives(double xi, double eta) const = 0;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_ELEMENT_ELEMENT_HPP
