/**
 * @file strain.hpp
 * @brief Common strain tensor containers.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_HPP



namespace dic {

struct Strain2D { double exx{0.0}; double eyy{0.0}; double exy{0.0}; bool valid{false}; };
struct Strain3D { double exx{0.0}; double eyy{0.0}; double ezz{0.0}; double exy{0.0}; double exz{0.0}; double eyz{0.0}; bool valid{false}; };

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_HPP
