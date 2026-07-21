/**
 * @file strain_3d.hpp
 * @brief 3D strain postprocessing placeholder.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP

#include <dic/postprocess/strain.hpp>

namespace dic {

Strain3D compute_strain_3d();

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_POSTPROCESS_STRAIN_3D_HPP
