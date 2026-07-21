/**
 * @file stereo_dic.hpp
 * @brief Stereo 3D DIC orchestration without duplicating 2D solvers.
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

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP



namespace dic {

enum class DICSolverType { Subset, Mesh };
class StereoDIC {
public:
    explicit StereoDIC(DICSolverType solver = DICSolverType::Subset);
    // TODO: 2D DIC solver -> stereo correspondence -> triangulation -> 3D
    // shape. Future solver enum may add PINN without duplicating subset_3d or
    // mesh_3d algorithms.
    void run();
private:
    DICSolverType solver_;
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_RECONSTRUCTION_STEREO_DIC_HPP
