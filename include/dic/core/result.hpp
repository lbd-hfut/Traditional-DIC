/**
 * @file result.hpp
 * @brief 2D and 3D DIC result containers.
 *
 * Responsibilities:
 * - Define stable result containers shared across C++ and Python APIs.
 * - Keep solver status and quality fields explicit for downstream filtering.
 *
 * Inputs:
 * - Solver outputs from Subset-DIC, Mesh-DIC, and 3D reconstruction.
 *
 * Outputs:
 * - 2D/3D displacement records with validity and quality indicators.
 *
 * Dependencies:
 * - Eigen for numerical types.
 * - OpenCV interfaces are reserved for image loading, SIFT, and calibration where needed.
 * - Internal Traditional-DIC modules declared by includes.
 *
 * TODO:
 * - Add covariance/uncertainty fields once solvers expose them.
 * - Add batch result containers if Python bindings need contiguous arrays.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP

#include <dic/core/types.hpp>

namespace dic {

struct Displacement2D {
    double x{0.0};
    double y{0.0};
    double u{0.0};
    double v{0.0};
    double du_dx{0.0};
    double du_dy{0.0};
    double dv_dx{0.0};
    double dv_dy{0.0};
    double correlation{0.0};
    SolverStatus status{SolverStatus::InvalidInput};
    bool valid{false};
};

struct Displacement3D {
    double X{0.0};
    double Y{0.0};
    double Z{0.0};
    double U{0.0};
    double V{0.0};
    double W{0.0};
    SolverStatus status{SolverStatus::InvalidInput};
    bool valid{false};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CORE_RESULT_HPP
