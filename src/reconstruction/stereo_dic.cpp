/**
 * @file stereo_dic.cpp
 * @brief Minimal implementation placeholder for stereo DIC.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/reconstruction/stereo_dic.hpp>

namespace dic {

StereoDIC::StereoDIC(DICSolverType solver) : solver_(solver) {}
void StereoDIC::run() { (void)solver_; }

} // namespace dic
