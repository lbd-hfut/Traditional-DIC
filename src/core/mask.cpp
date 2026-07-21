/**
 * @file mask.cpp
 * @brief Minimal implementation placeholder for mask.
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

#include <dic/core/mask.hpp>

namespace dic {

Mask::Mask(int width, int height) : width_(width), height_(height), data_(static_cast<std::size_t>(width * height), true) {}
bool Mask::valid(int x, int y) const { return x >= 0 && y >= 0 && x < width_ && y < height_ && (data_.empty() || data_[static_cast<std::size_t>(y * width_ + x)]); }

} // namespace dic
