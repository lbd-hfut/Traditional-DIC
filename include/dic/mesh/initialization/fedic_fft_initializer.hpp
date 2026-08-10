#ifndef TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_FEDIC_FFT_INITIALIZER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_FEDIC_FFT_INITIALIZER_HPP

#include <dic/core/image.hpp>
#include <dic/initialization/initializer.hpp>

#include <Eigen/Core>

namespace dic::mesh {

struct FEDICFFTInitialDisplacement {
    InitialDisplacement initial{};
    double peak_to_correlation_energy{0.0};
    double peak_to_entropy{0.0};
};

// FE-DIC's funIntegerSearchPt equivalent: square subset, normalized
// cross-correlation surface, and its strongest integer-pixel peak.
// initial_offset shifts the deformed-search window center away from (0,0);
// the returned initial displacement is initial_offset + the matched peak.
FEDICFFTInitialDisplacement estimate_fedic_fft_initial_displacement(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    int search_radius,
    int window_size,
    const Eigen::Vector2d& initial_offset = Eigen::Vector2d::Zero());

} // namespace dic::mesh

#endif // TRADITIONAL_DIC_INCLUDE_DIC_MESH_INITIALIZATION_FEDIC_FFT_INITIALIZER_HPP
