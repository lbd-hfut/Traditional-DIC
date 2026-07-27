#pragma once

#include <cstdint>
#include <vector>

namespace dic {

// ============================================================
// Local ICGN solver for per-node displacement initialization
//
// Provides:
//   coarse_search()  – integer-pixel template matching (ZNCC)
//   solve()          – 6-DOF affine ICGN subpixel refinement
//
// Uses BSplineInterpolator for deformed-image interpolation
// when available; falls back to constant (zero) otherwise.
// ============================================================

struct LocalICGNParams {
    int    max_iter         = 25;
    double cutoff_diffnorm  = 1e-3;
    double lambda_reg       = 1e-3;
};

struct LocalICGNResult {
    bool   success    = false;
    double u          = 0.0;
    double v          = 0.0;
    double corr_coef  = -1.0;
    double diffnorm   = 0.0;
    int    iterations = 0;
};

struct CoarseSearchResult {
    double dy = 0.0;
    double dx = 0.0;
};

class BSplineInterpolator;  // forward declaration

class LocalICGNSolver {
public:
    explicit LocalICGNSolver(const LocalICGNParams& params = LocalICGNParams{})
        : params_(params) {}

    // ---- 6-DOF affine ICGN subpixel refinement ----
    // f_buffer, fx_ref, fy_ref: reference image patch and gradients at (dx,dy)
    // xc, yc: subset center in image coordinates
    // dx, dy: local offsets from center (n_pixels elements each)
    // bsp: optional BSplineInterpolator for deformed image
    // u0, v0: initial integer-pixel displacement guess
    LocalICGNResult solve(
        const double* f_buffer,
        const double* fx_ref,
        const double* fy_ref,
        int n_pixels,
        double xc, double yc,
        const double* dx, const double* dy,
        BSplineInterpolator* bsp = nullptr,
        double u0 = 0.0, double v0 = 0.0);

    // ---- Integer-pixel template matching ----
    // Uses normalized cross-correlation (ZNCC via matchTemplate when
    // OpenCV is available, otherwise returns zero displacement).
    CoarseSearchResult coarse_search(
        const double* ref_img,
        const double* def_img,
        int img_h, int img_w,
        int cx, int cy,
        int subset_radius,
        int search_radius,
        const uint8_t* mask_pad = nullptr,
        int mask_pad_h = 0, int mask_pad_w = 0);

private:
    LocalICGNParams params_;
};

} // namespace dic
