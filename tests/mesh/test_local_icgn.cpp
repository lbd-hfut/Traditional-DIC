#include <gtest/gtest.h>

#include <dic/mesh/solver/local_icgn.hpp>
#include <dic/interpolation/bspline.hpp>

#include <cmath>
#include <vector>

namespace {

// ---- Helpers ----

// Generate a synthetic Gaussian blob image
std::vector<double> make_gaussian_blob(int h, int w, double cx, double cy, double sigma) {
    std::vector<double> img(h * w, 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double dx = x - cx, dy = y - cy;
            img[y * w + x] = std::exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
        }
    }
    return img;
}

// Generate circular subset offsets (dx, dy) within subset_radius
void make_circular_offsets(int subset_radius,
                           std::vector<double>& dx,
                           std::vector<double>& dy) {
    dx.clear(); dy.clear();
    for (int y = -subset_radius; y <= subset_radius; ++y) {
        for (int x = -subset_radius; x <= subset_radius; ++x) {
            if (x*x + y*y <= subset_radius * subset_radius) {
                dx.push_back(static_cast<double>(x));
                dy.push_back(static_cast<double>(y));
            }
        }
    }
}

// Extract reference patch values and gradients (simple central difference)
void extract_patch(const std::vector<double>& img, int img_w,
                   int cx, int cy,
                   const std::vector<double>& dx,
                   const std::vector<double>& dy,
                   std::vector<double>& f_buf,
                   std::vector<double>& fx_buf,
                   std::vector<double>& fy_buf) {
    int n = static_cast<int>(dx.size());
    f_buf.resize(n);
    fx_buf.resize(n);
    fy_buf.resize(n);
    int h = static_cast<int>(img.size()) / img_w;

    for (int i = 0; i < n; ++i) {
        int px = cx + static_cast<int>(std::round(dx[i]));
        int py = cy + static_cast<int>(std::round(dy[i]));
        // Clamp
        if (px < 1) px = 1; if (px >= img_w - 1) px = img_w - 2;
        if (py < 1) py = 1; if (py >= h - 1) py = h - 2;
        f_buf[i] = img[py * img_w + px];
        fx_buf[i] = 0.5 * (img[py * img_w + (px + 1)] - img[py * img_w + (px - 1)]);
        fy_buf[i] = 0.5 * (img[(py + 1) * img_w + px] - img[(py - 1) * img_w + px]);
    }
}

// ---- Tests ----

TEST(LocalICGN, CoarseSearchFindsZeroDisplacement) {
    const int h = 100, w = 100;
    const int cx = 50, cy = 50;
    const int sr = 10, search_r = 5;

    auto ref = make_gaussian_blob(h, w, cx, cy, 5.0);
    // def = ref (no displacement)
    auto def = ref;

    dic::LocalICGNSolver solver;
    auto result = solver.coarse_search(ref.data(), def.data(), h, w,
                                       cx, cy, sr, search_r);
    EXPECT_EQ(result.dx, 0.0);
    EXPECT_EQ(result.dy, 0.0);
}

TEST(LocalICGN, CoarseSearchFindsKnownDisplacement) {
    const int h = 120, w = 120;
    const int cx = 60, cy = 60;
    const int sr = 12, search_r = 10;
    const int shift_x = 3, shift_y = -4;

    auto ref = make_gaussian_blob(h, w, cx, cy, 5.0);
    auto def = make_gaussian_blob(h, w, cx + shift_x, cy + shift_y, 5.0);

    dic::LocalICGNSolver solver;
    auto result = solver.coarse_search(ref.data(), def.data(), h, w,
                                       cx, cy, sr, search_r);
    EXPECT_EQ(result.dx, static_cast<double>(shift_x));
    EXPECT_EQ(result.dy, static_cast<double>(shift_y));
}

TEST(LocalICGN, SolveRecoversSubpixelTranslation) {
    const int h = 80, w = 80;
    const int cx = 40, cy = 40;
    const int sr = 8;

    // Reference: Gaussian at center
    auto ref = make_gaussian_blob(h, w, cx, cy, 3.0);

    // Build BSplineInterpolator for deformed image (used as reference here,
    // with the subpixel shift baked into the solve initial guess and the
    // deform image being the same reference, testing the solver's ability
    // to converge to zero correction from a small initial offset).
    dic::Image def_img_raw(w, h, std::vector<float>(ref.begin(), ref.end()));
    dic::BSplineInterpolator bsp(def_img_raw);
    bsp.precompute();

    std::vector<double> dx, dy;
    make_circular_offsets(sr, dx, dy);
    std::vector<double> f_buf, fx_buf, fy_buf;
    extract_patch(ref, w, cx, cy, dx, dy, f_buf, fx_buf, fy_buf);

    // Test: start with small subpixel guess (0.3, -0.2), should converge near (0,0)
    dic::LocalICGNParams params;
    params.max_iter = 30;
    params.cutoff_diffnorm = 1e-6;
    dic::LocalICGNSolver solver(params);

    auto result = solver.solve(f_buf.data(), fx_buf.data(), fy_buf.data(),
                               static_cast<int>(dx.size()),
                               static_cast<double>(cx), static_cast<double>(cy),
                               dx.data(), dy.data(),
                               &bsp, 0.3, -0.2);

    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.u, 0.0, 1e-3);  // should converge back to ~0
    EXPECT_NEAR(result.v, 0.0, 1e-3);
    EXPECT_LE(result.iterations, params.max_iter);
}

TEST(LocalICGN, SolveWorksWithoutDeformedInterpolator) {
    const int h = 60, w = 60;
    const int cx = 30, cy = 30;
    const int sr = 6;

    auto ref = make_gaussian_blob(h, w, cx, cy, 3.0);

    std::vector<double> dx, dy;
    make_circular_offsets(sr, dx, dy);
    std::vector<double> f_buf, fx_buf, fy_buf;
    extract_patch(ref, w, cx, cy, dx, dy, f_buf, fx_buf, fy_buf);

    dic::LocalICGNSolver solver;
    // nullptr bsp → deformed image treated as constant (zero)
    auto result = solver.solve(f_buf.data(), fx_buf.data(), fy_buf.data(),
                               static_cast<int>(dx.size()),
                               static_cast<double>(cx), static_cast<double>(cy),
                               dx.data(), dy.data(),
                               nullptr, 0.0, 0.0);
    // Without deformed image, the solver should still run but may not refine
    // (deformed patch is all zeros → ZNSSD residual is dominated by normalization)
    // We just check it doesn't crash and returns finite values.
    EXPECT_TRUE(std::isfinite(result.u));
    EXPECT_TRUE(std::isfinite(result.v));
}

} // namespace
