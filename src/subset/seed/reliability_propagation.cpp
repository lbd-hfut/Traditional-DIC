/**
 * @file reliability_propagation.cpp
 * @brief Reliability-guided DIC propagation (ncorr RG-DIC algorithm).
 *
 * Algorithm:
 * 1. Build a reduced grid of full-resolution points using ncorr's
 *    stride convention: stride = spacing + 1.
 * 2. Place pre-computed seed point(s) onto the grid.
 * 3. Use a min-heap priority queue ordered by ZNSSD (lower = more reliable).
 * 4. Loop:
 *    a. Pop the most reliable point from the queue.
 *    b. For each 4-connected neighbor not yet visited:
 *       - If outside ROI / image bounds: mark calculated, skip.
 *       - Extrapolate initial guess from parent's affine parameters.
 *       - Run ICGN.
 *       - If ICGN passes strict acceptance (ZNSSD < 0.1, delta disp < 1 px):
 *         mark valid & push to queue for further propagation.
 * 5. Output: dense grid with every point marked valid or invalid.
 *
 * Reference: ncorr_alg_rgdic.cpp, ncorr_alg_dicanalysis.m
 */

#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/subset_initializer.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/subset/seed/reliability_propagation.hpp>
#include <dic/subset/solver/forward_gauss_newton.hpp>
#include <dic/subset/solver/icgn.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

namespace dic {
namespace {

// ---------------------------------------------------------------------------
// Queue item: min-heap on ZNSSD (lower = popped first = more reliable).
// ---------------------------------------------------------------------------
struct QueueItem {
    double corrcoef;
    int grid_x, grid_y;
    double u, v;
    double du_dx, du_dy, dv_dx, dv_dy;

    bool operator<(const QueueItem& other) const {
        return corrcoef > other.corrcoef;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
bool in_bounds(int x, int y, int w, int h) {
    return x >= 0 && y >= 0 && x < w && y < h;
}

int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

bool all_finite_6(double u, double v,
                  double dudx, double dudy,
                  double dvdx, double dvdy,
                  double corr) {
    return std::isfinite(u)  && std::isfinite(v) &&
           std::isfinite(dudx) && std::isfinite(dudy) &&
           std::isfinite(dvdx) && std::isfinite(dvdy) &&
           std::isfinite(corr);
}

int nearest_grid_index(int coordinate, int step, int limit)
{
    const int index = static_cast<int>(std::llround(static_cast<double>(coordinate) /
                                                    static_cast<double>(step)));
    return std::clamp(index, 0, limit - 1);
}

BSplinePrecomputedImage build_reference_gradient_cache(const Image& reference, BSplinePrecomputeConfig config)
{
    BSplinePrecomputedImage cache;
    cache.width = reference.width();
    cache.height = reference.height();
    cache.config = config;
    cache.coefficients = Eigen::MatrixXd::Zero(1, 1);
    cache.gradient_x = Eigen::MatrixXd::Zero(cache.height, cache.width);
    cache.gradient_y = Eigen::MatrixXd::Zero(cache.height, cache.width);

    for (int y = 0; y < cache.height; ++y) {
        const int ym = std::max(0, y - 1);
        const int yp = std::min(cache.height - 1, y + 1);
        const double y_denominator = static_cast<double>(std::max(1, yp - ym));
        for (int x = 0; x < cache.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cache.width - 1, x + 1);
            const double x_denominator = static_cast<double>(std::max(1, xp - xm));
            cache.gradient_x(y, x) =
                (static_cast<double>(reference.at(xp, y)) - static_cast<double>(reference.at(xm, y))) /
                x_denominator;
            cache.gradient_y(y, x) =
                (static_cast<double>(reference.at(x, yp)) - static_cast<double>(reference.at(x, ym))) /
                y_denominator;
        }
    }

    return cache;
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ReliabilityPropagation::ReliabilityPropagation(SubsetConfig config)
    : config_(config)
{
    if (config_.propagation_spacing < 0) {
        config_.propagation_spacing = 0;
    }
}

// ---------------------------------------------------------------------------
// propagate
// ---------------------------------------------------------------------------
PropagationResult ReliabilityPropagation::propagate(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const std::vector<PropagationSeed>& seeds
) const
{
    PropagationResult result;

    // --- Input validation ---
    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width()  != deformed.width()  ||
        reference.height() != deformed.height() ||
        reference.width()  != roi.width()        ||
        reference.height() != roi.height()       ||
        config_.subset_radius < 1                ||
        config_.propagation_spacing < 0          ||
        seeds.empty()) {
        return result;
    }

    const int ref_w  = reference.width();
    const int ref_h  = reference.height();
    const int radius = config_.subset_radius;
    const int step   = config_.propagation_spacing + 1;

    // --- Reduced-grid dimensions ---
    // ncorr's spacing parameter is a gap count; its propagation stride is
    // spacing + 1, and output plots use ceil(image_size / stride).
    const int grid_w = ceil_div(ref_w, step);
    const int grid_h = ceil_div(ref_h, step);
    if (grid_w <= 0 || grid_h <= 0) return result;

    const int total_grid = grid_w * grid_h;
    result.grid_width  = grid_w;
    result.grid_height = grid_h;
    result.spacing     = step;
    result.points.resize(total_grid);

    // Mark all grid points as uncalculated / invalid initially
    std::vector<uint8_t> calculated(total_grid, 0);

    // Seed initial positions into the grid
    for (int i = 0; i < total_grid; ++i) {
        result.points[i].x = static_cast<double>((i % grid_w) * step);
        result.points[i].y = static_cast<double>((i / grid_w) * step);
    }

    // --- Shared precomputed data for all solver calls ---
    BSplinePrecomputeConfig bspline_cfg = config_.image_precompute;
    bspline_cfg.precompute_local_blocks = false;
    BSplineImagePreprocessor preprocessor(bspline_cfg);
    const auto reference_precomputed = build_reference_gradient_cache(reference, bspline_cfg);
    const auto deformed_precomputed = preprocessor.compute(deformed);
    const BSplineInterpolator reference_interp(&reference_precomputed);
    const BSplineInterpolator deformed_interp(deformed_precomputed);

    int points_computed = 0;

    const double cutoff_corrcoef_propagation = config_.propagation_max_znssd;
    const double cutoff_disp = static_cast<double>(step);
    std::unique_ptr<SubsetSolver> solver;
    if (config_.optimizer == SubsetOptimizationMethod::ForwardGaussNewton) {
        solver = std::make_unique<ForwardGaussNewtonSolver>(config_);
    } else {
        solver = std::make_unique<ICGNSolver>(config_);
    }

    // -------------------------------------------------------------------
    // Process each seed
    // -------------------------------------------------------------------
    for (const auto& seed : seeds) {
        const int seed_x_raw = static_cast<int>(std::round(seed.point.x()));
        const int seed_y_raw = static_cast<int>(std::round(seed.point.y()));

        if (!in_bounds(seed_x_raw, seed_y_raw, ref_w, ref_h)) continue;
        if (!roi.valid(seed_x_raw, seed_y_raw)) continue;

        const int gx_seed = nearest_grid_index(seed_x_raw, step, grid_w);
        const int gy_seed = nearest_grid_index(seed_y_raw, step, grid_h);
        if (gx_seed < 0 || gx_seed >= grid_w ||
            gy_seed < 0 || gy_seed >= grid_h) continue;

        const int seed_x = gx_seed * step;
        const int seed_y = gy_seed * step;
        if (!in_bounds(seed_x, seed_y, ref_w, ref_h)) continue;
        if (!roi.valid(seed_x, seed_y)) continue;

        const int seed_idx = gy_seed * grid_w + gx_seed;
        if (calculated[seed_idx]) continue;

        InitialDisplacement seed_initial;
        seed_initial.u = seed.displacement.u;
        seed_initial.v = seed.displacement.v;
        seed_initial.du_dx = seed.displacement.du_dx;
        seed_initial.du_dy = seed.displacement.du_dy;
        seed_initial.dv_dx = seed.displacement.dv_dx;
        seed_initial.dv_dy = seed.displacement.dv_dy;
        seed_initial.confidence = seed.displacement.correlation;
        seed_initial.valid = true;

        Displacement2D aligned_seed = seed.displacement;
        if (seed_x != seed_x_raw || seed_y != seed_y_raw) {
            aligned_seed = config_.truncate_roi_subsets
                ? solver->solve_with_mask(
                    reference, deformed,
                    roi,
                    Eigen::Vector2d(static_cast<double>(seed_x), static_cast<double>(seed_y)),
                    seed_initial,
                    reference_interp,
                    deformed_interp)
                : solver->solve_with_interpolators(
                    reference, deformed,
                    Eigen::Vector2d(static_cast<double>(seed_x), static_cast<double>(seed_y)),
                    seed_initial,
                    reference_interp,
                    deformed_interp);
            if (!aligned_seed.valid ||
                aligned_seed.status != SolverStatus::Success ||
                !all_finite_6(aligned_seed.u, aligned_seed.v,
                              aligned_seed.du_dx, aligned_seed.du_dy,
                              aligned_seed.dv_dx, aligned_seed.dv_dy,
                              aligned_seed.correlation)) {
                continue;
            }
        }

        // Place seed
        calculated[seed_idx] = 1;
        result.points[seed_idx]          = aligned_seed;
        result.points[seed_idx].x        = static_cast<double>(seed_x);
        result.points[seed_idx].y        = static_cast<double>(seed_y);
        result.points[seed_idx].valid    = true;
        result.points[seed_idx].status   = SolverStatus::Success;
        ++points_computed;

        QueueItem seed_item;
        seed_item.corrcoef = std::max(0.0, aligned_seed.correlation);
        seed_item.grid_x   = gx_seed;
        seed_item.grid_y   = gy_seed;
        seed_item.u        = aligned_seed.u;
        seed_item.v        = aligned_seed.v;
        seed_item.du_dx    = aligned_seed.du_dx;
        seed_item.du_dy    = aligned_seed.du_dy;
        seed_item.dv_dx    = aligned_seed.dv_dx;
        seed_item.dv_dy    = aligned_seed.dv_dy;

        std::priority_queue<QueueItem> queue;
        queue.push(seed_item);

        // ---------------------------------------------------------------
        // Propagation loop
        // ---------------------------------------------------------------
        while (!queue.empty()) {
            const QueueItem item = queue.top();
            queue.pop();

            const int pop_x = item.grid_x * step;
            const int pop_y = item.grid_y * step;

            // 4-connected neighbors: up, right, down, left
            const int neighbors[4][2] = {
                {pop_x,          pop_y - step},
                {pop_x + step,   pop_y},
                {pop_x,          pop_y + step},
                {pop_x - step,   pop_y}
            };

            for (int n = 0; n < 4; ++n) {
                const int nx = neighbors[n][0];
                const int ny = neighbors[n][1];

                // Bounds
                if (!in_bounds(nx, ny, ref_w, ref_h)) continue;

                const int gnx = nx / step;
                const int gny = ny / step;
                if (gnx < 0 || gny < 0 || gnx >= grid_w || gny >= grid_h) continue;

                const int nidx = gny * grid_w + gnx;
                if (calculated[nidx]) continue;

                // ROI check – mark calculated even if outside
                if (!roi.valid(nx, ny)) {
                    calculated[nidx] = 1;
                    continue;
                }

                // Margin check
                if (nx - radius < 0 || ny - radius < 0 ||
                    nx + radius >= ref_w || ny + radius >= ref_h) {
                    calculated[nidx] = 1;
                    continue;
                }

                // --- Extrapolate initial guess ---
                const double dx = static_cast<double>(nx - pop_x);
                const double dy = static_cast<double>(ny - pop_y);
                const double init_u = item.u + item.du_dx * dx + item.du_dy * dy;
                const double init_v = item.v + item.dv_dx * dx + item.dv_dy * dy;

                // --- ICGN solve ---
                InitialDisplacement initial;
                initial.u      = init_u;
                initial.v      = init_v;
                initial.du_dx  = item.du_dx;
                initial.du_dy  = item.du_dy;
                initial.dv_dx  = item.dv_dx;
                initial.dv_dy  = item.dv_dy;
                initial.valid   = true;

                const auto icgn_result = config_.truncate_roi_subsets
                    ? solver->solve_with_mask(
                        reference, deformed,
                        roi,
                        Eigen::Vector2d(static_cast<double>(nx), static_cast<double>(ny)),
                        initial,
                        reference_interp,
                        deformed_interp)
                    : solver->solve_with_interpolators(
                        reference, deformed,
                        Eigen::Vector2d(static_cast<double>(nx), static_cast<double>(ny)),
                        initial,
                        reference_interp,
                        deformed_interp);

                calculated[nidx] = 1;

                // --- Propagation acceptance (strict, ncorr lines 464-471) ---
                const bool accepted =
                    icgn_result.valid &&
                    icgn_result.status == SolverStatus::Success &&
                    icgn_result.correlation < cutoff_corrcoef_propagation &&
                    std::abs(init_u - icgn_result.u) < cutoff_disp &&
                    std::abs(init_v - icgn_result.v) < cutoff_disp &&
                    all_finite_6(icgn_result.u, icgn_result.v,
                                 icgn_result.du_dx, icgn_result.du_dy,
                                 icgn_result.dv_dx, icgn_result.dv_dy,
                                 icgn_result.correlation);

                if (accepted) {
                    result.points[nidx] = icgn_result;
                    result.points[nidx].x = static_cast<double>(nx);
                    result.points[nidx].y = static_cast<double>(ny);
                    ++points_computed;

                    QueueItem next;
                    next.corrcoef = icgn_result.correlation;
                    next.grid_x   = gnx;
                    next.grid_y   = gny;
                    next.u        = icgn_result.u;
                    next.v        = icgn_result.v;
                    next.du_dx    = icgn_result.du_dx;
                    next.du_dy    = icgn_result.du_dy;
                    next.dv_dx    = icgn_result.dv_dx;
                    next.dv_dy    = icgn_result.dv_dy;
                    queue.push(next);
                }
            }
        }
    }

    result.points_computed = points_computed;
    return result;
}

} // namespace dic
