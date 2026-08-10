#include <dic/mesh/initialization/pyramid_initializer.hpp>
#include <dic/mesh/initialization/fedic_fft_initializer.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace dic::mesh {
namespace {

#ifdef TRADITIONAL_DIC_HAS_OPENCV
cv::Mat to_cv_mat(const Image& image)
{
    cv::Mat m(image.height(), image.width(), CV_32F);
    for (int row = 0; row < image.height(); ++row) {
        for (int col = 0; col < image.width(); ++col) {
            m.at<float>(row, col) = image.at(col, row);
        }
    }
    return m;
}

Image image_from_cv_mat(const cv::Mat& m)
{
    std::vector<float> data;
    data.reserve(static_cast<std::size_t>(m.cols) * static_cast<std::size_t>(m.rows));
    for (int row = 0; row < m.rows; ++row) {
        for (int col = 0; col < m.cols; ++col) {
            data.push_back(m.at<float>(row, col));
        }
    }
    return Image(m.cols, m.rows, std::move(data));
}
#endif

} // namespace

std::vector<InitialDisplacement> estimate_pyramid_initial_displacements(
    const Image& reference,
    const Image& deformed,
    const std::vector<Eigen::Vector2d>& points,
    const PyramidInitializationConfig& cfg)
{
    std::vector<InitialDisplacement> result;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return result;
    }
    result.resize(points.size());

#ifdef TRADITIONAL_DIC_HAS_OPENCV
    if (!(cfg.scale_factor > 0.0 && cfg.scale_factor < 1.0) ||
        cfg.coarse_search_radius < 0 || cfg.refinement_radius < 0) {
        return result; // all invalid -> blind-search fallback
    }

    // ---- 1. Build the pyramid with adaptive truncation ----
    // Keep enough margin around the coarsest-level search window so the
    // coarse search stays inside the image. The window scales with the level
    // so a coarse level uses a small patch and stays inside a small image
    // even with a large search radius.
    const int window_size = std::max(1, cfg.window_size);
    const int coarse_radius = std::max(0, cfg.coarse_search_radius);
    const int min_window = 16;
    const int max_levels = std::max(1, cfg.num_levels);
    const auto level_window = [&](double s) {
        return std::max(min_window, static_cast<int>(std::llround(window_size * s)));
    };

    std::vector<Image> ref_levels;
    std::vector<Image> def_levels;
    std::vector<double> scales;
    ref_levels.push_back(reference);
    def_levels.push_back(deformed);
    scales.push_back(1.0);
    for (int level = 1; level < max_levels; ++level) {
        const double s = scales.back() * cfg.scale_factor;
        const int w = static_cast<int>(std::llround(reference.width() * s));
        const int h = static_cast<int>(std::llround(reference.height() * s));
        if (w < 1 || h < 1) break;
        const int win = level_window(s);
        if (std::min(h, w) < win + 2 * coarse_radius + 2) break;
        cv::Mat ref_mat = to_cv_mat(ref_levels.back());
        cv::Mat def_mat = to_cv_mat(def_levels.back());
        cv::Mat ref_down, def_down;
        cv::resize(ref_mat, ref_down, cv::Size(w, h), 0.0, 0.0, cv::INTER_AREA);
        cv::resize(def_mat, def_down, cv::Size(w, h), 0.0, 0.0, cv::INTER_AREA);
        ref_levels.push_back(image_from_cv_mat(ref_down));
        def_levels.push_back(image_from_cv_mat(def_down));
        scales.push_back(s);
    }

    const int n_levels = static_cast<int>(scales.size());
    if (n_levels < 2) {
        return result; // image too small for a meaningful pyramid
    }

    // ---- 2. Per-node coarse-to-fine ----
    const int coarsest = n_levels - 1;
    for (std::size_t i = 0; i < points.size(); ++i) {
        InitialDisplacement d;
        d.valid = false;

        // Coarsest level: blind search with the coarse radius.
        int level = coarsest;
        const double s_coarse = scales[level];
        const Eigen::Vector2d pt_coarse(points[i].x() * s_coarse, points[i].y() * s_coarse);
        const Image& ref_coarse = ref_levels[level];
        const Image& def_coarse = def_levels[level];
        if (pt_coarse.x() < 0.0 || pt_coarse.y() < 0.0 ||
            pt_coarse.x() >= static_cast<double>(ref_coarse.width()) ||
            pt_coarse.y() >= static_cast<double>(ref_coarse.height())) {
            result[i] = d;
            continue;
        }
        const auto coarse = estimate_fedic_fft_initial_displacement(
            ref_coarse, def_coarse, pt_coarse, cfg.coarse_search_radius, level_window(s_coarse));
        if (!coarse.initial.valid) {
            result[i] = d;
            continue;
        }
        d = coarse.initial;

        // Refine upward: the coarser displacement, scaled back to this level,
        // seeds the search center; a small radius then polishes it.
        for (--level; level >= 0; --level) {
            const double s = scales[level];
            const Eigen::Vector2d pt_cur(points[i].x() * s, points[i].y() * s);
            const Image& ref_cur = ref_levels[level];
            const Image& def_cur = def_levels[level];
            if (pt_cur.x() < 0.0 || pt_cur.y() < 0.0 ||
                pt_cur.x() >= static_cast<double>(ref_cur.width()) ||
                pt_cur.y() >= static_cast<double>(ref_cur.height())) {
                d.valid = false;
                break;
            }
            const Eigen::Vector2d offset(d.u / cfg.scale_factor, d.v / cfg.scale_factor);
            const auto refined = estimate_fedic_fft_initial_displacement(
                ref_cur, def_cur, pt_cur, cfg.refinement_radius, level_window(s), offset);
            if (!refined.initial.valid) {
                d.valid = false;
                break;
            }
            d = refined.initial;
        }
        result[i] = d;
    }
    return result;
#else
    (void)points;
    (void)cfg;
    return result;
#endif
}

} // namespace dic::mesh
