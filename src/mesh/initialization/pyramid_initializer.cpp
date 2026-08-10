#include <dic/mesh/initialization/pyramid_initializer.hpp>
#include <dic/mesh/initialization/fedic_fft_initializer.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace dic::mesh {
namespace {
#ifdef TRADITIONAL_DIC_HAS_OPENCV
cv::Mat to_cv_mat(const Image& image) {
    cv::Mat result(image.height(), image.width(), CV_32F);
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x) result.at<float>(y, x) = image.at(x, y);
    return result;
}

Image from_cv_mat(const cv::Mat& image) {
    std::vector<float> data;
    data.reserve(static_cast<std::size_t>(image.rows) * image.cols);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) data.push_back(image.at<float>(y, x));
    return Image(image.cols, image.rows, std::move(data));
}
#endif
} // namespace

std::vector<InitialDisplacement> estimate_pyramid_initial_displacements(
    const Image& reference, const Image& deformed,
    const std::vector<Eigen::Vector2d>& points,
    const PyramidInitializationConfig& cfg)
{
    std::vector<InitialDisplacement> result(points.size());
#ifdef TRADITIONAL_DIC_HAS_OPENCV
    if (reference.empty() || deformed.empty() || reference.width() != deformed.width() ||
        reference.height() != deformed.height() || !(cfg.scale_factor > 0.0 && cfg.scale_factor < 1.0)) return result;
    std::vector<Image> refs{reference}, defs{deformed};
    std::vector<double> scales{1.0};
    for (int level = 1; level < std::max(1, cfg.num_levels); ++level) {
        const double scale = scales.back() * cfg.scale_factor;
        const int width = static_cast<int>(std::llround(reference.width() * scale));
        const int height = static_cast<int>(std::llround(reference.height() * scale));
        const int window = std::max(16, static_cast<int>(std::llround(cfg.window_size * scale)));
        if (width < window + 2 * cfg.coarse_search_radius + 2 || height < window + 2 * cfg.coarse_search_radius + 2) break;
        cv::Mat ref_down, def_down;
        cv::resize(to_cv_mat(refs.back()), ref_down, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
        cv::resize(to_cv_mat(defs.back()), def_down, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
        refs.push_back(from_cv_mat(ref_down)); defs.push_back(from_cv_mat(def_down)); scales.push_back(scale);
    }
    if (scales.size() < 2) return result;
    for (std::size_t i = 0; i < points.size(); ++i) {
        int level = static_cast<int>(scales.size()) - 1;
        const auto window = [&](double scale) { return std::max(16, static_cast<int>(std::llround(cfg.window_size * scale))); };
        auto displacement = estimate_fedic_fft_initial_displacement(refs[level], defs[level], points[i] * scales[level], cfg.coarse_search_radius, window(scales[level])).initial;
        for (--level; displacement.valid && level >= 0; --level) {
            displacement = estimate_fedic_fft_initial_displacement(refs[level], defs[level], points[i] * scales[level], cfg.refinement_radius, window(scales[level]), Eigen::Vector2d(displacement.u / cfg.scale_factor, displacement.v / cfg.scale_factor)).initial;
        }
        result[i] = displacement;
    }
#else
    (void)reference; (void)deformed; (void)cfg;
#endif
    return result;
}
} // namespace dic::mesh
