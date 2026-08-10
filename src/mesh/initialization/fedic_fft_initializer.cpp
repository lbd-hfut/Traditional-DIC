#include <dic/mesh/initialization/fedic_fft_initializer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace dic::mesh {
namespace {

void refine_quadratic_peak(const cv::Mat& correlation, cv::Point peak,
                           double& offset_x, double& offset_y, double& peak_value)
{
    if (peak.x <= 0 || peak.y <= 0 ||
        peak.x >= correlation.cols - 1 || peak.y >= correlation.rows - 1) {
        return;
    }
    // Same 3x3 least-squares quadratic used in FE-DIC's findpeak(..., 1).
    Eigen::Matrix<double, 9, 6> design;
    Eigen::Matrix<double, 9, 1> samples;
    int index = 0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            design.row(index) << 1.0, x, y, x * y, x * x, y * y;
            samples(index) = correlation.at<float>(peak.y + y, peak.x + x);
            ++index;
        }
    }
    const Eigen::Matrix<double, 6, 1> coefficients =
        design.colPivHouseholderQr().solve(samples);
    const double denominator = coefficients(3) * coefficients(3) -
                               4.0 * coefficients(4) * coefficients(5);
    if (!coefficients.allFinite() || std::abs(denominator) < 1.0e-12) {
        return;
    }
    const double candidate_x = (-coefficients(2) * coefficients(3) +
                                2.0 * coefficients(5) * coefficients(1)) / denominator;
    const double candidate_y = (-coefficients(3) * coefficients(1) +
                                2.0 * coefficients(4) * coefficients(2)) / denominator;
    if (!std::isfinite(candidate_x) || !std::isfinite(candidate_y) ||
        std::abs(candidate_x) > 1.0 || std::abs(candidate_y) > 1.0) {
        return;
    }
    offset_x = std::round(candidate_x * 1000.0) / 1000.0;
    offset_y = std::round(candidate_y * 1000.0) / 1000.0;
    Eigen::Matrix<double, 1, 6> polynomial;
    polynomial << 1.0, offset_x, offset_y, offset_x * offset_y,
                  offset_x * offset_x, offset_y * offset_y;
    peak_value = (polynomial * coefficients)(0);
}

} // namespace

FEDICFFTInitialDisplacement estimate_fedic_fft_initial_displacement(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    int search_radius,
    int window_size,
    const Eigen::Vector2d& initial_offset)
{
    FEDICFFTInitialDisplacement result;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return result;
    }

    const int x = static_cast<int>(std::llround(point.x()));
    const int y = static_cast<int>(std::llround(point.y()));
    search_radius = std::max(0, search_radius);
    window_size = std::max(1, window_size);
    const int x0 = static_cast<int>(std::ceil(point.x() - 0.5 * window_size));
    const int x1 = static_cast<int>(std::floor(point.x() + 0.5 * window_size));
    const int y0 = static_cast<int>(std::ceil(point.y() - 0.5 * window_size));
    const int y1 = static_cast<int>(std::floor(point.y() + 0.5 * window_size));
    // Search-window origin shifted by the coarse pyramid seed so large
    // disparities are covered by a small radius around (u0, v0) instead of
    // a blind (0,0) search.
    const int u0 = static_cast<int>(std::llround(initial_offset.x()));
    const int v0 = static_cast<int>(std::llround(initial_offset.y()));
    const int sx0 = x0 + u0;
    const int sy0 = y0 + v0;
    const int sx1 = x1 + u0;
    const int sy1 = y1 + v0;
    if (sx0 - search_radius < 0 || sy0 - search_radius < 0 ||
        sx1 + search_radius >= reference.width() || sy1 + search_radius >= reference.height()) {
        return result;
    }

#ifdef TRADITIONAL_DIC_HAS_OPENCV
    const int patch_width = x1 - x0 + 1;
    const int patch_height = y1 - y0 + 1;
    const int search_width = patch_width + 2 * search_radius;
    const int search_height = patch_height + 2 * search_radius;
    cv::Mat reference_patch(patch_height, patch_width, CV_32F);
    cv::Mat deformed_search(search_height, search_width, CV_32F);
    for (int row = 0; row < patch_height; ++row) {
        for (int col = 0; col < patch_width; ++col) {
            reference_patch.at<float>(row, col) = static_cast<float>(
                reference.at(x0 + col, y0 + row));
        }
    }
    for (int row = 0; row < search_height; ++row) {
        for (int col = 0; col < search_width; ++col) {
            deformed_search.at<float>(row, col) = static_cast<float>(
                deformed.at(sx0 - search_radius + col, sy0 - search_radius + row));
        }
    }

    // This is the normalized cross-correlation surface used by FE-DIC's
    // normxcorr2 route. OpenCV selects its DFT correlation backend where
    // profitable, while preserving the normalized peak definition.
    cv::Mat correlation;
    cv::matchTemplate(
        deformed_search, reference_patch, correlation, cv::TM_CCOEFF_NORMED);
    double peak = -std::numeric_limits<double>::infinity();
    cv::Point peak_location;
    cv::minMaxLoc(correlation, nullptr, &peak, nullptr, &peak_location);
    if (!std::isfinite(peak)) {
        return result;
    }
    double peak_offset_x = 0.0;
    double peak_offset_y = 0.0;
    refine_quadratic_peak(correlation, peak_location, peak_offset_x, peak_offset_y, peak);

    double minimum = 0.0;
    cv::minMaxLoc(correlation, &minimum, nullptr);
    cv::Mat shifted = correlation - minimum;
    const double energy = cv::sum(shifted.mul(shifted))[0] /
                          static_cast<double>(shifted.total());
    const double shifted_peak = peak - minimum;
    result.peak_to_correlation_energy = energy > 1.0e-12
        ? (shifted_peak * shifted_peak) / energy : 0.0;

    constexpr int bins = 30;
    if (shifted_peak > 0.0) {
        cv::Mat histogram;
        const float range[] = {0.0F, static_cast<float>(shifted_peak)};
        const float* ranges[] = {range};
        const int channels[] = {0};
        const int histogram_size[] = {bins};
        cv::calcHist(&shifted, 1, channels, cv::Mat(), histogram, 1,
                     histogram_size, ranges, true, false);
        double entropy = 0.0;
        const double total = static_cast<double>(shifted.total());
        for (int i = 0; i < bins; ++i) {
            const double probability = histogram.at<float>(i) / total;
            if (probability > 0.0) {
                entropy += probability * std::log(1.0 / probability);
            }
        }
        result.peak_to_entropy = entropy > 1.0e-12 ? 1.0 / entropy : 0.0;
    }

    result.initial.u = initial_offset.x() +
        static_cast<double>(peak_location.x - search_radius) + peak_offset_x;
    result.initial.v = initial_offset.y() +
        static_cast<double>(peak_location.y - search_radius) + peak_offset_y;
    result.initial.zncc = peak;
    result.initial.znssd = std::max(0.0, 2.0 * (1.0 - std::clamp(peak, -1.0, 1.0)));
    result.initial.confidence = result.initial.znssd;
    result.initial.valid = true;
    return result;
#else
    (void)search_radius;
    (void)window_size;
    (void)initial_offset;
    return result;
#endif
}

} // namespace dic::mesh
