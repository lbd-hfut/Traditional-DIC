#include <dic/initialization/feature_matcher.hpp>

#include <algorithm>
#include <cmath>

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#endif

namespace dic {
namespace {

double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), mid, values.end());
    double result = *mid;
    if (values.size() % 2U == 0U) {
        const auto left = std::max_element(values.begin(), mid);
        result = 0.5 * (result + *left);
    }
    return result;
}

void mark_robust_inliers(std::vector<FeatureMatch>& matches, double mad_factor)
{
    std::vector<double> us;
    std::vector<double> vs;
    for (const auto& match : matches) {
        if (match.mutual) {
            us.push_back(match.displacement.x());
            vs.push_back(match.displacement.y());
        }
    }
    if (us.empty()) {
        return;
    }

    const double med_u = median_value(us);
    const double med_v = median_value(vs);
    std::vector<double> abs_du;
    std::vector<double> abs_dv;
    abs_du.reserve(us.size());
    abs_dv.reserve(vs.size());
    for (double u : us) {
        abs_du.push_back(std::abs(u - med_u));
    }
    for (double v : vs) {
        abs_dv.push_back(std::abs(v - med_v));
    }

    const double gate_u = std::max(3.0, mad_factor * 1.4826 * median_value(abs_du));
    const double gate_v = std::max(3.0, mad_factor * 1.4826 * median_value(abs_dv));
    for (auto& match : matches) {
        match.robust_inlier = match.mutual &&
            std::abs(match.displacement.x() - med_u) <= gate_u &&
            std::abs(match.displacement.y() - med_v) <= gate_v;
        match.valid = match.robust_inlier;
    }
}

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
cv::Mat image_to_u8_mat(const Image& image)
{
    cv::Mat mat(image.height(), image.width(), CV_8U);
    for (int y = 0; y < image.height(); ++y) {
        auto* row = mat.ptr<unsigned char>(y);
        for (int x = 0; x < image.width(); ++x) {
            const double value = std::max(0.0, std::min(1.0, static_cast<double>(image.at(x, y))));
            row[x] = static_cast<unsigned char>(std::lround(value * 255.0));
        }
    }
    return mat;
}
#endif

} // namespace

FeatureMatcher::FeatureMatcher(FeatureMatcherConfig config)
    : config_(config)
{
    config_.max_features = std::max(1, config_.max_features);
    config_.ratio_threshold = std::max(0.0, config_.ratio_threshold);
    config_.robust_mad_factor = std::max(0.0, config_.robust_mad_factor);
}

std::vector<FeatureMatch> FeatureMatcher::match(const Image& reference, const Image& deformed) const
{
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return {};
    }

#if defined(TRADITIONAL_DIC_HAS_OPENCV)
    const auto ref_mat = image_to_u8_mat(reference);
    const auto def_mat = image_to_u8_mat(deformed);
    auto sift = cv::SIFT::create(config_.max_features);

    std::vector<cv::KeyPoint> ref_keypoints;
    std::vector<cv::KeyPoint> def_keypoints;
    cv::Mat ref_descriptors;
    cv::Mat def_descriptors;
    sift->detectAndCompute(ref_mat, cv::noArray(), ref_keypoints, ref_descriptors);
    sift->detectAndCompute(def_mat, cv::noArray(), def_keypoints, def_descriptors);
    if (ref_descriptors.empty() || def_descriptors.empty()) {
        return {};
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> ref_to_def;
    std::vector<std::vector<cv::DMatch>> def_to_ref;
    matcher.knnMatch(ref_descriptors, def_descriptors, ref_to_def, 2);
    matcher.knnMatch(def_descriptors, ref_descriptors, def_to_ref, 2);

    std::vector<int> reverse_best(static_cast<std::size_t>(def_descriptors.rows), -1);
    for (const auto& pair : def_to_ref) {
        if (pair.size() >= 2U && pair[0].distance < config_.ratio_threshold * pair[1].distance) {
            reverse_best[static_cast<std::size_t>(pair[0].queryIdx)] = pair[0].trainIdx;
        }
    }

    std::vector<FeatureMatch> matches;
    for (const auto& pair : ref_to_def) {
        if (pair.size() < 2U) {
            continue;
        }
        const auto& best = pair[0];
        const bool ratio_passed = best.distance < config_.ratio_threshold * pair[1].distance;
        if (!ratio_passed) {
            continue;
        }

        const auto& p0 = ref_keypoints[static_cast<std::size_t>(best.queryIdx)].pt;
        const auto& p1 = def_keypoints[static_cast<std::size_t>(best.trainIdx)].pt;
        FeatureMatch match;
        match.reference_point = {p0.x, p0.y};
        match.deformed_point = {p1.x, p1.y};
        match.displacement = match.deformed_point - match.reference_point;
        match.confidence = 1.0 / (1.0 + static_cast<double>(best.distance));
        match.ratio_passed = true;
        match.mutual = best.trainIdx >= 0 &&
            best.trainIdx < static_cast<int>(reverse_best.size()) &&
            reverse_best[static_cast<std::size_t>(best.trainIdx)] == best.queryIdx;
        matches.push_back(match);
    }
    mark_robust_inliers(matches, config_.robust_mad_factor);
    return matches;
#else
    return {};
#endif
}


} // namespace dic
