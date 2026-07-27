#include <dic/initialization/sift_initializer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace dic {

SIFTInitializer::SIFTInitializer(int search_radius)
{
    config_.interpolation_radius = static_cast<double>(std::max(1, search_radius));
}

SIFTInitializer::SIFTInitializer(SIFTInitializerConfig config)
    : config_(config)
{
    config_.interpolation_neighbors = std::max(1, config_.interpolation_neighbors);
    config_.interpolation_radius = std::max(1.0, config_.interpolation_radius);
}

InitialDisplacement SIFTInitializer::estimate(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point) const
{
    const FeatureMatcher matcher(config_.matcher);
    return estimate_from_matches(matcher.match(reference, deformed), point);
}

InitialDisplacement SIFTInitializer::estimate_from_matches(
    const std::vector<FeatureMatch>& matches,
    const Eigen::Vector2d& point) const
{
    std::vector<std::pair<double, const FeatureMatch*>> nearby;
    nearby.reserve(matches.size());
    for (const auto& match : matches) {
        if (!match.valid || !match.robust_inlier) {
            continue;
        }
        const double dx = point.x() - match.reference_point.x();
        const double dy = point.y() - match.reference_point.y();
        const double distance = std::sqrt(dx * dx + dy * dy);
        if (distance <= config_.interpolation_radius) {
            nearby.push_back({distance, &match});
        }
    }

    std::sort(nearby.begin(), nearby.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    if (static_cast<int>(nearby.size()) > config_.interpolation_neighbors) {
        nearby.resize(static_cast<std::size_t>(config_.interpolation_neighbors));
    }
    if (nearby.empty()) {
        return {};
    }

    double sum_w = 0.0;
    double sum_u = 0.0;
    double sum_v = 0.0;
    double sum_conf = 0.0;
    for (const auto& item : nearby) {
        const double weight = 1.0 / (item.first * item.first + 1.0);
        sum_w += weight;
        sum_u += weight * item.second->displacement.x();
        sum_v += weight * item.second->displacement.y();
        sum_conf += weight * item.second->confidence;
    }

    InitialDisplacement result;
    result.u = sum_u / sum_w;
    result.v = sum_v / sum_w;
    result.confidence = sum_conf / sum_w;
    result.valid = true;
    return result;
}

} // namespace dic
