#include <dic/initialization/subset_initializer.hpp>
#include <dic/subset/seed/seed_selector.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

namespace dic {
namespace {

struct RoiPoint {
    int x{0};
    int y{0};
};

int ceil_sqrt(int value)
{
    return static_cast<int>(std::ceil(std::sqrt(static_cast<double>(std::max(1, value)))));
}

double distance_squared(double x0, double y0, double x1, double y1)
{
    const double dx = x0 - x1;
    const double dy = y0 - y1;
    return dx * dx + dy * dy;
}

std::vector<RoiPoint> deterministic_sample(const std::vector<RoiPoint>& points, int limit)
{
    if (limit <= 0 || static_cast<int>(points.size()) <= limit) {
        return points;
    }

    std::vector<RoiPoint> sampled;
    sampled.reserve(static_cast<std::size_t>(limit));
    const double step = static_cast<double>(points.size() - 1U) / static_cast<double>(limit - 1);
    for (int i = 0; i < limit; ++i) {
        const auto index = static_cast<std::size_t>(std::llround(static_cast<double>(i) * step));
        sampled.push_back(points[std::min(index, points.size() - 1U)]);
    }
    return sampled;
}

std::vector<Eigen::Vector2d> initialize_centers(const std::vector<RoiPoint>& points, int requested)
{
    std::vector<Eigen::Vector2d> centers;
    if (points.empty() || requested <= 0) {
        return centers;
    }

    const int count = std::min(requested, static_cast<int>(points.size()));
    centers.reserve(static_cast<std::size_t>(count));
    const double step = count == 1
                            ? 0.0
                            : static_cast<double>(points.size() - 1U) / static_cast<double>(count - 1);
    for (int i = 0; i < count; ++i) {
        const auto index = static_cast<std::size_t>(std::llround(static_cast<double>(i) * step));
        const auto& point = points[std::min(index, points.size() - 1U)];
        centers.emplace_back(static_cast<double>(point.x), static_cast<double>(point.y));
    }
    return centers;
}

void run_kmeans(const std::vector<RoiPoint>& points,
                std::vector<Eigen::Vector2d>& centers,
                int iterations)
{
    if (points.empty() || centers.empty() || iterations <= 0) {
        return;
    }

    std::vector<Eigen::Vector2d> sums(centers.size(), Eigen::Vector2d::Zero());
    std::vector<int> counts(centers.size(), 0);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::fill(sums.begin(), sums.end(), Eigen::Vector2d::Zero());
        std::fill(counts.begin(), counts.end(), 0);

        for (const auto& point : points) {
            std::size_t best = 0U;
            double best_distance = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < centers.size(); ++i) {
                const double d2 = distance_squared(
                    static_cast<double>(point.x),
                    static_cast<double>(point.y),
                    centers[i].x(),
                    centers[i].y()
                );
                if (d2 < best_distance) {
                    best_distance = d2;
                    best = i;
                }
            }
            sums[best] += Eigen::Vector2d(static_cast<double>(point.x), static_cast<double>(point.y));
            ++counts[best];
        }

        for (std::size_t i = 0; i < centers.size(); ++i) {
            if (counts[i] > 0) {
                centers[i] = sums[i] / static_cast<double>(counts[i]);
            }
        }
    }
}

} // namespace

SeedSelector::SeedSelector(SubsetConfig config)
    : config_(config)
{
    config_.seed_selection.seed_count = std::max(1, config_.seed_selection.seed_count);
    config_.seed_selection.threads = std::max(1, config_.seed_selection.threads);
    config_.seed_selection.kmeans_iterations = std::max(0, config_.seed_selection.kmeans_iterations);
    config_.seed_selection.kmeans_sample_limit = std::max(1, config_.seed_selection.kmeans_sample_limit);
}

SeedSelectionResult SeedSelector::select_best_seed(const Image& reference,
                                                   const Image& deformed,
                                                   const Mask& roi) const
{
    SeedSelectionResult result;
    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height()) {
        return result;
    }

    const auto points = generate_candidates(reference, roi);
    result.candidates.reserve(points.size());

    SubsetInitializer initializer(config_);
    BSplineInterpolator reference_interpolator(nullptr);
    BSplineInterpolator deformed_interpolator(nullptr);
    std::optional<BSplinePrecomputedImage> deformed_precomputed;
    if (config_.seed_initialization.subpixel.enabled &&
        config_.seed_initialization.subpixel.optimizer == SubsetOptimizationMethod::ICGN) {
        auto precompute_config = config_.image_precompute;
        precompute_config.use_exact_prefilter = false;
        BSplineImagePreprocessor preprocessor(precompute_config);
        deformed_precomputed = preprocessor.compute_lazy(deformed);
        deformed_interpolator = BSplineInterpolator(&(*deformed_precomputed));
    }

    for (const auto& point : points) {
        auto evaluation = evaluate_candidate(
            reference,
            deformed,
            point,
            initializer,
            reference_interpolator,
            deformed_interpolator
        );
        if (evaluation.valid && evaluation.quality_passed &&
            evaluation.displacement_norm >= config_.seed_selection.min_displacement_norm) {
            if (!result.found ||
                evaluation.displacement_norm > result.best_seed.displacement_norm) {
                result.best_seed = evaluation;
                result.found = true;
            }
        }
        result.candidates.push_back(evaluation);
    }

    return result;
}

std::vector<Eigen::Vector2d> SeedSelector::generate_candidates(const Image& reference,
                                                               const Mask& roi) const
{
    std::vector<Eigen::Vector2d> candidates;
    if (reference.empty() || roi.empty() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height()) {
        return candidates;
    }

    std::vector<RoiPoint> points;
    points.reserve(roi.size());
    for (int y = 0; y < roi.height(); ++y) {
        for (int x = 0; x < roi.width(); ++x) {
            if (roi.valid(x, y) && is_candidate_margin_valid(roi, x, y)) {
                points.push_back({x, y});
            }
        }
    }

    if (points.empty()) {
        for (int y = 0; y < roi.height(); ++y) {
            for (int x = 0; x < roi.width(); ++x) {
                if (roi.valid(x, y)) {
                    points.push_back({x, y});
                }
            }
        }
    }
    if (points.empty()) {
        return candidates;
    }

    const int requested = std::max(1, config_.seed_selection.seed_count);
    if (requested == 1) {
        const auto& point = *std::max_element(points.begin(), points.end(), [&](const RoiPoint& lhs, const RoiPoint& rhs) {
            return local_texture_std(reference, lhs.x, lhs.y) < local_texture_std(reference, rhs.x, rhs.y);
        });
        candidates.emplace_back(static_cast<double>(point.x), static_cast<double>(point.y));
        return candidates;
    }

    const auto sampled_points = deterministic_sample(points, config_.seed_selection.kmeans_sample_limit);
    auto centers = initialize_centers(sampled_points, requested);
    run_kmeans(sampled_points, centers, config_.seed_selection.kmeans_iterations);

    candidates.reserve(centers.size());
    const double min_texture_std = config_.seed_selection.min_texture_std;
    for (const auto& center : centers) {
        const int rounded_x = static_cast<int>(std::llround(center.x()));
        const int rounded_y = static_cast<int>(std::llround(center.y()));
        if (roi.valid(rounded_x, rounded_y) &&
            is_candidate_margin_valid(roi, rounded_x, rounded_y) &&
            local_texture_std(reference, rounded_x, rounded_y) >= min_texture_std) {
            candidates.emplace_back(static_cast<double>(rounded_x), static_cast<double>(rounded_y));
            continue;
        }

        std::vector<std::size_t> order(sampled_points.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            const auto& lp = sampled_points[lhs];
            const auto& rp = sampled_points[rhs];
            return distance_squared(static_cast<double>(lp.x), static_cast<double>(lp.y), center.x(), center.y()) <
                   distance_squared(static_cast<double>(rp.x), static_cast<double>(rp.y), center.x(), center.y());
        });

        const std::size_t limit = std::min<std::size_t>(50U, order.size());
        double best_std = -1.0;
        RoiPoint best_point = sampled_points[order.front()];
        for (std::size_t i = 0; i < limit; ++i) {
            const auto& point = sampled_points[order[i]];
            const double texture = local_texture_std(reference, point.x, point.y);
            if (texture > best_std) {
                best_std = texture;
                best_point = point;
            }
        }

        candidates.emplace_back(static_cast<double>(best_point.x), static_cast<double>(best_point.y));
    }

    return candidates;
}

std::vector<Eigen::Vector2d> SeedSelector::generate_uniform_candidates(const Mask& roi) const
{
    std::vector<Eigen::Vector2d> candidates;
    if (roi.empty()) {
        return candidates;
    }

    int left = roi.width();
    int right = -1;
    int top = roi.height();
    int bottom = -1;
    for (int y = 0; y < roi.height(); ++y) {
        for (int x = 0; x < roi.width(); ++x) {
            if (roi.valid(x, y)) {
                left = std::min(left, x);
                right = std::max(right, x);
                top = std::min(top, y);
                bottom = std::max(bottom, y);
            }
        }
    }

    if (right < left || bottom < top) {
        return candidates;
    }

    const int requested = config_.seed_selection.seed_count;
    const int columns = ceil_sqrt(requested);
    const int rows = static_cast<int>(std::ceil(static_cast<double>(requested) / static_cast<double>(columns)));
    const double tile_width = static_cast<double>(right - left + 1) / static_cast<double>(columns);
    const double tile_height = static_cast<double>(bottom - top + 1) / static_cast<double>(rows);

    for (int row = 0; row < rows && static_cast<int>(candidates.size()) < requested; ++row) {
        for (int col = 0; col < columns && static_cast<int>(candidates.size()) < requested; ++col) {
            const double target_x = static_cast<double>(left) + (static_cast<double>(col) + 0.5) * tile_width;
            const double target_y = static_cast<double>(top) + (static_cast<double>(row) + 0.5) * tile_height;

            double best_distance = std::numeric_limits<double>::infinity();
            int best_x = -1;
            int best_y = -1;

            const int x0 = static_cast<int>(std::floor(static_cast<double>(left) + col * tile_width));
            const int x1 = static_cast<int>(std::ceil(static_cast<double>(left) + (col + 1) * tile_width));
            const int y0 = static_cast<int>(std::floor(static_cast<double>(top) + row * tile_height));
            const int y1 = static_cast<int>(std::ceil(static_cast<double>(top) + (row + 1) * tile_height));

            for (int y = std::max(top, y0); y <= std::min(bottom, y1); ++y) {
                for (int x = std::max(left, x0); x <= std::min(right, x1); ++x) {
                    if (!roi.valid(x, y) || !is_candidate_margin_valid(roi, x, y)) {
                        continue;
                    }
                    const double d2 = distance_squared(static_cast<double>(x), static_cast<double>(y), target_x, target_y);
                    if (d2 < best_distance) {
                        best_distance = d2;
                        best_x = x;
                        best_y = y;
                    }
                }
            }

            if (best_x >= 0) {
                candidates.emplace_back(static_cast<double>(best_x), static_cast<double>(best_y));
            }
        }
    }

    return candidates;
}

bool SeedSelector::is_candidate_margin_valid(const Mask& roi, int x, int y) const
{
    const auto& integer_config = config_.seed_initialization.integer_search;
    const auto& subpixel_config = config_.seed_initialization.subpixel;
    const int radius = std::max(integer_config.subset_radius, subpixel_config.subset_radius);
    const int search_radius = integer_config.search_radius;
    const int margin = radius + search_radius;

    if (x - margin < 0 || y - margin < 0 ||
        x + margin >= roi.width() || y + margin >= roi.height()) {
        return false;
    }

    for (int yy = y - radius; yy <= y + radius; ++yy) {
        for (int xx = x - radius; xx <= x + radius; ++xx) {
            const int dx = xx - x;
            const int dy = yy - y;
            if (dx * dx + dy * dy <= radius * radius && !roi.valid(xx, yy)) {
                return false;
            }
        }
    }
    return true;
}

double SeedSelector::local_texture_std(const Image& reference, int x, int y) const
{
    const int radius = config_.seed_initialization.integer_search.subset_radius;
    if (reference.empty() ||
        x - radius < 0 || y - radius < 0 ||
        x + radius >= reference.width() ||
        y + radius >= reference.height()) {
        return 0.0;
    }

    double sum = 0.0;
    double square_sum = 0.0;
    int count = 0;
    for (int yy = y - radius; yy <= y + radius; ++yy) {
        for (int xx = x - radius; xx <= x + radius; ++xx) {
            const double value = static_cast<double>(reference.at(xx, yy));
            sum += value;
            square_sum += value * value;
            ++count;
        }
    }

    if (count == 0) {
        return 0.0;
    }
    const double mean = sum / static_cast<double>(count);
    const double variance = std::max(0.0, square_sum / static_cast<double>(count) - mean * mean);
    return std::sqrt(variance);
}

bool SeedSelector::quality_passes(double quality) const
{
    switch (config_.seed_selection.quality_metric) {
    case SeedQualityMetric::ZNCC:
        return quality >= config_.seed_selection.min_zncc;
    case SeedQualityMetric::SSD:
        return quality <= config_.seed_selection.max_ssd;
    case SeedQualityMetric::ZNSSD:
    default:
        return quality <= config_.seed_selection.max_znssd;
    }
}

SeedEvaluation SeedSelector::evaluate_candidate(const Image& reference,
                                                const Image& deformed,
                                                const Eigen::Vector2d& point,
                                                const SubsetInitializer& initializer,
                                                const BSplineInterpolator& reference_interpolator,
                                                const BSplineInterpolator& deformed_interpolator) const
{
    SeedEvaluation evaluation;
    evaluation.point = point;

    const auto initial = initializer.estimate_with_interpolators(
        reference,
        deformed,
        point,
        reference_interpolator,
        deformed_interpolator
    );
    if (!initial.valid ||
        !std::isfinite(initial.u) ||
        !std::isfinite(initial.v) ||
        !std::isfinite(initial.confidence)) {
        return evaluation;
    }

    evaluation.displacement.x = point.x();
    evaluation.displacement.y = point.y();
    evaluation.displacement.u = initial.u;
    evaluation.displacement.v = initial.v;
    evaluation.displacement.du_dx = initial.du_dx;
    evaluation.displacement.du_dy = initial.du_dy;
    evaluation.displacement.dv_dx = initial.dv_dx;
    evaluation.displacement.dv_dy = initial.dv_dy;
    evaluation.displacement.correlation = initial.confidence;
    evaluation.displacement.status = SolverStatus::Success;
    evaluation.displacement.valid = true;
    evaluation.quality = initial.confidence;
    evaluation.displacement_norm = std::hypot(initial.u, initial.v);
    evaluation.quality_passed = quality_passes(evaluation.quality);
    evaluation.valid = true;
    return evaluation;
}

} // namespace dic
