#include <dic/initialization/integer_search.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dic {
namespace {

constexpr double kEpsilon = 1e-12;
constexpr double kLocalPreferenceZnccMargin = 0.02;

struct CandidateScore {
    int u{0};
    int v{0};
    double zncc{-std::numeric_limits<double>::infinity()};
};

struct SubsetSample {
    int dx{0};
    int dy{0};
    double reference_value{0.0};
};

double znssd_from_zncc(double zncc)
{
    if (!std::isfinite(zncc)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, 2.0 * (1.0 - std::max(-1.0, std::min(1.0, zncc))));
}

int full_circle_sample_count(int radius, int sample_step = 1)
{
    int count = 0;
    sample_step = std::max(1, sample_step);
    for (int dy = -radius; dy <= radius; dy += sample_step) {
        for (int dx = -radius; dx <= radius; dx += sample_step) {
            if (dx * dx + dy * dy <= radius * radius) {
                ++count;
            }
        }
    }
    return count;
}

int min_required_samples(int radius, int sample_step = 1)
{
    return std::max(12, static_cast<int>(std::ceil(0.5 * full_circle_sample_count(radius, sample_step))));
}

bool subset_in_bounds(const Image& image, int center_x, int center_y, int radius)
{
    return center_x - radius >= 0 &&
           center_y - radius >= 0 &&
           center_x + radius < image.width() &&
           center_y + radius < image.height();
}

std::vector<SubsetSample> collect_circular_reference_samples(
    const Image& reference,
    int center_x,
    int center_y,
    int radius,
    int sample_step = 1
)
{
    sample_step = std::max(1, sample_step);
    std::vector<SubsetSample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));
    for (int dy = -radius; dy <= radius; dy += sample_step) {
        for (int dx = -radius; dx <= radius; dx += sample_step) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            samples.push_back({dx, dy, static_cast<double>(reference.at(center_x + dx, center_y + dy))});
        }
    }
    return samples;
}

std::vector<SubsetSample> collect_masked_reference_samples(
    const Image& reference,
    const Mask& roi,
    int center_x,
    int center_y,
    int radius,
    int sample_step = 1
)
{
    sample_step = std::max(1, sample_step);
    std::vector<SubsetSample> samples;
    samples.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));
    for (int dy = -radius; dy <= radius; dy += sample_step) {
        for (int dx = -radius; dx <= radius; dx += sample_step) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (!reference.contains(x, y) || !roi.valid(x, y)) {
                continue;
            }
            samples.push_back({dx, dy, static_cast<double>(reference.at(x, y))});
        }
    }
    return samples;
}

double sample_mean(const std::vector<SubsetSample>& samples)
{
    double mean = 0.0;
    for (const auto& sample : samples) {
        mean += sample.reference_value;
    }
    return samples.empty() ? 0.0 : mean / static_cast<double>(samples.size());
}

double sample_norm(const std::vector<SubsetSample>& samples, double mean)
{
    double norm = 0.0;
    for (const auto& sample : samples) {
        const double diff = sample.reference_value - mean;
        norm += diff * diff;
    }
    return norm;
}

double circular_subset_zncc(
    const Image& deformed,
    const std::vector<SubsetSample>& samples,
    double ref_mean,
    double ref_norm,
    int center_x,
    int center_y,
    int u,
    int v
)
{
    if (samples.empty() || ref_norm <= kEpsilon) {
        return -std::numeric_limits<double>::infinity();
    }

    double def_mean = 0.0;
    for (const auto& sample : samples) {
        if (!deformed.contains(center_x + u + sample.dx, center_y + v + sample.dy)) {
            return -std::numeric_limits<double>::infinity();
        }
        def_mean += static_cast<double>(deformed.at(center_x + u + sample.dx, center_y + v + sample.dy));
    }
    def_mean /= static_cast<double>(samples.size());

    double numerator = 0.0;
    double def_norm = 0.0;
    for (const auto& sample : samples) {
        const double ref_diff = sample.reference_value - ref_mean;
        const double def_diff = static_cast<double>(deformed.at(center_x + u + sample.dx, center_y + v + sample.dy)) - def_mean;
        numerator += ref_diff * def_diff;
        def_norm += def_diff * def_diff;
    }

    if (def_norm <= kEpsilon) {
        return -std::numeric_limits<double>::infinity();
    }
    return numerator / std::sqrt(ref_norm * def_norm);
}

CandidateScore search_displacements(
    const Image& deformed,
    const std::vector<SubsetSample>& samples,
    double ref_mean,
    double ref_norm,
    int center_x,
    int center_y,
    int subset_radius,
    int min_u,
    int max_u,
    int min_v,
    int max_v,
    int displacement_step
)
{
    CandidateScore best;
    displacement_step = std::max(1, displacement_step);
    for (int v = min_v; v <= max_v; v += displacement_step) {
        for (int u = min_u; u <= max_u; u += displacement_step) {
            if (!subset_in_bounds(deformed, center_x + u, center_y + v, subset_radius)) {
                continue;
            }
            const double zncc = circular_subset_zncc(
                deformed,
                samples,
                ref_mean,
                ref_norm,
                center_x,
                center_y,
                u,
                v
            );
            if (zncc > best.zncc) {
                best = {u, v, zncc};
            }
        }
    }
    return best;
}

CandidateScore search_ncorr_multigrid(
    const Image& deformed,
    const std::vector<SubsetSample>& samples,
    double ref_mean,
    double ref_norm,
    const std::vector<SubsetSample>& coarse_samples,
    double coarse_ref_mean,
    double coarse_ref_norm,
    int center_x,
    int center_y,
    int subset_radius,
    int coarse_step,
    int refinement_radius
)
{
    const int min_u = subset_radius - center_x;
    const int max_u = deformed.width() - 1 - subset_radius - center_x;
    const int min_v = subset_radius - center_y;
    const int max_v = deformed.height() - 1 - subset_radius - center_y;
    if (min_u > max_u || min_v > max_v) {
        return {};
    }

    coarse_step = std::max(1, coarse_step);
    CandidateScore coarse = search_displacements(
        deformed,
        coarse_samples,
        coarse_ref_mean,
        coarse_ref_norm,
        center_x,
        center_y,
        subset_radius,
        min_u,
        max_u,
        min_v,
        max_v,
        coarse_step
    );
    if (!std::isfinite(coarse.zncc)) {
        return coarse;
    }

    refinement_radius = std::max(refinement_radius, coarse_step);
    return search_displacements(
        deformed,
        samples,
        ref_mean,
        ref_norm,
        center_x,
        center_y,
        subset_radius,
        std::max(min_u, coarse.u - refinement_radius),
        std::min(max_u, coarse.u + refinement_radius),
        std::max(min_v, coarse.v - refinement_radius),
        std::min(max_v, coarse.v + refinement_radius),
        1
    );
}

CandidateScore prefer_local_when_ambiguous(const CandidateScore& global_best,
                                           const CandidateScore& local_best)
{
    if (!std::isfinite(local_best.zncc)) {
        return global_best;
    }
    if (!std::isfinite(global_best.zncc)) {
        return local_best;
    }
    if (local_best.zncc + kLocalPreferenceZnccMargin >= global_best.zncc) {
        return local_best;
    }
    return global_best;
}

} // namespace

IntegerSearchInitializer::IntegerSearchInitializer(int search_radius)
    : IntegerSearchInitializer(search_radius, 15)
{
}

IntegerSearchInitializer::IntegerSearchInitializer(int search_radius, int subset_radius)
{
    config_.method = SeedInitializationMethod::IntegerSearch;
    config_.integer_search.search_radius = std::max(0, search_radius);
    config_.integer_search.subset_radius = std::max(1, subset_radius);
}

IntegerSearchInitializer::IntegerSearchInitializer(SeedInitializationConfig config)
    : IntegerSearchInitializer(config, BSplinePrecomputeConfig{})
{
}

IntegerSearchInitializer::IntegerSearchInitializer(SeedInitializationConfig config,
                                                   BSplinePrecomputeConfig image_precompute)
    : config_(config),
      image_precompute_(image_precompute)
{
    config_.integer_search.search_radius = std::max(0, config_.integer_search.search_radius);
    config_.integer_search.subset_radius = std::max(1, config_.integer_search.subset_radius);
    config_.integer_search.pyramid_scale = std::max(1, config_.integer_search.pyramid_scale);
    config_.integer_search.pyramid_refinement_radius =
        std::max(1, config_.integer_search.pyramid_refinement_radius);
    config_.subpixel.subset_radius = std::max(1, config_.subpixel.subset_radius);
    config_.subpixel.max_iterations = std::max(0, config_.subpixel.max_iterations);
}

InitialDisplacement IntegerSearchInitializer::estimate(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point
) const
{
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return invalid;
    }

    BSplineInterpolator reference_interpolator(nullptr);
    BSplineInterpolator deformed_interpolator(nullptr);
    return estimate_with_interpolators(reference, deformed, point, reference_interpolator, deformed_interpolator);
}

InitialDisplacement IntegerSearchInitializer::estimate_with_interpolators(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height()) {
        return invalid;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    const auto& integer_config = config_.integer_search;
    if (!subset_in_bounds(reference, center_x, center_y, integer_config.subset_radius)) {
        return invalid;
    }

    const auto samples = collect_circular_reference_samples(
        reference,
        center_x,
        center_y,
        integer_config.subset_radius
    );
    const double ref_mean = sample_mean(samples);
    const double ref_norm = sample_norm(samples, ref_mean);
    if (samples.empty() || ref_norm <= kEpsilon) {
        return invalid;
    }

    const int search_radius = integer_config.search_radius;
    const int coarse_step = integer_config.pyramid_enabled
                                ? std::max(1, integer_config.pyramid_scale)
                                : 1;
    const auto coarse_samples = collect_circular_reference_samples(
        reference,
        center_x,
        center_y,
        integer_config.subset_radius,
        coarse_step
    );
    const double coarse_ref_mean = sample_mean(coarse_samples);
    const double coarse_ref_norm = sample_norm(coarse_samples, coarse_ref_mean);
    CandidateScore best;
    if (!coarse_samples.empty() && coarse_ref_norm > kEpsilon) {
        best = search_ncorr_multigrid(
            deformed,
            samples,
            ref_mean,
            ref_norm,
            coarse_samples,
            coarse_ref_mean,
            coarse_ref_norm,
            center_x,
            center_y,
            integer_config.subset_radius,
            coarse_step,
            std::max(integer_config.pyramid_refinement_radius,
                     static_cast<int>(std::ceil(1.5 * static_cast<double>(integer_config.subset_radius))))
        );
    }

    if (search_radius > 0) {
        const auto local_best = search_displacements(
            deformed,
            samples,
            ref_mean,
            ref_norm,
            center_x,
            center_y,
            integer_config.subset_radius,
            -search_radius,
            search_radius,
            -search_radius,
            search_radius,
            1
        );
        best = prefer_local_when_ambiguous(best, local_best);
    }

    if (!std::isfinite(best.zncc)) {
        return invalid;
    }

    InitialDisplacement integer_initial;
    integer_initial.u = static_cast<double>(best.u);
    integer_initial.v = static_cast<double>(best.v);
    integer_initial.confidence = znssd_from_zncc(best.zncc);
    integer_initial.valid = true;
    integer_initial.zncc = best.zncc;
    integer_initial.znssd = integer_initial.confidence;

    (void)reference_interpolator;
    (void)deformed_interpolator;
    return integer_initial;
}

InitialDisplacement IntegerSearchInitializer::estimate_with_mask(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point
) const
{
    BSplineInterpolator reference_interpolator(nullptr);
    BSplineInterpolator deformed_interpolator(nullptr);
    return estimate_with_mask_interpolators(
        reference, deformed, roi, point, reference_interpolator, deformed_interpolator);
}

InitialDisplacement IntegerSearchInitializer::estimate_with_mask_interpolators(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    const BSplineInterpolator& reference_interpolator,
    const BSplineInterpolator& deformed_interpolator
) const
{
    (void)reference_interpolator;
    (void)deformed_interpolator;

    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() || roi.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height()) {
        return invalid;
    }

    const int center_x = static_cast<int>(std::round(point.x()));
    const int center_y = static_cast<int>(std::round(point.y()));
    if (!roi.valid(center_x, center_y)) {
        return invalid;
    }

    const auto& integer_config = config_.integer_search;
    const auto samples = collect_masked_reference_samples(
        reference, roi, center_x, center_y, integer_config.subset_radius);
    if (static_cast<int>(samples.size()) < min_required_samples(integer_config.subset_radius)) {
        return invalid;
    }

    const double ref_mean = sample_mean(samples);
    const double ref_norm = sample_norm(samples, ref_mean);
    if (ref_norm <= kEpsilon) {
        return invalid;
    }

    const int search_radius = integer_config.search_radius;
    const int coarse_step = integer_config.pyramid_enabled
                                ? std::max(1, integer_config.pyramid_scale)
                                : 1;
    const auto coarse_samples = collect_masked_reference_samples(
        reference, roi, center_x, center_y, integer_config.subset_radius, coarse_step);
    const double coarse_ref_mean = sample_mean(coarse_samples);
    const double coarse_ref_norm = sample_norm(coarse_samples, coarse_ref_mean);
    CandidateScore best;
    if (static_cast<int>(coarse_samples.size()) >= min_required_samples(integer_config.subset_radius, coarse_step) &&
        coarse_ref_norm > kEpsilon) {
        best = search_ncorr_multigrid(
            deformed,
            samples,
            ref_mean,
            ref_norm,
            coarse_samples,
            coarse_ref_mean,
            coarse_ref_norm,
            center_x,
            center_y,
            integer_config.subset_radius,
            coarse_step,
            std::max(integer_config.pyramid_refinement_radius,
                     static_cast<int>(std::ceil(1.5 * static_cast<double>(integer_config.subset_radius))))
        );
    }

    if (search_radius > 0) {
        const auto local_best = search_displacements(
            deformed,
            samples,
            ref_mean,
            ref_norm,
            center_x,
            center_y,
            integer_config.subset_radius,
            -search_radius,
            search_radius,
            -search_radius,
            search_radius,
            1
        );
        best = prefer_local_when_ambiguous(best, local_best);
    }

    if (!std::isfinite(best.zncc)) {
        return invalid;
    }

    InitialDisplacement integer_initial;
    integer_initial.u = static_cast<double>(best.u);
    integer_initial.v = static_cast<double>(best.v);
    integer_initial.confidence = znssd_from_zncc(best.zncc);
    integer_initial.valid = true;
    integer_initial.zncc = best.zncc;
    integer_initial.znssd = integer_initial.confidence;
    return integer_initial;
}

InitialDisplacement IntegerSearchInitializer::estimate_around_displacement(
    const Image& reference,
    const Image& deformed,
    const Eigen::Vector2d& point,
    double initial_u,
    double initial_v) const
{
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        !std::isfinite(initial_u) || !std::isfinite(initial_v)) {
        return invalid;
    }

    const auto& integer_config = config_.integer_search;
    const int center_x = static_cast<int>(std::llround(point.x()));
    const int center_y = static_cast<int>(std::llround(point.y()));
    const auto samples = collect_circular_reference_samples(
        reference, center_x, center_y, integer_config.subset_radius, 1);
    if (static_cast<int>(samples.size()) < min_required_samples(integer_config.subset_radius, 1)) {
        return invalid;
    }
    const double ref_mean = sample_mean(samples);
    const double ref_norm = sample_norm(samples, ref_mean);
    if (ref_norm <= kEpsilon) {
        return invalid;
    }

    const int search_radius = std::max(0, integer_config.search_radius);
    const int prior_u = static_cast<int>(std::llround(initial_u));
    const int prior_v = static_cast<int>(std::llround(initial_v));
    const int min_u = integer_config.subset_radius - center_x;
    const int max_u = deformed.width() - 1 - integer_config.subset_radius - center_x;
    const int min_v = integer_config.subset_radius - center_y;
    const int max_v = deformed.height() - 1 - integer_config.subset_radius - center_y;
    const auto best = search_displacements(
        deformed,
        samples,
        ref_mean,
        ref_norm,
        center_x,
        center_y,
        integer_config.subset_radius,
        std::max(min_u, prior_u - search_radius),
        std::min(max_u, prior_u + search_radius),
        std::max(min_v, prior_v - search_radius),
        std::min(max_v, prior_v + search_radius),
        1
    );
    if (!std::isfinite(best.zncc)) {
        return invalid;
    }

    InitialDisplacement result;
    result.u = static_cast<double>(best.u);
    result.v = static_cast<double>(best.v);
    result.confidence = znssd_from_zncc(best.zncc);
    result.valid = true;
    result.zncc = best.zncc;
    result.znssd = result.confidence;
    return result;
}

InitialDisplacement IntegerSearchInitializer::estimate_with_mask_around_displacement(
    const Image& reference,
    const Image& deformed,
    const Mask& roi,
    const Eigen::Vector2d& point,
    double initial_u,
    double initial_v) const
{
    if (roi.empty()) {
        return estimate_around_displacement(reference, deformed, point, initial_u, initial_v);
    }
    InitialDisplacement invalid;
    if (reference.empty() || deformed.empty() ||
        reference.width() != deformed.width() ||
        reference.height() != deformed.height() ||
        reference.width() != roi.width() ||
        reference.height() != roi.height() ||
        !std::isfinite(initial_u) || !std::isfinite(initial_v)) {
        return invalid;
    }

    const auto& integer_config = config_.integer_search;
    const int center_x = static_cast<int>(std::llround(point.x()));
    const int center_y = static_cast<int>(std::llround(point.y()));
    const auto samples = collect_masked_reference_samples(
        reference, roi, center_x, center_y, integer_config.subset_radius, 1);
    if (static_cast<int>(samples.size()) < min_required_samples(integer_config.subset_radius, 1)) {
        return invalid;
    }
    const double ref_mean = sample_mean(samples);
    const double ref_norm = sample_norm(samples, ref_mean);
    if (ref_norm <= kEpsilon) {
        return invalid;
    }

    const int search_radius = std::max(0, integer_config.search_radius);
    const int prior_u = static_cast<int>(std::llround(initial_u));
    const int prior_v = static_cast<int>(std::llround(initial_v));
    const int min_u = integer_config.subset_radius - center_x;
    const int max_u = deformed.width() - 1 - integer_config.subset_radius - center_x;
    const int min_v = integer_config.subset_radius - center_y;
    const int max_v = deformed.height() - 1 - integer_config.subset_radius - center_y;
    const auto best = search_displacements(
        deformed,
        samples,
        ref_mean,
        ref_norm,
        center_x,
        center_y,
        integer_config.subset_radius,
        std::max(min_u, prior_u - search_radius),
        std::min(max_u, prior_u + search_radius),
        std::max(min_v, prior_v - search_radius),
        std::min(max_v, prior_v + search_radius),
        1
    );
    if (!std::isfinite(best.zncc)) {
        return invalid;
    }

    InitialDisplacement result;
    result.u = static_cast<double>(best.u);
    result.v = static_cast<double>(best.v);
    result.confidence = znssd_from_zncc(best.zncc);
    result.valid = true;
    result.zncc = best.zncc;
    result.znssd = result.confidence;
    return result;
}

} // namespace dic
