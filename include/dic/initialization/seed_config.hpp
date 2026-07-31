#ifndef TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SEED_CONFIG_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SEED_CONFIG_HPP

namespace dic {

enum class SeedInitializationMethod {
    IntegerSearch,
    SIFT
};

enum class SubsetShapeFunctionMethod {
    FirstOrder,
    SecondOrder
};

enum class SubsetOptimizationMethod {
    ICGN,
    ForwardGaussNewton
};

enum class CorrelationCriterionKind {
    SSD,
    ZNSSD
};

enum class SeedQualityMetric {
    ZNCC,
    ZNSSD,
    SSD
};

struct SeedIntegerSearchConfig {
    int subset_radius{10};
    int search_radius{30};
    bool sift_enabled{false};
    bool pyramid_enabled{true};
    int pyramid_scale{4};
    int pyramid_refinement_radius{4};
};

struct SeedSubpixelRefinementConfig {
    bool enabled{true};
    SubsetShapeFunctionMethod shape_function{SubsetShapeFunctionMethod::FirstOrder};
    SubsetOptimizationMethod optimizer{SubsetOptimizationMethod::ICGN};
    CorrelationCriterionKind objective{CorrelationCriterionKind::ZNSSD};
    int subset_radius{15};
    int max_iterations{30};
    double convergence_threshold{1e-3};
};

struct SeedInitializationConfig {
    SeedInitializationMethod method{SeedInitializationMethod::IntegerSearch};
    SeedIntegerSearchConfig integer_search{};
    SeedSubpixelRefinementConfig subpixel{};
};

struct SeedSelectionConfig {
    int seed_count{256};
    int threads{1};
    SeedQualityMetric quality_metric{SeedQualityMetric::ZNSSD};
    double min_zncc{0.7};
    double max_znssd{0.2};
    double max_ssd{0.05};
    double min_displacement_norm{0.0};
    double min_texture_std{0.0};
    int kmeans_iterations{20};
    int kmeans_sample_limit{20000};
};

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_INITIALIZATION_SEED_CONFIG_HPP
