/**
 * @file yaml_parser.cpp
 * @brief YAML → SubsetConfig parser implementation.
 */

#include <dic/config/yaml_parser.hpp>
#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace dic {
namespace {

// ---------------------------------------------------------------------------
// String → enum helpers
// ---------------------------------------------------------------------------

BSplineDegree parse_bspline_degree(const std::string& value)
{
    if (value == "1") return BSplineDegree::Linear;
    if (value == "3") return BSplineDegree::Cubic;
    if (value == "5") return BSplineDegree::Quintic;
    throw std::runtime_error("interpolation.degree must be 1, 3, or 5: " + value);
}

SubsetShapeFunctionMethod parse_shape_function(const std::string& value)
{
    if (value == "first_order" || value == "1") return SubsetShapeFunctionMethod::FirstOrder;
    if (value == "second_order" || value == "2") return SubsetShapeFunctionMethod::SecondOrder;
    throw std::runtime_error("Unknown shape function: " + value);
}

SubsetOptimizationMethod parse_optimizer(const std::string& value)
{
    if (value == "icgn") return SubsetOptimizationMethod::ICGN;
    if (value == "forward_gauss_newton") return SubsetOptimizationMethod::ForwardGaussNewton;
    throw std::runtime_error("Unknown optimization method: " + value);
}

CorrelationCriterionKind parse_correlation_criterion(const std::string& value)
{
    if (value == "znssd") return CorrelationCriterionKind::ZNSSD;
    if (value == "ssd") return CorrelationCriterionKind::SSD;
    throw std::runtime_error("Unknown correlation criterion: " + value);
}

SeedInitializationMethod parse_init_method(const std::string& value)
{
    if (value == "integer_search") return SeedInitializationMethod::IntegerSearch;
    if (value == "sift") return SeedInitializationMethod::SIFT;
    throw std::runtime_error("Unknown initialization method: " + value);
}

SeedQualityMetric parse_quality_metric(const std::string& value)
{
    if (value == "zncc" ) return SeedQualityMetric::ZNCC;
    if (value == "znssd") return SeedQualityMetric::ZNSSD;
    if (value == "ssd"  ) return SeedQualityMetric::SSD;
    throw std::runtime_error("Unknown seed quality metric: " + value);
}

// ---------------------------------------------------------------------------
// Section parsers: each populates part of the SubsetConfig from a YAML node.
// ---------------------------------------------------------------------------

void parse_optimization_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["method"]) {
        config.optimizer = parse_optimizer(node["method"].as<std::string>());
    }
    if (node["max_iterations"]) {
        config.max_iterations = node["max_iterations"].as<int>();
    }
    if (node["convergence_threshold"]) {
        config.convergence_threshold = node["convergence_threshold"].as<double>();
    }
}

void parse_correlation_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["criterion"]) {
        config.objective = parse_correlation_criterion(node["criterion"].as<std::string>());
        config.seed_initialization.subpixel.objective = config.objective;
    }
}

void parse_interpolation_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["degree"]) {
        config.image_precompute.degree = parse_bspline_degree(
            node["degree"].as<std::string>());
    }
}

void parse_subset_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["radius"]) {
        config.subset_radius = node["radius"].as<int>();
    }
    if (node["truncate_roi_subsets"]) {
        config.truncate_roi_subsets = node["truncate_roi_subsets"].as<bool>();
    }
    if (node["min_valid_sample_ratio"]) {
        config.min_valid_sample_ratio = node["min_valid_sample_ratio"].as<double>();
    }
    if (node["min_valid_samples"]) {
        config.min_valid_samples = node["min_valid_samples"].as<int>();
    }
}

void parse_shape_function_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["order"]) {
        config.shape_function = parse_shape_function(node["order"].as<std::string>());
    }
}

void parse_integer_search(const YAML::Node& node, SeedIntegerSearchConfig& cfg)
{
    if (node["subset_radius"]) {
        cfg.subset_radius = node["subset_radius"].as<int>();
    }
    if (node["search_radius"]) {
        cfg.search_radius = node["search_radius"].as<int>();
    }
    if (node["pyramid_enabled"]) {
        cfg.pyramid_enabled = node["pyramid_enabled"].as<bool>();
    }
    if (node["pyramid_scale"]) {
        cfg.pyramid_scale = node["pyramid_scale"].as<int>();
    }
    if (node["pyramid_refinement_radius"]) {
        cfg.pyramid_refinement_radius = node["pyramid_refinement_radius"].as<int>();
    }
}

void parse_subpixel_refinement(const YAML::Node& node, SeedSubpixelRefinementConfig& cfg)
{
    if (node["enabled"]) {
        cfg.enabled = node["enabled"].as<bool>();
    }
    if (node["shape_function"]) {
        cfg.shape_function = parse_shape_function(node["shape_function"].as<std::string>());
    }
    if (node["optimizer"]) {
        cfg.optimizer = parse_optimizer(node["optimizer"].as<std::string>());
    }
    if (node["objective"]) {
        cfg.objective = parse_correlation_criterion(node["objective"].as<std::string>());
    }
    if (node["criterion"]) {
        cfg.objective = parse_correlation_criterion(node["criterion"].as<std::string>());
    }
    if (node["subset_radius"]) {
        cfg.subset_radius = node["subset_radius"].as<int>();
    }
    if (node["max_iterations"]) {
        cfg.max_iterations = node["max_iterations"].as<int>();
    }
    if (node["convergence_threshold"]) {
        cfg.convergence_threshold = node["convergence_threshold"].as<double>();
    }
}

void parse_initialization_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["method"]) {
        config.seed_initialization.method =
            parse_init_method(node["method"].as<std::string>());
    }
    if (node["integer_search"]) {
        parse_integer_search(node["integer_search"],
                             config.seed_initialization.integer_search);
    }
    if (node["subpixel_refinement"]) {
        parse_subpixel_refinement(node["subpixel_refinement"],
                                  config.seed_initialization.subpixel);
    }
}

void parse_seed_selection_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["seed_count"]) {
        config.seed_selection.seed_count = node["seed_count"].as<int>();
    }
    if (node["threads"]) {
        config.seed_selection.threads = node["threads"].as<int>();
    }
    if (node["quality_metric"]) {
        config.seed_selection.quality_metric =
            parse_quality_metric(node["quality_metric"].as<std::string>());
    }
    if (node["min_zncc"]) {
        config.seed_selection.min_zncc = node["min_zncc"].as<double>();
    }
    if (node["max_znssd"]) {
        config.seed_selection.max_znssd = node["max_znssd"].as<double>();
    }
    if (node["max_ssd"]) {
        config.seed_selection.max_ssd = node["max_ssd"].as<double>();
    }
    if (node["min_displacement_norm"]) {
        config.seed_selection.min_displacement_norm =
            node["min_displacement_norm"].as<double>();
    }
    if (node["min_texture_std"]) {
        config.seed_selection.min_texture_std =
            node["min_texture_std"].as<double>();
    }
    if (node["kmeans_iterations"]) {
        config.seed_selection.kmeans_iterations =
            node["kmeans_iterations"].as<int>();
    }
    if (node["kmeans_sample_limit"]) {
        config.seed_selection.kmeans_sample_limit =
            node["kmeans_sample_limit"].as<int>();
    }
}

void parse_reliability_propagation_section(const YAML::Node& node, SubsetConfig& config)
{
    if (node["spacing"]) {
        config.propagation_spacing = node["spacing"].as<int>();
    }
    if (node["max_znssd"]) {
        config.propagation_max_znssd = node["max_znssd"].as<double>();
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

SubsetConfig load_subset_config_from_yaml(const std::string& path)
{
    SubsetConfig config;

    try {
        YAML::Node root = YAML::LoadFile(path);
        if (!root.IsMap()) {
            throw std::runtime_error("YAML root must be a map.");
        }

        if (root["subset"]) {
            parse_subset_section(root["subset"], config);
        }
        if (root["shape_function"]) {
            parse_shape_function_section(root["shape_function"], config);
        }
        if (root["correlation"]) {
            parse_correlation_section(root["correlation"], config);
        }
        if (root["optimization"]) {
            parse_optimization_section(root["optimization"], config);
        }
        if (root["interpolation"]) {
            parse_interpolation_section(root["interpolation"], config);
        }
        if (root["initialization"]) {
            parse_initialization_section(root["initialization"], config);
        }
        if (root["seed_selection"]) {
            parse_seed_selection_section(root["seed_selection"], config);
        }
        if (root["reliability_propagation"]) {
            parse_reliability_propagation_section(root["reliability_propagation"], config);
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("YAML parse error in ") + path + ": " + e.what());
    }

    return config;
}

SubsetConfig load_subset_config_from_yaml_string(const std::string& content)
{
    SubsetConfig config;

    try {
        YAML::Node root = YAML::Load(content);
        if (!root.IsMap()) {
            std::string type = "undefined";
            if (root.IsNull())     type = "null";
            else if (root.IsScalar()) type = "scalar";
            else if (root.IsSequence()) type = "sequence";
            std::string preview = content.substr(0, std::min<size_t>(content.size(), 120));
            throw std::runtime_error(
                "YAML content root is not a map (got " + type +
                "). Content preview: [" + preview + "]");
        }

        if (root["subset"]) {
            parse_subset_section(root["subset"], config);
        }
        if (root["shape_function"]) {
            parse_shape_function_section(root["shape_function"], config);
        }
        if (root["correlation"]) {
            parse_correlation_section(root["correlation"], config);
        }
        if (root["optimization"]) {
            parse_optimization_section(root["optimization"], config);
        }
        if (root["interpolation"]) {
            parse_interpolation_section(root["interpolation"], config);
        }
        if (root["initialization"]) {
            parse_initialization_section(root["initialization"], config);
        }
        if (root["seed_selection"]) {
            parse_seed_selection_section(root["seed_selection"], config);
        }
        if (root["reliability_propagation"]) {
            parse_reliability_propagation_section(root["reliability_propagation"], config);
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("YAML parse error: ") + e.what());
    }

    return config;
}

} // namespace dic
