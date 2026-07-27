/**
 * @file yaml_parser.hpp
 * @brief YAML configuration parser for Traditional-DIC workflows.
 *
 * Responsibilities:
 * - Load SubsetConfig (and future MeshConfig, etc.) from YAML files at runtime.
 * - Bridge between human-readable YAML config files and C++ configuration structs.
 *
 * Dependencies:
 * - yaml-cpp (linked via traditional_dic_core).
 * - SubsetConfig and related types from dic/subset and dic/initialization.
 */

#ifndef TRADITIONAL_DIC_INCLUDE_DIC_CONFIG_YAML_PARSER_HPP
#define TRADITIONAL_DIC_INCLUDE_DIC_CONFIG_YAML_PARSER_HPP

#include <dic/subset/subset_config.hpp>

#include <string>

namespace dic {

/**
 * @brief Load a SubsetConfig from a YAML file.
 *
 * The YAML file must follow the same structure as config/subset_2d.yaml.
 * Unspecified keys retain their C++ struct default values.
 *
 * @param path Filesystem path to the YAML configuration file.
 * @return Populated SubsetConfig.
 * @throws std::runtime_error if the file cannot be opened or parsed.
 */
SubsetConfig load_subset_config_from_yaml(const std::string& path);

/**
 * @brief Load a SubsetConfig from YAML-formatted string content.
 */
SubsetConfig load_subset_config_from_yaml_string(const std::string& content);

} // namespace dic

#endif // TRADITIONAL_DIC_INCLUDE_DIC_CONFIG_YAML_PARSER_HPP
