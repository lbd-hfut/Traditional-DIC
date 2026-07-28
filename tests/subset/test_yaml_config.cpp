#include <dic/config/yaml_parser.hpp>
#include <gtest/gtest.h>

TEST(YamlConfig, ParsesSubsetCorrelationCriterion)
{
    const auto config = dic::load_subset_config_from_yaml_string(R"(
subset:
  radius: 21
correlation:
  criterion: ssd
initialization:
  subpixel_refinement:
    enabled: true
    optimizer: icgn
)");

    EXPECT_EQ(config.objective, dic::CorrelationCriterionKind::SSD);
    EXPECT_EQ(config.seed_initialization.subpixel.objective, dic::CorrelationCriterionKind::SSD);
}

TEST(YamlConfig, AllowsSubpixelCriterionOverride)
{
    const auto config = dic::load_subset_config_from_yaml_string(R"(
correlation:
  criterion: ssd
initialization:
  subpixel_refinement:
    criterion: znssd
)");

    EXPECT_EQ(config.objective, dic::CorrelationCriterionKind::SSD);
    EXPECT_EQ(config.seed_initialization.subpixel.objective, dic::CorrelationCriterionKind::ZNSSD);
}
