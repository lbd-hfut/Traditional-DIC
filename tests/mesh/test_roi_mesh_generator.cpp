#include <gtest/gtest.h>

#include <dic/core/mask.hpp>
#include <dic/core/roi.hpp>
#include <dic/mesh/generation/boundary_exporter.hpp>
#include <dic/mesh/generation/roi_mesh_generator.hpp>

#include <filesystem>
#include <fstream>

TEST(ROIMeshGenerator, Placeholder)
{
    // TODO: Validate rectangle ROI.
    // TODO: Validate irregular ROI.
    // TODO: Validate ROI boundary clipping.
    // TODO: Validate no elements outside ROI.
    // TODO: Validate no orphan nodes after cleanup.
    SUCCEED();
}

TEST(BoundaryExporter, ExtractsRectangleMaskBoundary)
{
    dic::Mask mask(6, 5);
    mask.fill(false);
    for (int y = 1; y <= 3; ++y) {
        for (int x = 2; x <= 4; ++x) {
            mask.set(x, y, true);
        }
    }

    const auto loops = dic::mesh::extract_boundary_loops(mask);
    ASSERT_EQ(loops.size(), 1U);
    EXPECT_EQ(loops[0].points.size(), 4U);
}

TEST(ROIMeshGenerator, LoadsManualQ4MeshFiles)
{
    const auto dir = std::filesystem::temp_directory_path() / "traditional_dic_manual_mesh_test";
    std::filesystem::create_directories(dir);
    const auto nodes_path = dir / "nodes_Q4.txt";
    const auto elements_path = dir / "elements_Q4.txt";

    {
        std::ofstream nodes(nodes_path);
        nodes << "1, 0, 0\n";
        nodes << "2, 10, 0\n";
        nodes << "3, 10, 10\n";
        nodes << "4, 0, 10\n";
    }
    {
        std::ofstream elements(elements_path);
        elements << "1, 1, 2, 3, 4\n";
    }

    dic::mesh::MeshGenerationConfig config;
    config.method = dic::mesh::MeshGenerationMethod::Manual;
    config.element_type = dic::mesh::MeshElementType::Q4;
    config.nodes_file = nodes_path.string();
    config.elements_file = elements_path.string();

    dic::mesh::ROIMeshGenerator generator;
    const auto mesh = generator.generate(dic::ROI::all(), config);
    ASSERT_EQ(mesh.nodes().size(), 4U);
    ASSERT_EQ(mesh.elements().size(), 1U);
    EXPECT_EQ(mesh.elements()[0].node_ids[0], 0U);
    EXPECT_EQ(mesh.elements()[0].node_ids[3], 3U);
}
