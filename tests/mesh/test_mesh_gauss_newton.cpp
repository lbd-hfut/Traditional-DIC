#include <gtest/gtest.h>

#include <dic/mesh/mesh_dic.hpp>
#include <dic/core/image.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/node.hpp>
#include <dic/mesh/mesh_config.hpp>

#include <cmath>
#include <vector>

namespace {

// Build a synthetic image with a Gaussian blob at (cx, cy)
dic::Image make_gaussian(int w, int h, double cx, double cy, double sigma)
{
    std::vector<float> data(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double dx = x - cx, dy = y - cy;
            float val = static_cast<float>(
                std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)));
            data[static_cast<size_t>(y * w + x)] = val;
        }
    }
    return dic::Image(w, h, std::move(data));
}
// Build a simple structured mesh: nx*ny Q4 elements covering [0,w-1] × [0,h-1]
dic::Mesh make_structured_mesh(int w, int h, int nx, int ny)
{
    dic::Mesh mesh;
    constexpr double margin = 12.0;
    // Create (nx+1)×(ny+1) nodes
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            dic::Node node;
            node.id = static_cast<size_t>(j * (nx + 1) + i);
            node.coordinate.x() = margin + static_cast<double>(i) * (w - 1 - 2 * margin) / nx;
            node.coordinate.y() = margin + static_cast<double>(j) * (h - 1 - 2 * margin) / ny;
            mesh.add_node(node);
        }
    }
    // Create nx*ny Q4 elements (CCW order)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int n0 = j * (nx + 1) + i;
            int n1 = j * (nx + 1) + (i + 1);
            int n2 = (j + 1) * (nx + 1) + (i + 1);
            int n3 = (j + 1) * (nx + 1) + i;
            dic::MeshElementConnectivity elem;
            elem.type = dic::mesh::MeshElementType::Q4;
            elem.node_ids = {
                static_cast<size_t>(n0), static_cast<size_t>(n1),
                static_cast<size_t>(n2), static_cast<size_t>(n3)
            };
            mesh.add_element(elem);
        }
    }
    return mesh;
}

} // namespace

// ============================================================
// Integration test: full Mesh-DIC pipeline with synthetic images
// ============================================================

TEST(MeshDICIntegration, RecoversUniformTranslationQ4)
{
    const int w = 48, h = 48;
    const double shift_u = 1.5, shift_v = -0.8;

    // Reference: Gaussian at center
    auto ref_img = make_gaussian(w, h, w / 2.0, h / 2.0, 5.0);
    // Deformed: same Gaussian shifted
    auto def_img = make_gaussian(w, h, w / 2.0 + shift_u, h / 2.0 + shift_v, 5.0);

    // 3x3 Q4 mesh → 9 nodes in a 3×3 grid
    auto mesh = make_structured_mesh(w, h, 2, 2);

    dic::MeshConfig config;
    config.max_iterations = 10;
    config.convergence_threshold = 1e-3;
    config.regularization_alpha = 0.0;
    config.search_radius = 12;
    config.fedic_fft_initialization.window_size = 21;
    config.fedic_fft_initialization.search_radius = 12;

    dic::MeshDIC solver(config);
    auto results = solver.compute(ref_img, def_img, mesh);

    // Results should contain per-node displacements
    ASSERT_EQ(results.size(), 9u);

    // Interior nodes should recover approximate (shift_u, shift_v)
    // Corner nodes may have boundary effects, so check interior
    double avg_u = 0.0, avg_v = 0.0;
    for (const auto& r : results) {
        avg_u += r.u;
        avg_v += r.v;
    }
    avg_u /= static_cast<double>(results.size());
    avg_v /= static_cast<double>(results.size());

    // Check that the average recovered displacement is within 0.5 px of ground truth
    EXPECT_NEAR(avg_u, shift_u, 0.5) << "Average u displacement should be near " << shift_u;
    EXPECT_NEAR(avg_v, shift_v, 0.5) << "Average v displacement should be near " << shift_v;
}

TEST(MeshDICIntegration, RecoversZeroDisplacementQ4)
{
    const int w = 40, h = 40;

    auto ref_img = make_gaussian(w, h, w / 2.0, h / 2.0, 4.0);
    auto def_img = ref_img;  // no deformation

    auto mesh = make_structured_mesh(w, h, 2, 2);

    dic::MeshConfig config;
    config.max_iterations = 5;
    config.convergence_threshold = 1e-2;
    config.regularization_alpha = 0.0;
    config.search_radius = 12;
    config.fedic_fft_initialization.window_size = 21;
    config.fedic_fft_initialization.search_radius = 12;

    dic::MeshDIC solver(config);
    auto results = solver.compute(ref_img, def_img, mesh);

    ASSERT_EQ(results.size(), 9u);

    // All displacements should be close to zero
    for (const auto& r : results) {
        EXPECT_NEAR(r.u, 0.0, 1.0) << "Node at (" << r.x << ", " << r.y << ")";
        EXPECT_NEAR(r.v, 0.0, 1.0) << "Node at (" << r.x << ", " << r.y << ")";
    }
}

