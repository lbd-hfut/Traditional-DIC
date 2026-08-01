// Test Q4/Q8 ICGN on synthetic translation data

#include <gtest/gtest.h>

#include <dic/mesh/mesh_dic.hpp>
#include <dic/core/image.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/node.hpp>
#include <dic/mesh/mesh_config.hpp>

#include "element/shape_func_internal.hpp"

#include <cmath>
#include <vector>

namespace {

dic::Image make_img(int w, int h, double cx, double cy) {
    std::vector<float> d(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double dx = x - cx, dy = y - cy;
            double g = std::exp(-(dx*dx+dy*dy)/(2.0*8.0*8.0));
            double s = 0.3*std::sin(x*0.15)*std::cos(y*0.12);
            d[y*w+x] = static_cast<float>(g+s);
        }
    return dic::Image(w, h, std::move(d));
}

dic::Mesh make_q4_mesh(int w, int h, int nx, int ny) {
    dic::Mesh m;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            dic::Node n; n.id = static_cast<size_t>(j*(nx+1)+i);
            n.coordinate.x() = i*(w-1.0)/nx;
            n.coordinate.y() = j*(h-1.0)/ny;
            m.add_node(n);
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int n0=j*(nx+1)+i, n1=n0+1, n2=n0+(nx+1)+1, n3=n0+(nx+1);
            dic::MeshElementConnectivity e;
            e.type = dic::mesh::MeshElementType::Q4;
            e.node_ids = {static_cast<size_t>(n0),static_cast<size_t>(n1),
                          static_cast<size_t>(n2),static_cast<size_t>(n3)};
            m.add_element(e);
        }
    return m;
}

dic::Mesh make_q8_mesh(int w, int h, int nx, int ny) {
    int npx = 2*nx+1, npy = 2*ny+1;
    dic::Mesh m;
    for (int j = 0; j < npy; ++j)
        for (int i = 0; i < npx; ++i) {
            dic::Node n; n.id = static_cast<size_t>(j*npx+i);
            n.coordinate.x() = i*(w-1.0)/(npx-1.0);
            n.coordinate.y() = j*(h-1.0)/(npy-1.0);
            m.add_node(n);
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int c0=(2*j)*npx+(2*i), c1=c0+2, c2=c0+2*npx+2, c3=c0+2*npx;
            int m4=c0+1, m5=c0+npx+2, m6=c0+2*npx+1, m7=c0+npx;
            dic::MeshElementConnectivity e;
            e.type = dic::mesh::MeshElementType::Q8;
            e.node_ids = {static_cast<size_t>(c0),static_cast<size_t>(c1),
                          static_cast<size_t>(c2),static_cast<size_t>(c3),
                          static_cast<size_t>(m4),static_cast<size_t>(m5),
                          static_cast<size_t>(m6),static_cast<size_t>(m7)};
            m.add_element(e);
        }
    return m;
}

} // namespace

TEST(MeshICGN, Q4_ZeroDisplacement) {
    const int w=60, h=60, nx=3, ny=3;
    auto ref = make_img(w,h,w/2.0,h/2.0);
    auto def = make_img(w,h,w/2.0,h/2.0);

    dic::MeshConfig cfg;
    cfg.max_iterations = 10; cfg.convergence_threshold = 1e-4;
    cfg.regularization_alpha = 0.0; cfg.search_radius = 4;
    cfg.fedic_fft_initialization.window_size = 21;
    cfg.fedic_fft_initialization.search_radius = 4;

    dic::MeshDIC s(cfg);
    auto r = s.compute(ref, def, make_q4_mesh(w,h,nx,ny));
    for (const auto& d : r) {
        EXPECT_NEAR(d.u, 0.0, 0.5) << "u at ("<<d.x<<","<<d.y<<")";
        EXPECT_NEAR(d.v, 0.0, 0.5) << "v at ("<<d.x<<","<<d.y<<")";
    }
}

TEST(MeshICGN, Q4_SubpixelTranslation) {
    const int w=60, h=60, nx=2, ny=2;
    const double du=1.5, dv=-1.0;
    auto ref = make_img(w,h,w/2.0,h/2.0);
    auto def = make_img(w,h,w/2.0+du,h/2.0+dv);

    dic::MeshConfig cfg;
    cfg.max_iterations = 15; cfg.convergence_threshold = 1e-5;
    cfg.regularization_alpha = 1e-6; cfg.search_radius = 8;
    cfg.fedic_fft_initialization.window_size = 21;
    cfg.fedic_fft_initialization.search_radius = 8;

    dic::MeshDIC s(cfg);
    auto r = s.compute(ref, def, make_q4_mesh(w,h,nx,ny));

    int npx=nx+1, npy=ny+1; double su=0, sv=0; int cnt=0;
    for (size_t i = 0; i < r.size(); ++i) {
        int col=i%npx, row=i/npx;
        if (col==0||col==npx-1||row==0||row==npy-1) continue;
        su+=r[i].u; sv+=r[i].v; ++cnt;
    }
    ASSERT_GT(cnt, 0);
    // Q4 bilinear on coarse mesh: expect ~70% accuracy at subpixel level
    EXPECT_NEAR(su/cnt, du, 0.7);
    EXPECT_NEAR(sv/cnt, dv, 0.7);
}

TEST(MeshFGN, Q4_SubpixelTranslation) {
    const int w=60, h=60, nx=2, ny=2;
    const double du=1.5, dv=-1.0;
    auto ref = make_img(w,h,w/2.0,h/2.0);
    auto def = make_img(w,h,w/2.0+du,h/2.0+dv);

    dic::MeshConfig cfg;
    cfg.max_iterations = 15; cfg.convergence_threshold = 1e-5;
    cfg.regularization_alpha = 1e-6; cfg.search_radius = 8;
    cfg.fedic_fft_initialization.window_size = 21;
    cfg.fedic_fft_initialization.search_radius = 8;
    cfg.optimization_method = dic::MeshOptimizationMethod::FEDICElementFGN;

    dic::MeshDIC s(cfg);
    auto r = s.compute(ref, def, make_q4_mesh(w,h,nx,ny));

    int npx=nx+1, npy=ny+1; double su=0, sv=0; int cnt=0;
    for (size_t i = 0; i < r.size(); ++i) {
        int col=i%npx, row=i/npx;
        if (col==0||col==npx-1||row==0||row==npy-1) continue;
        su+=r[i].u; sv+=r[i].v; ++cnt;
    }
    ASSERT_GT(cnt, 0);
    EXPECT_NEAR(su/cnt, du, 0.7);
    EXPECT_NEAR(sv/cnt, dv, 0.7);
}

TEST(MeshICGN, Q8_ZeroDisplacement) {
    const int w=60, h=60, nx=2, ny=2;
    auto ref = make_img(w,h,w/2.0,h/2.0);
    auto def = make_img(w,h,w/2.0,h/2.0);

    dic::MeshConfig cfg;
    cfg.max_iterations = 10; cfg.convergence_threshold = 1e-4;
    cfg.regularization_alpha = 0.0; cfg.search_radius = 4;
    cfg.fedic_fft_initialization.window_size = 21;
    cfg.fedic_fft_initialization.search_radius = 4;

    dic::MeshDIC s(cfg);
    auto r = s.compute(ref, def, make_q8_mesh(w,h,nx,ny));
    for (const auto& d : r) {
        EXPECT_NEAR(d.u, 0.0, 0.5);
        EXPECT_NEAR(d.v, 0.0, 0.5);
    }
}

TEST(MeshICGN, Q8_SubpixelTranslation) {
    const int w=64, h=64, nx=2, ny=2;
    const double du=0.5, dv=0.3;
    auto ref = make_img(w,h,w/2.0,h/2.0);
    auto def = make_img(w,h,w/2.0+du,h/2.0+dv);

    dic::MeshConfig cfg;
    cfg.max_iterations = 20; cfg.convergence_threshold = 1e-5;
    cfg.regularization_alpha = 0.0; cfg.search_radius = 6;
    cfg.fedic_fft_initialization.window_size = 21;
    cfg.fedic_fft_initialization.search_radius = 6;

    dic::MeshDIC s(cfg);
    auto r = s.compute(ref, def, make_q8_mesh(w,h,nx,ny));

    double su=0, sv=0; int cnt=0;
    for (const auto& d : r) { su+=d.u; sv+=d.v; ++cnt; }
    EXPECT_NEAR(su/cnt, du, 0.5);
    EXPECT_NEAR(sv/cnt, dv, 0.5);
}
