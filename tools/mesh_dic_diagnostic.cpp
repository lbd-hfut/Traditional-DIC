// mesh_dic_diagnostic: end-to-end Mesh-DIC on real ring images
//
// Usage:
//   mesh_dic_diagnostic <ref.bmp> <def.bmp> <nodes.txt> <elements.txt> <out_dir>
//
// Reads Q4/Q8/T3 mesh from text files, builds inform, runs full pipeline,
// writes U.csv and strain.csv to out_dir.
//
// BMP reader adapted from subset_dic_diagnostic.cpp.

#include <dic/mesh/mesh_dic.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/node.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <dic/core/image.hpp>
#include <dic/mesh/postprocess/strain.hpp>
#include <dic/mesh/solver/local_icgn.hpp>
#include <dic/interpolation/bspline.hpp>

#include "coordinate/g2l_internal.hpp"
#include "solver/fem_assembler.hpp"
#include "generation/inform_builder.hpp"
#include "postprocess/field_interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// ============================================================
// BMP reader (same as subset_dic_diagnostic.cpp)
// ============================================================
namespace {

std::uint16_t read_u16(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}
std::uint32_t read_u32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset] | (bytes[offset + 1] << 8) |
                                      (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24));
}
std::int32_t read_i32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}
std::vector<unsigned char> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(in),
                                       std::istreambuf_iterator<char>());
}

dic::Image read_bmp(const std::string& path) {
    const auto bytes = read_file_bytes(path);
    if (bytes.size() < 54U || bytes[0] != 'B' || bytes[1] != 'M')
        throw std::runtime_error("Not a BMP: " + path);

    auto data_offset = read_u32(bytes, 10);
    auto dib_size = read_u32(bytes, 14);
    if (dib_size < 40U) throw std::runtime_error("Unsupported DIB: " + path);

    int w = read_i32(bytes, 18);
    int sh = read_i32(bytes, 22);
    int h = std::abs(sh);
    bool top_down = sh < 0;
    auto bpp = read_u16(bytes, 28);
    auto comp = read_u32(bytes, 30);

    if (w <= 0 || h <= 0 || comp != 0U)
        throw std::runtime_error("Unsupported BMP layout: " + path);
    if (bpp != 8U && bpp != 24U && bpp != 32U)
        throw std::runtime_error("Only 8/24/32-bit BMP: " + path);

    int bytes_per_pixel = static_cast<int>(bpp / 8U);
    int row_stride = ((w * static_cast<int>(bpp) + 31) / 32) * 4;

    std::vector<float> pixels(static_cast<size_t>(w * h), 0.0F);
    for (int y = 0; y < h; ++y) {
        int src_y = top_down ? y : h - 1 - y;
        auto row_off = static_cast<size_t>(data_offset + src_y * row_stride);
        for (int x = 0; x < w; ++x) {
            auto px_off = row_off + static_cast<size_t>(x * bytes_per_pixel);
            double gray;
            if (bpp == 8U) {
                gray = static_cast<double>(bytes[px_off]);
            } else {
                double b = static_cast<double>(bytes[px_off]);
                double g = static_cast<double>(bytes[px_off + 1U]);
                double r = static_cast<double>(bytes[px_off + 2U]);
                gray = 0.114 * b + 0.587 * g + 0.299 * r;
            }
            pixels[static_cast<size_t>(y * w + x)] = static_cast<float>(gray / 255.0);
        }
    }
    return dic::Image(w, h, std::move(pixels));
}

// ============================================================
// Mesh file reader
// Format: "node_id, x, y" per line (1-based IDs)
// ============================================================
void read_nodes(const std::string& path, std::vector<double>& coords, int& n_nodes) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open nodes file: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int id; double x, y;
        if (iss >> id >> x >> y) {
            coords.push_back(x);
            coords.push_back(y);
        }
    }
    n_nodes = static_cast<int>(coords.size() / 2);
}

// Format: "elem_id, n0, n1, ..." per line (1-based node IDs)
void read_elements(const std::string& path, std::vector<int>& elems,
                   int& n_elements, dic::mesh::MeshElementType etype) {
    using namespace dic::mesh::internal;
    int nn = nodes_per_element(etype);
    int stride = (etype == dic::mesh::MeshElementType::Q8) ? 9 : nn;

    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open elements file: " + path);
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int eid, nid;
        iss >> eid;
        for (int k = 0; k < nn; ++k) {
            if (iss >> nid) {
                elems.push_back(nid);
            } else {
                elems.push_back(0);
            }
        }
        // Pad Q8 9th column
        if (stride > nn) {
            if (!(iss >> nid)) nid = 0;
            elems.push_back(nid);
        }
        ++count;
    }
    n_elements = count;
}

// ============================================================
// Output helpers
// ============================================================
void write_csv(const std::string& path,
               const std::vector<double>& nodes_coord, int n_nodes,
               const Eigen::VectorXd& U) {
    std::ofstream out(path);
    out << "node,x,y,u,v\n";
    for (int i = 0; i < n_nodes; ++i) {
        out << i << ","
            << nodes_coord[2 * i] << ","
            << nodes_coord[2 * i + 1] << ","
            << U(2 * i) << ","
            << U(2 * i + 1) << "\n";
    }
}

void write_strain_csv(const std::string& path, int n_nodes,
                      const std::vector<double>& nodes_coord,
                      const std::vector<double>& Exx,
                      const std::vector<double>& Eyy,
                      const std::vector<double>& Exy) {
    std::ofstream out(path);
    out << "node,x,y,Exx,Eyy,Exy\n";
    for (int i = 0; i < n_nodes; ++i) {
        out << i << ","
            << nodes_coord[2 * i] << ","
            << nodes_coord[2 * i + 1] << ","
            << Exx[i] << "," << Eyy[i] << "," << Exy[i] << "\n";
    }
}

} // namespace

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: mesh_dic_diagnostic <ref.bmp> <def.bmp>"
                  << " <nodes.txt> <elements.txt> <out_dir> [element_type=Q4]\n";
        return EXIT_FAILURE;
    }

    try {
        std::string ref_path = argv[1];
        std::string def_path = argv[2];
        std::string nodes_path = argv[3];
        std::string elems_path = argv[4];
        std::string out_dir = argv[5];
        std::string etype_str = (argc >= 7) ? argv[6] : "Q4";

        dic::mesh::MeshElementType etype;
        if (etype_str == "Q8") etype = dic::mesh::MeshElementType::Q8;
        else if (etype_str == "T3") etype = dic::mesh::MeshElementType::T3;
        else etype = dic::mesh::MeshElementType::Q4;

        auto t0 = std::chrono::steady_clock::now();
        auto t_print = [&t0](const char* label) {
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  [" << label << "] " << ms << " ms\n";
            t0 = t1;
        };

        t_print("start");

        std::cout << "Loading images...\n";
        auto ref_img = read_bmp(ref_path);
        auto def_img = read_bmp(def_path);
        int img_h = ref_img.height(), img_w = ref_img.width();
        std::cout << "  Image size: " << img_w << " x " << img_h << "\n";
        t_print("load images");

        t_print("load mesh");
        std::cout << "Loading mesh...\n";
        std::vector<double> nodes_coord;
        std::vector<int> elements_flat;
        int n_nodes, n_elements;
        read_nodes(nodes_path, nodes_coord, n_nodes);
        read_elements(elems_path, elements_flat, n_elements, etype);
        std::cout << "  Nodes: " << n_nodes << ", Elements: " << n_elements
                  << ", Type: " << etype_str << "\n";

        t_print("build inform");
        std::cout << "Building inform array...\n";
        auto inform = dic::mesh::build_inform(
            nodes_coord.data(), n_nodes,
            elements_flat.data(), n_elements, etype,
            img_h, img_w);
        int n_pixels = static_cast<int>(inform.size() / 3);
        std::cout << "  Pixels: " << n_pixels << "\n";

        t_print("compute G2L");
        std::cout << "Computing G2L...\n";
        dic::mesh::internal::G2LParams g2l_params;
        g2l_params.max_iter = 200;
        auto g2l = dic::mesh::internal::compute_global_to_local(
            inform.data(), n_pixels,
            nodes_coord.data(), n_nodes,
            elements_flat.data(), n_elements,
            img_h, img_w, etype, g2l_params);

        // Count valid G2L pixels
        int g2l_valid = 0;
        int total = img_h * img_w;
        for (int i = 0; i < total; ++i) if (g2l.valid[i]) ++g2l_valid;
        std::cout << "  Valid G2L: " << g2l_valid << " / " << total << "\n";

        // Extract flat image data and gradients
        t_print("extract data");
        std::cout << "Extracting image data...\n";
        std::vector<double> ref_flat(img_h * img_w);
        std::vector<double> def_flat(img_h * img_w);
        std::vector<double> fx_flat(img_h * img_w);
        std::vector<double> fy_flat(img_h * img_w);
        for (int y = 0; y < img_h; ++y) {
            for (int x = 0; x < img_w; ++x) {
                int idx = y * img_w + x;
                ref_flat[idx] = static_cast<double>(ref_img.at(x, y));
                def_flat[idx] = static_cast<double>(def_img.at(x, y));
                if (x > 0 && x < img_w - 1)
                    fx_flat[idx] = 0.5 * (ref_flat[y * img_w + x + 1] - ref_flat[y * img_w + x - 1]);
                if (y > 0 && y < img_h - 1)
                    fy_flat[idx] = 0.5 * (ref_flat[(y + 1) * img_w + x] - ref_flat[(y - 1) * img_w + x]);
            }
        }

        t_print("assemble stiff");
        std::cout << "Assembling stiffness...\n";
        double alpha = 1e-3;
        auto cache = dic::mesh::internal::assemble_stiffness(
            g2l, img_h, img_w,
            fx_flat.data(), fy_flat.data(),
            n_nodes, elements_flat.data(), n_elements, etype, alpha, 0.0,
            etype != dic::mesh::MeshElementType::Q8, true);
        std::cout << "  FEM size: " << cache.fem_size
                  << ", Hessian NNZ: " << cache.A.nonZeros() << "\n";

        // BSpline for deformed image
        t_print("bspline pre");
        std::cout << "BSpline precompute...\n";
        dic::BSplinePrecomputeConfig bsp_cfg(dic::BSplineDegree::Cubic, 3, false);
        bsp_cfg.precompute_local_blocks = false;  // skip local blocks (slow to build)
        dic::BSplineImagePreprocessor preproc(bsp_cfg);
        auto def_precomp = preproc.compute_lazy(def_img);  // lazy: skip local blocks
        const dic::BSplinePrecomputedImage* def_precomp_ptr = &def_precomp;
        dic::BSplineInterpolator def_interp(def_precomp_ptr);
        t_print("bspline done");

        // Initial displacement: zero (solver handles convergence without init)
        t_print("init");
        std::cout << "Initializing displacements (zero init)...\n";
        Eigen::VectorXd U = Eigen::VectorXd::Zero(2 * n_nodes);

        t_print("solver");
        std::cout << "Running Global ICGN solver...\n";
        int iterations = dic::mesh::internal::global_icgn(
            cache, g2l, ref_flat.data(), img_h, img_w,
            elements_flat.data(), n_elements,
            U, &def_interp, alpha, 1e-3, 15, 0.0);
        std::cout << "  Iterations: " << iterations << "\n";

        t_print("compute strain");
        std::cout << "Computing strain...\n";
        std::vector<double> Exx, Eyy, Exy;
        dic::compute_strain(etype, U.data(), n_nodes,
                            nodes_coord.data(), elements_flat.data(), n_elements,
                            Exx, Eyy, Exy);

        // Interpolate full-field displacement using shape functions
        t_print("field interp");
        std::cout << "Interpolating full-field displacement...\n";
        std::vector<double> u_field, v_field;
        dic::mesh::internal::interpolate_displacement_to_pixels(
            g2l, elements_flat.data(), n_elements, etype,
            U, img_h, img_w, u_field, v_field);

        // Write output
        t_print("write output");
        std::cout << "Writing results to " << out_dir << "...\n";
        write_csv(out_dir + "/U.csv", nodes_coord, n_nodes, U);
        write_strain_csv(out_dir + "/strain.csv", n_nodes, nodes_coord, Exx, Eyy, Exy);

        // Write full-field float32 binary (w,h,u_field,v_field)
        {
            auto write_f32 = [](const std::string& path, const std::vector<double>& v, int w, int h) {
                std::ofstream out(path, std::ios::binary);
                int dims[2] = {w, h};
                out.write(reinterpret_cast<const char*>(dims), 8);
                for (double val : v) {
                    float f = std::isfinite(val) ? static_cast<float>(val) : 0.0f;
                    out.write(reinterpret_cast<const char*>(&f), 4);
                }
            };
            write_f32(out_dir + "/u_field.bin", u_field, img_w, img_h);
            write_f32(out_dir + "/v_field.bin", v_field, img_w, img_h);
            // Also write mask of valid pixels
            std::ofstream mask_out(out_dir + "/field_mask.bin", std::ios::binary);
            int dims[2] = {img_w, img_h};
            mask_out.write(reinterpret_cast<const char*>(dims), 8);
            for (double val : u_field) {
                unsigned char v = std::isfinite(val) ? 1 : 0;
                mask_out.write(reinterpret_cast<const char*>(&v), 1);
            }
        }

        std::cout << "\n=== Summary ===\n"
                  << "  Image: " << img_w << "x" << img_h << "\n"
                  << "  Element type: " << etype_str << "\n"
                  << "  Nodes: " << n_nodes << "\n"
                  << "  Elements: " << n_elements << "\n"
                  << "  Inform pixels: " << n_pixels << "\n"
                  << "  G2L valid: " << g2l_valid << "\n"
                  << "  Solver iterations: " << iterations << "\n"
                  << "  U range: ["
                  << U.minCoeff() << ", " << U.maxCoeff() << "]\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_dic_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
