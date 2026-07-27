#include <dic/mesh/mesh_generation_config.hpp>

#include "coordinate/g2l_internal.hpp"
#include "generation/inform_builder.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

dic::mesh::MeshElementType parse_element_type(const std::string& value)
{
    if (value == "T3" || value == "t3") {
        return dic::mesh::MeshElementType::T3;
    }
    if (value == "Q8" || value == "q8") {
        return dic::mesh::MeshElementType::Q8;
    }
    return dic::mesh::MeshElementType::Q4;
}

std::string element_type_name(dic::mesh::MeshElementType type)
{
    switch (type) {
    case dic::mesh::MeshElementType::T3:
        return "T3";
    case dic::mesh::MeshElementType::Q4:
        return "Q4";
    case dic::mesh::MeshElementType::Q8:
        return "Q8";
    }
    return "Q4";
}

int nodes_per_element(dic::mesh::MeshElementType type)
{
    switch (type) {
    case dic::mesh::MeshElementType::T3:
        return 3;
    case dic::mesh::MeshElementType::Q4:
        return 4;
    case dic::mesh::MeshElementType::Q8:
        return 8;
    }
    return 4;
}

void read_nodes(const std::filesystem::path& path, std::vector<double>& coords)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open nodes file: " + path.string());
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int id = 0;
        double x = 0.0;
        double y = 0.0;
        if (iss >> id >> x >> y) {
            coords.push_back(x);
            coords.push_back(y);
        }
    }
}

void read_elements(
    const std::filesystem::path& path,
    dic::mesh::MeshElementType type,
    std::vector<int>& elements,
    int& n_elements)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open elements file: " + path.string());
    }

    const int nn = nodes_per_element(type);
    const int stride = (type == dic::mesh::MeshElementType::Q8) ? 9 : nn;
    std::string line;
    n_elements = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int eid = 0;
        int nid = 0;
        if (!(iss >> eid)) {
            continue;
        }
        for (int i = 0; i < nn; ++i) {
            if (!(iss >> nid)) {
                throw std::runtime_error("Element row has too few node ids: " + line);
            }
            elements.push_back(nid);
        }
        if (stride > nn) {
            elements.push_back(0);
        }
        ++n_elements;
    }
}

void write_mesh_svg(
    const std::filesystem::path& path,
    const std::vector<double>& coords,
    const std::vector<int>& elements,
    int n_elements,
    dic::mesh::MeshElementType type,
    int image_width,
    int image_height)
{
    const int nn = nodes_per_element(type);
    const int stride = (type == dic::mesh::MeshElementType::Q8) ? 9 : nn;
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write " + path.string());
    }
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << image_width
        << "\" height=\"" << image_height << "\" viewBox=\"0 0 " << image_width
        << " " << image_height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << image_width << "\" height=\""
        << image_height << "\" fill=\"none\" stroke=\"#ccc\"/>\n";
    out << "<g fill=\"none\" stroke=\"#555\" stroke-width=\"1\">\n";
    for (int e = 0; e < n_elements; ++e) {
        out << "<polyline points=\"";
        for (int k = 0; k < nn; ++k) {
            const int nid = elements[e * stride + k] - 1;
            out << coords[2 * nid] << "," << coords[2 * nid + 1] << " ";
        }
        const int nid0 = elements[e * stride] - 1;
        out << coords[2 * nid0] << "," << coords[2 * nid0 + 1] << "\"/>\n";
    }
    out << "</g>\n";
    out << "<g fill=\"#d62728\">\n";
    for (std::size_t i = 0; i < coords.size() / 2U; ++i) {
        out << "<circle cx=\"" << coords[2 * i] << "\" cy=\"" << coords[2 * i + 1]
            << "\" r=\"3\"/>\n";
    }
    out << "</g>\n";
    out << "</svg>\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: mesh_input_diagnostic <nodes.txt> <elements.txt> <out_dir>"
                  << " [element_type=Q4] [image_width=1280] [image_height=1280]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path nodes_path = argv[1];
        const std::filesystem::path elements_path = argv[2];
        const std::filesystem::path out_dir = argv[3];
        const auto type = argc >= 5 ? parse_element_type(argv[4]) : dic::mesh::MeshElementType::Q4;
        const int image_width = argc >= 6 ? std::atoi(argv[5]) : 1280;
        const int image_height = argc >= 7 ? std::atoi(argv[6]) : 1280;

        std::filesystem::create_directories(out_dir);

        std::vector<double> coords;
        std::vector<int> elements;
        int n_elements = 0;
        read_nodes(nodes_path, coords);
        read_elements(elements_path, type, elements, n_elements);
        const int n_nodes = static_cast<int>(coords.size() / 2);

        auto inform = dic::mesh::build_inform(
            coords.data(), n_nodes,
            elements.data(), n_elements,
            type, image_height, image_width);
        const int n_pixels = static_cast<int>(inform.size() / 3);

        dic::mesh::internal::G2LParams params;
        params.max_iter = 200;
        auto g2l = dic::mesh::internal::compute_global_to_local(
            inform.data(), n_pixels,
            coords.data(), n_nodes,
            elements.data(), n_elements,
            image_height, image_width,
            type, params);

        int valid_pixels = 0;
        for (unsigned char flag : g2l.valid) {
            if (flag) {
                ++valid_pixels;
            }
        }

        write_mesh_svg(out_dir / "mesh_preview.svg", coords, elements, n_elements,
                       type, image_width, image_height);

        std::ofstream summary(out_dir / "summary.txt");
        summary << "element_type=" << element_type_name(type) << "\n";
        summary << "image_width=" << image_width << "\n";
        summary << "image_height=" << image_height << "\n";
        summary << "nodes=" << n_nodes << "\n";
        summary << "elements=" << n_elements << "\n";
        summary << "inform_pixels=" << n_pixels << "\n";
        summary << "g2l_valid_pixels=" << valid_pixels << "\n";

        std::cout << "Wrote mesh input diagnostics to " << out_dir.string() << "\n";
        std::cout << "Nodes: " << n_nodes << ", elements: " << n_elements << "\n";
        std::cout << "Inform pixels: " << n_pixels << ", valid G2L: " << valid_pixels << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "mesh_input_diagnostic failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
