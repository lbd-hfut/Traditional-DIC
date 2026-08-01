/**
 * @file bind_mesh.cpp
 * @brief pybind11 bindings for the 2D Mesh-DIC pipeline.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/roi.hpp>
#include <dic/core/result.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/mesh/generation/annulus_mesh_generator.hpp>
#include <dic/mesh/generation/roi_mesh_generator.hpp>
#include <dic/mesh/mesh.hpp>
#include <dic/mesh/mesh_config.hpp>
#include <dic/mesh/mesh_dic.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

dic::Image mesh_image_from_object(py::object obj)
{
    if (py::isinstance<py::array_t<float>>(obj)) {
        py::array_t<float, py::array::c_style | py::array::forcecast> arr = py::cast<py::array_t<float>>(obj);
        auto buf = arr.request();
        if (buf.ndim != 2) throw std::runtime_error("Image must be a 2D numpy array.");
        const int h = static_cast<int>(buf.shape[0]);
        const int w = static_cast<int>(buf.shape[1]);
        const auto* src = static_cast<const float*>(buf.ptr);
        std::vector<float> data(src, src + static_cast<std::size_t>(w * h));
        return dic::Image(w, h, std::move(data));
    }
    if (py::isinstance<py::array_t<double>>(obj)) {
        py::array_t<double, py::array::c_style | py::array::forcecast> arr = py::cast<py::array_t<double>>(obj);
        auto buf = arr.request();
        if (buf.ndim != 2) throw std::runtime_error("Image must be a 2D numpy array.");
        const int h = static_cast<int>(buf.shape[0]);
        const int w = static_cast<int>(buf.shape[1]);
        const auto* src = static_cast<const double*>(buf.ptr);
        std::vector<float> data(static_cast<std::size_t>(w * h));
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(src[i]);
        return dic::Image(w, h, std::move(data));
    }
    throw std::runtime_error("image must be a 2D numpy array (float32 or float64).");
}

dic::Mask mesh_mask_from_object(py::object obj)
{
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<std::uint8_t>>(obj);
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("ROI must be a 2D numpy array.");
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const std::uint8_t*>(buf.ptr);
    std::vector<bool> data(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = src[i] != 0;
    return dic::Mask(w, h, std::move(data));
}

dic::mesh::MeshElementType parse_element_type(const std::string& value)
{
    if (value == "T3" || value == "t3") return dic::mesh::MeshElementType::T3;
    if (value == "Q4" || value == "q4") return dic::mesh::MeshElementType::Q4;
    if (value == "Q8" || value == "q8") return dic::mesh::MeshElementType::Q8;
    throw std::runtime_error("Unsupported element_type: " + value);
}

int nodes_per_element(dic::mesh::MeshElementType type)
{
    switch (type) {
    case dic::mesh::MeshElementType::T3: return 3;
    case dic::mesh::MeshElementType::Q4: return 4;
    case dic::mesh::MeshElementType::Q8: return 8;
    }
    return 4;
}

std::string element_type_name(dic::mesh::MeshElementType type)
{
    switch (type) {
    case dic::mesh::MeshElementType::T3: return "T3";
    case dic::mesh::MeshElementType::Q4: return "Q4";
    case dic::mesh::MeshElementType::Q8: return "Q8";
    }
    return "Q4";
}

dic::Mesh mesh_from_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> elements,
    dic::mesh::MeshElementType type,
    bool one_based)
{
    auto nb = nodes.request();
    auto eb = elements.request();
    if (nb.ndim != 2 || nb.shape[1] != 2) {
        throw std::runtime_error("nodes must have shape (n_nodes, 2).");
    }
    if (eb.ndim != 2 || eb.shape[1] < nodes_per_element(type)) {
        throw std::runtime_error("elements must have shape (n_elements, nodes_per_element).");
    }

    dic::Mesh mesh;
    const auto* node_ptr = static_cast<const double*>(nb.ptr);
    const auto n_nodes = static_cast<std::size_t>(nb.shape[0]);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        dic::Node node;
        node.id = i;
        node.coordinate = Eigen::Vector2d{node_ptr[2 * i], node_ptr[2 * i + 1]};
        mesh.add_node(node);
    }

    const auto* elem_ptr = static_cast<const std::int64_t*>(eb.ptr);
    const int nn = nodes_per_element(type);
    const auto stride = static_cast<std::size_t>(eb.shape[1]);
    for (std::size_t e = 0; e < static_cast<std::size_t>(eb.shape[0]); ++e) {
        dic::MeshElementConnectivity elem;
        elem.type = type;
        elem.node_ids.reserve(static_cast<std::size_t>(nn));
        for (int k = 0; k < nn; ++k) {
            std::int64_t raw = elem_ptr[e * stride + static_cast<std::size_t>(k)];
            if (one_based) --raw;
            if (raw < 0 || raw >= static_cast<std::int64_t>(n_nodes)) {
                throw std::runtime_error("element references an out-of-range node index.");
            }
            elem.node_ids.push_back(static_cast<std::size_t>(raw));
        }
        mesh.add_element(elem);
    }
    return mesh;
}

py::dict mesh_to_dict(const dic::Mesh& mesh)
{
    const auto& nodes = mesh.nodes();
    const auto& elements = mesh.elements();
    py::array_t<double> node_arr({static_cast<py::ssize_t>(nodes.size()), py::ssize_t{2}});
    auto* node_ptr = static_cast<double*>(node_arr.request().ptr);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        node_ptr[2 * i] = nodes[i].coordinate.x();
        node_ptr[2 * i + 1] = nodes[i].coordinate.y();
    }

    const auto type = elements.empty() ? dic::mesh::MeshElementType::Q4 : elements.front().type;
    const int nn = nodes_per_element(type);
    py::array_t<std::int64_t> elem_arr({static_cast<py::ssize_t>(elements.size()), static_cast<py::ssize_t>(nn)});
    auto* elem_ptr = static_cast<std::int64_t*>(elem_arr.request().ptr);
    for (std::size_t e = 0; e < elements.size(); ++e) {
        for (int k = 0; k < nn; ++k) {
            elem_ptr[e * static_cast<std::size_t>(nn) + static_cast<std::size_t>(k)] =
                static_cast<std::int64_t>(elements[e].node_ids[static_cast<std::size_t>(k)]);
        }
    }

    py::dict result;
    result["nodes"] = node_arr;
    result["elements"] = elem_arr;
    result["element_type"] = element_type_name(type);
    return result;
}

py::dict annulus_summary_to_dict(const dic::mesh::AnnulusMeshGenerationSummary& summary)
{
    py::dict d;
    d["center"] = py::make_tuple(summary.center_x, summary.center_y);
    d["center_x"] = summary.center_x;
    d["center_y"] = summary.center_y;
    d["inner_radius"] = summary.inner_radius;
    d["outer_radius"] = summary.outer_radius;
    d["radial_divisions"] = summary.radial_divisions;
    d["circumferential_divisions"] = summary.circumferential_divisions;
    d["target_element_size"] = summary.target_element_size;
    d["min_element_size"] = summary.min_element_size;
    d["max_element_size"] = summary.max_element_size;
    return d;
}

py::dict results_to_dict(const std::vector<dic::Displacement2D>& results)
{
    const std::size_t n = results.size();
    py::array_t<double> x(n), y(n), u(n), v(n), mag(n);
    py::array_t<bool> valid(n);
    auto* px = static_cast<double*>(x.request().ptr);
    auto* py_ = static_cast<double*>(y.request().ptr);
    auto* pu = static_cast<double*>(u.request().ptr);
    auto* pv = static_cast<double*>(v.request().ptr);
    auto* pm = static_cast<double*>(mag.request().ptr);
    auto* pok = static_cast<bool*>(valid.request().ptr);
    for (std::size_t i = 0; i < n; ++i) {
        px[i] = results[i].x;
        py_[i] = results[i].y;
        pu[i] = results[i].u;
        pv[i] = results[i].v;
        pm[i] = std::hypot(results[i].u, results[i].v);
        pok[i] = results[i].valid;
    }
    py::dict d;
    d["x"] = x;
    d["y"] = y;
    d["u"] = u;
    d["v"] = v;
    d["mag"] = mag;
    d["valid"] = valid;
    return d;
}

dic::MeshConfig mesh_config_from_dict(py::dict d)
{
    dic::MeshConfig cfg;
    auto get_int = [](py::dict obj, const char* key, int fallback) {
        return obj.contains(key) ? py::cast<int>(obj[key]) : fallback;
    };
    auto get_double = [](py::dict obj, const char* key, double fallback) {
        return obj.contains(key) ? py::cast<double>(obj[key]) : fallback;
    };
    auto get_bool = [](py::dict obj, const char* key, bool fallback) {
        return obj.contains(key) ? py::cast<bool>(obj[key]) : fallback;
    };

    if (d.contains("mesh")) {
        py::dict m = py::cast<py::dict>(d["mesh"]);
        cfg.max_iterations = get_int(m, "max_iterations", cfg.max_iterations);
        cfg.convergence_threshold = get_double(m, "convergence_threshold", cfg.convergence_threshold);
        cfg.search_radius = get_int(m, "search_radius", cfg.search_radius);
        cfg.regularization_alpha = get_double(m, "regularization_alpha", cfg.regularization_alpha);
        cfg.mirror_image_padding = get_bool(m, "mirror_image_padding", cfg.mirror_image_padding);
    }

    if (d.contains("interpolation")) {
        py::dict interp = py::cast<py::dict>(d["interpolation"]);
        if (interp.contains("degree")) {
            const int degree = py::cast<int>(interp["degree"]);
            if (degree == 1) cfg.image_precompute.degree = dic::BSplineDegree::Linear;
            else if (degree == 3) cfg.image_precompute.degree = dic::BSplineDegree::Cubic;
            else if (degree == 5) cfg.image_precompute.degree = dic::BSplineDegree::Quintic;
            else throw std::runtime_error("interpolation.degree must be 1, 3, or 5.");
        }
        cfg.image_precompute.border = get_int(interp, "border", cfg.image_precompute.border);
        cfg.image_precompute.use_exact_prefilter =
            get_bool(interp, "use_exact_prefilter", cfg.image_precompute.use_exact_prefilter);
        cfg.image_precompute.precompute_local_blocks =
            get_bool(interp, "precompute_local_blocks", cfg.image_precompute.precompute_local_blocks);
    }

    if (d.contains("initialization")) {
        py::dict init = py::cast<py::dict>(d["initialization"]);
        if (init.contains("method")) {
            const std::string method = py::cast<std::string>(init["method"]);
            if (method != "fedic_fft" && method != "fe_dic_fft") {
                throw std::runtime_error("Mesh-DIC initialization.method must be 'fedic_fft'.");
            }
        }
        if (init.contains("fedic_fft")) {
            py::dict fft = py::cast<py::dict>(init["fedic_fft"]);
            cfg.fedic_fft_initialization.window_size =
                get_int(fft, "window_size", cfg.fedic_fft_initialization.window_size);
            cfg.fedic_fft_initialization.search_radius =
                get_int(fft, "search_radius", cfg.fedic_fft_initialization.search_radius);
            cfg.fedic_fft_initialization.mirror_boundary_fallback =
                get_bool(fft, "mirror_boundary_fallback",
                         cfg.fedic_fft_initialization.mirror_boundary_fallback);
        }
        if (init.contains("quality_control")) {
            py::dict qc = py::cast<py::dict>(init["quality_control"]);
            cfg.initialization_quality.enabled =
                get_bool(qc, "enabled", cfg.initialization_quality.enabled);
            cfg.initialization_quality.min_zncc =
                get_double(qc, "min_zncc", cfg.initialization_quality.min_zncc);
            cfg.initialization_quality.max_znssd =
                get_double(qc, "max_znssd", cfg.initialization_quality.max_znssd);
            cfg.initialization_quality.fedic_qfactor_enabled =
                get_bool(qc, "fedic_qfactor_enabled", cfg.initialization_quality.fedic_qfactor_enabled);
            cfg.initialization_quality.fedic_qfactor_std_factor =
                get_double(qc, "fedic_qfactor_std_factor", cfg.initialization_quality.fedic_qfactor_std_factor);
            cfg.initialization_quality.neighbor_mad_factor =
                get_double(qc, "neighbor_mad_factor", cfg.initialization_quality.neighbor_mad_factor);
            cfg.initialization_quality.max_neighbor_deviation =
                get_double(qc, "max_neighbor_deviation", cfg.initialization_quality.max_neighbor_deviation);
            cfg.initialization_quality.interpolation_neighbors =
                get_int(qc, "interpolation_neighbors", cfg.initialization_quality.interpolation_neighbors);
        }
    }

    return cfg;
}

py::dict mesh_compute(
    py::object reference,
    py::object deformed,
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> elements,
    const std::string& element_type,
    py::object config_arg,
    bool one_based)
{
    dic::MeshConfig cfg;
    if (!config_arg.is_none()) {
        if (!py::isinstance<py::dict>(config_arg)) {
            throw std::runtime_error("config must be a dict or None.");
        }
        cfg = mesh_config_from_dict(py::cast<py::dict>(config_arg));
    }

    auto type = parse_element_type(element_type);
    dic::Mesh mesh = mesh_from_numpy(nodes, elements, type, one_based);
    dic::MeshDIC solver(cfg);
    return results_to_dict(solver.compute(mesh_image_from_object(reference), mesh_image_from_object(deformed), mesh));
}

py::dict make_mesh(
    py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> elements,
    const std::string& element_type,
    bool one_based)
{
    return mesh_to_dict(mesh_from_numpy(nodes, elements, parse_element_type(element_type), one_based));
}

py::dict generate_mesh_from_roi(py::object roi, const std::string& element_type, py::dict config_arg)
{
    dic::mesh::MeshGenerationConfig cfg;
    cfg.element_type = parse_element_type(element_type);
    if (config_arg.contains("target_element_size")) {
        cfg.target_element_size = py::cast<double>(config_arg["target_element_size"]);
    }
    if (config_arg.contains("min_element_size")) {
        cfg.min_element_size = py::cast<double>(config_arg["min_element_size"]);
    }
    if (config_arg.contains("max_element_size")) {
        cfg.max_element_size = py::cast<double>(config_arg["max_element_size"]);
    }
    if (config_arg.contains("min_element_quality")) {
        cfg.min_element_quality = py::cast<double>(config_arg["min_element_quality"]);
    }
    return mesh_to_dict(dic::mesh::generate_annulus_mesh_from_mask(mesh_mask_from_object(roi), cfg));
}

py::dict generate_annulus_meshes_from_mask(py::object roi, py::dict config_arg)
{
    dic::mesh::MeshGenerationConfig cfg;
    if (config_arg.contains("target_element_size")) {
        cfg.target_element_size = py::cast<double>(config_arg["target_element_size"]);
    }
    if (config_arg.contains("min_element_size")) {
        cfg.min_element_size = py::cast<double>(config_arg["min_element_size"]);
    }
    if (config_arg.contains("max_element_size")) {
        cfg.max_element_size = py::cast<double>(config_arg["max_element_size"]);
    }
    if (config_arg.contains("min_element_quality")) {
        cfg.min_element_quality = py::cast<double>(config_arg["min_element_quality"]);
    }

    auto generated = dic::mesh::generate_annulus_meshes_from_mask(mesh_mask_from_object(roi), cfg);
    py::dict result;
    result["T3"] = mesh_to_dict(generated.t3);
    result["Q4"] = mesh_to_dict(generated.q4);
    result["Q8"] = mesh_to_dict(generated.q8);
    result["summary"] = annulus_summary_to_dict(generated.summary);
    return result;
}

} // namespace

void bind_mesh(py::module_& m)
{
    auto sub = m.def_submodule("mesh");

    sub.def("compute", &mesh_compute,
        py::arg("reference"),
        py::arg("deformed"),
        py::arg("nodes"),
        py::arg("elements"),
        py::arg("element_type") = "Q4",
        py::arg("config") = py::none(),
        py::arg("one_based") = false);

    sub.def("make_mesh", &make_mesh,
        py::arg("nodes"),
        py::arg("elements"),
        py::arg("element_type") = "Q4",
        py::arg("one_based") = false);

    sub.def("generate_mesh_from_roi", &generate_mesh_from_roi,
        py::arg("roi"),
        py::arg("element_type") = "Q4",
        py::arg("config") = py::dict());

    sub.def("generate_annulus_meshes_from_mask", &generate_annulus_meshes_from_mask,
        py::arg("roi"),
        py::arg("config") = py::dict());
}
