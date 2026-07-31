/**
 * @file bind_subset.cpp
 * @brief pybind11 bindings for the 2D Subset-DIC pipeline.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dic/config/yaml_parser.hpp>
#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/result.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/subset/subset_config.hpp>
#include <dic/subset/subset_dic.hpp>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: numpy array → dic::Image
// ---------------------------------------------------------------------------
dic::Image image_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    std::vector<float> data(static_cast<std::size_t>(w * h));
    const float* src = static_cast<const float*>(buf.ptr);
    // numpy is row-major (H, W), Image is row-major (W, H) ... wait,
    // Image stores as vector<float> indexed by y*w + x which is also row-major.
    // But Image::at(x, y) uses data[y * w + x].
    // numpy (H, W) → data[i] is at row i/W, col i%W.
    // Image data[y*w + x] = pixel at (x, y).
    // So they're compatible: just copy flat.
    std::copy(src, src + data.size(), data.begin());
    return dic::Image(w, h, std::move(data));
}

dic::Image image_from_numpy_64(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    std::vector<float> data(static_cast<std::size_t>(w * h));
    const double* src = static_cast<const double*>(buf.ptr);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(src[i]);
    }
    return dic::Image(w, h, std::move(data));
}

// ---------------------------------------------------------------------------
// Helper: numpy array → dic::Mask
// ---------------------------------------------------------------------------
dic::Mask mask_from_numpy(py::array_t<bool, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Mask must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    std::vector<bool> data(static_cast<std::size_t>(w * h));
    const bool* src = static_cast<const bool*>(buf.ptr);
    std::copy(src, src + data.size(), data.begin());
    return dic::Mask(w, h, std::move(data));
}

dic::Mask mask_from_numpy_uint8(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Mask must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    std::vector<bool> data(static_cast<std::size_t>(w * h));
    const std::uint8_t* src = static_cast<const std::uint8_t*>(buf.ptr);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = (src[i] != 0);
    }
    return dic::Mask(w, h, std::move(data));
}

// ---------------------------------------------------------------------------
// Helper: SubsetDIC results → Python dict of numpy arrays
// ---------------------------------------------------------------------------
py::dict results_to_dict(const std::vector<dic::Displacement2D>& results)
{
    const std::size_t n = results.size();
    py::array_t<double> x_arr(n);
    py::array_t<double> y_arr(n);
    py::array_t<double> u_arr(n);
    py::array_t<double> v_arr(n);
    py::array_t<double> du_dx_arr(n);
    py::array_t<double> du_dy_arr(n);
    py::array_t<double> dv_dx_arr(n);
    py::array_t<double> dv_dy_arr(n);
    py::array_t<double> corr_arr(n);
    py::array_t<bool>   valid_arr(n);

    auto* px  = static_cast<double*>(x_arr.request().ptr);
    auto* py_ = static_cast<double*>(y_arr.request().ptr);
    auto* pu  = static_cast<double*>(u_arr.request().ptr);
    auto* pv  = static_cast<double*>(v_arr.request().ptr);
    auto* pdx = static_cast<double*>(du_dx_arr.request().ptr);
    auto* pdy = static_cast<double*>(du_dy_arr.request().ptr);
    auto* pvx = static_cast<double*>(dv_dx_arr.request().ptr);
    auto* pvy = static_cast<double*>(dv_dy_arr.request().ptr);
    auto* pc  = static_cast<double*>(corr_arr.request().ptr);
    auto* pok = static_cast<bool*>(valid_arr.request().ptr);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& r = results[i];
        px[i]  = r.x;
        py_[i] = r.y;
        pu[i]  = r.u;
        pv[i]  = r.v;
        pdx[i] = r.du_dx;
        pdy[i] = r.du_dy;
        pvx[i] = r.dv_dx;
        pvy[i] = r.dv_dy;
        pc[i]  = r.correlation;
        pok[i] = r.valid;
    }

    py::dict d;
    d["x"]           = x_arr;
    d["y"]           = y_arr;
    d["u"]           = u_arr;
    d["v"]           = v_arr;
    d["du_dx"]       = du_dx_arr;
    d["du_dy"]       = du_dy_arr;
    d["dv_dx"]       = dv_dx_arr;
    d["dv_dy"]       = dv_dy_arr;
    d["correlation"] = corr_arr;
    d["valid"]       = valid_arr;
    return d;
}

// ---------------------------------------------------------------------------
// Config from Python dict (YAML parsed by Python side)
// ---------------------------------------------------------------------------
static dic::SubsetConfig config_from_dict(py::dict d) {
    dic::SubsetConfig cfg;
    auto gi = [&](py::dict& dd, const char* k, int def) { return dd.contains(k) ? py::cast<int>(dd[k]) : def; };
    auto gd = [&](py::dict& dd, const char* k, double def) { return dd.contains(k) ? py::cast<double>(dd[k]) : def; };

    if (d.contains("subset")) { py::dict s = d["subset"]; cfg.subset_radius = gi(s, "radius", cfg.subset_radius); }
    if (d.contains("optimization")) {
        py::dict o = d["optimization"];
        cfg.max_iterations = gi(o, "max_iterations", cfg.max_iterations);
        cfg.convergence_threshold = gd(o, "convergence_threshold", cfg.convergence_threshold);
    }
    if (d.contains("interpolation")) {
        py::dict o = d["interpolation"];
        if (o.contains("degree")) { int deg = py::cast<int>(o["degree"]);
            if (deg==1) cfg.image_precompute.degree=dic::BSplineDegree::Linear;
            else if (deg==3) cfg.image_precompute.degree=dic::BSplineDegree::Cubic; }
    }
    if (d.contains("initialization")) {
        py::dict o = d["initialization"];
        if (o.contains("method")) {
            const std::string method = py::cast<std::string>(o["method"]);
            if (method != "integer_search") {
                throw std::runtime_error("subset initialization.method must be integer_search");
            }
            cfg.seed_initialization.method = dic::SeedInitializationMethod::IntegerSearch;
        }
        if (o.contains("integer_search")) { py::dict is_ = o["integer_search"];
            cfg.seed_initialization.integer_search.subset_radius = gi(is_,"subset_radius",cfg.seed_initialization.integer_search.subset_radius);
            cfg.seed_initialization.integer_search.search_radius = gi(is_,"search_radius",cfg.seed_initialization.integer_search.search_radius);
            if (is_.contains("sift_enabled")) cfg.seed_initialization.integer_search.sift_enabled = py::cast<bool>(is_["sift_enabled"]);
            if (is_.contains("pyramid_enabled")) cfg.seed_initialization.integer_search.pyramid_enabled = py::cast<bool>(is_["pyramid_enabled"]);
            cfg.seed_initialization.integer_search.pyramid_scale = gi(is_,"pyramid_scale",cfg.seed_initialization.integer_search.pyramid_scale);
            cfg.seed_initialization.integer_search.pyramid_refinement_radius = gi(is_,"pyramid_refinement_radius",cfg.seed_initialization.integer_search.pyramid_refinement_radius); }
        if (o.contains("subpixel_refinement")) { py::dict sr = o["subpixel_refinement"];
            if (sr.contains("enabled")) cfg.seed_initialization.subpixel.enabled = py::cast<bool>(sr["enabled"]);
            cfg.seed_initialization.subpixel.subset_radius = gi(sr,"subset_radius",cfg.seed_initialization.subpixel.subset_radius);
            cfg.seed_initialization.subpixel.max_iterations = gi(sr,"max_iterations",cfg.seed_initialization.subpixel.max_iterations);
            cfg.seed_initialization.subpixel.convergence_threshold = gd(sr,"convergence_threshold",cfg.seed_initialization.subpixel.convergence_threshold); }
    }
    if (d.contains("seed_selection")) { py::dict o = d["seed_selection"];
        cfg.seed_selection.seed_count = gi(o,"seed_count",cfg.seed_selection.seed_count);
        cfg.seed_selection.max_znssd = gd(o,"max_znssd",cfg.seed_selection.max_znssd);
        cfg.seed_selection.min_displacement_norm = gd(o,"min_displacement_norm",cfg.seed_selection.min_displacement_norm);
        cfg.seed_selection.min_texture_std = gd(o,"min_texture_std",cfg.seed_selection.min_texture_std);
        cfg.seed_selection.kmeans_iterations = gi(o,"kmeans_iterations",cfg.seed_selection.kmeans_iterations);
        cfg.seed_selection.kmeans_sample_limit = gi(o,"kmeans_sample_limit",cfg.seed_selection.kmeans_sample_limit); }
    if (d.contains("reliability_propagation")) { py::dict o = d["reliability_propagation"];
        cfg.propagation_spacing = gi(o,"spacing",cfg.propagation_spacing);
        cfg.propagation_max_znssd = gd(o,"max_znssd",cfg.propagation_max_znssd); }
    return cfg;
}

// ---------------------------------------------------------------------------
// Python-exposed compute function
// ---------------------------------------------------------------------------
py::dict subset_compute(
    py::object reference,
    py::object deformed,
    py::object config_arg,
    py::object roi_arg
)
{
    // --- Parse reference image ---
    dic::Image ref_img;
    if (py::isinstance<py::array_t<float>>(reference)) {
        ref_img = image_from_numpy(py::cast<py::array_t<float>>(reference));
    } else if (py::isinstance<py::array_t<double>>(reference)) {
        ref_img = image_from_numpy_64(py::cast<py::array_t<double>>(reference));
    } else {
        throw std::runtime_error("reference must be a 2D numpy array (float32 or float64).");
    }

    // --- Parse deformed image ---
    dic::Image def_img;
    if (py::isinstance<py::array_t<float>>(deformed)) {
        def_img = image_from_numpy(py::cast<py::array_t<float>>(deformed));
    } else if (py::isinstance<py::array_t<double>>(deformed)) {
        def_img = image_from_numpy_64(py::cast<py::array_t<double>>(deformed));
    } else {
        throw std::runtime_error("deformed must be a 2D numpy array (float32 or float64).");
    }

    // --- Parse config (dict from Python YAML parsing) ---
    dic::SubsetConfig cfg;
    if (!config_arg.is_none()) {
        if (py::isinstance<py::dict>(config_arg)) {
            cfg = config_from_dict(py::cast<py::dict>(config_arg));
        } else {
            throw std::runtime_error("config must be a dict (from YAML) or None.");
        }
    }

    // --- Parse ROI ---
    dic::Mask roi_mask;
    bool has_roi = false;
    if (!roi_arg.is_none()) {
        if (py::isinstance<py::array_t<bool>>(roi_arg)) {
            roi_mask = mask_from_numpy(py::cast<py::array_t<bool>>(roi_arg));
        } else if (py::isinstance<py::array_t<std::uint8_t>>(roi_arg)) {
            roi_mask = mask_from_numpy_uint8(py::cast<py::array_t<std::uint8_t>>(roi_arg));
        } else {
            throw std::runtime_error("roi must be a 2D numpy array (bool or uint8) or None.");
        }
        has_roi = true;
    }

    // --- Run Subset-DIC ---
    dic::SubsetDIC solver(cfg);
    std::vector<dic::Displacement2D> results;

    if (has_roi) {
        results = solver.compute(ref_img, def_img, roi_mask);
    } else {
        results = solver.compute(ref_img, def_img);
    }

    return results_to_dict(results);
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
void bind_subset(py::module_& m)
{
    auto sub = m.def_submodule("subset");

    sub.def(
        "compute",
        &subset_compute,
        py::arg("reference"),
        py::arg("deformed"),
        py::arg("config")  = py::none(),
        py::arg("roi")     = py::none(),
        R"pbdoc(
        Run 2D Subset-DIC with reliability-guided propagation.

        Parameters
        ----------
        reference : numpy.ndarray (H, W) float32 or float64
            Reference (undeformed) image.
        deformed : numpy.ndarray (H, W) float32 or float64
            Deformed image. Must have same shape as reference.
        config : str or None
            Path to a YAML configuration file. If None, default parameters are used.
        roi : numpy.ndarray (H, W) bool or uint8, or None
            Region-of-interest mask. True/1 = inside ROI. If None, the entire image
            is treated as the ROI.

        Returns
        -------
        dict of numpy.ndarray
            'x', 'y'       — grid point coordinates [px]
            'u', 'v'       — displacement [px]
            'du_dx', 'du_dy', 'dv_dx', 'dv_dy' — affine warp gradient parameters
            'correlation'  — ZNSSD correlation coefficient
            'valid'        — bool, whether the point converged
        )pbdoc"
    );
}
