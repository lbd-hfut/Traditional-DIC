/**
 * @file bind_core.cpp
 * @brief pybind11 bindings for lightweight core image/mask utilities.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/core/observation_mask.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

dic::Image core_image_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const float*>(buf.ptr);
    std::vector<float> data(src, src + static_cast<std::size_t>(w * h));
    return dic::Image(w, h, std::move(data));
}

dic::Image core_image_from_numpy64(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const double*>(buf.ptr);
    std::vector<float> data(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(src[i]);
    }
    return dic::Image(w, h, std::move(data));
}

py::array_t<float> image_to_numpy(const dic::Image& image)
{
    py::array_t<float> arr({image.height(), image.width()});
    auto* dst = static_cast<float*>(arr.request().ptr);
    std::copy(image.data().begin(), image.data().end(), dst);
    return arr;
}

dic::Mask core_mask_from_numpy_u8(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Mask must be a 2D numpy array.");
    }
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const auto* src = static_cast<const std::uint8_t*>(buf.ptr);
    std::vector<bool> data(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = src[i] != 0;
    }
    return dic::Mask(w, h, std::move(data));
}

py::array_t<std::uint8_t> mask_to_numpy(const dic::Mask& mask)
{
    py::array_t<std::uint8_t> arr({mask.height(), mask.width()});
    auto* dst = static_cast<std::uint8_t*>(arr.request().ptr);
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            dst[static_cast<std::size_t>(y * mask.width() + x)] = mask.valid(x, y) ? 1 : 0;
        }
    }
    return arr;
}

py::array_t<std::uint8_t> u8_mask_array(const std::vector<unsigned char>& data, int height, int width)
{
    py::array_t<std::uint8_t> arr({height, width});
    auto* dst = static_cast<std::uint8_t*>(arr.request().ptr);
    std::copy(data.begin(), data.end(), dst);
    return arr;
}

py::array_t<double> points_array(const std::vector<Eigen::Vector2d>& points)
{
    py::array_t<double> arr({static_cast<py::ssize_t>(points.size()), py::ssize_t{2}});
    auto rows = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(points.size()); ++i) {
        rows(i, 0) = points[static_cast<std::size_t>(i)].x();
        rows(i, 1) = points[static_cast<std::size_t>(i)].y();
    }
    return arr;
}

dic::ObservationMaskOptions observation_mask_options_from_dict(const py::dict& raw)
{
    dic::ObservationMaskOptions options;
    auto set_int = [&](const char* key, int& value) {
        if (raw.contains(key)) {
            value = py::cast<int>(raw[key]);
        }
    };
    auto set_double = [&](const char* key, double& value) {
        if (raw.contains(key)) {
            value = py::cast<double>(raw[key]);
        }
    };
    set_int("outlier_k", options.outlier_k);
    set_double("outlier_knn_scale", options.outlier_knn_scale);
    set_double("component_radius_scale", options.component_radius_scale);
    set_double("edge_scale", options.edge_scale);
    set_double("radius_scale", options.radius_scale);
    set_int("min_hole_area", options.min_hole_area);
    set_int("tiny_hole_fill_area", options.tiny_hole_fill_area);
    return options;
}

py::dict observation_mask_result_to_dict(const dic::ObservationMaskResult& result)
{
    py::dict out;
    out["camera_index"] = result.camera_index;
    out["width"] = result.width;
    out["height"] = result.height;
    out["mask"] = u8_mask_array(result.mask, result.height, result.width);
    out["hull_mask"] = u8_mask_array(result.hull_mask, result.height, result.width);
    out["supported_mask"] = u8_mask_array(result.supported_mask, result.height, result.width);
    out["rejected_hole_mask"] = u8_mask_array(result.rejected_hole_mask, result.height, result.width);
    out["observations"] = points_array(result.observations);
    out["clean_observations"] = points_array(result.clean_observations);
    out["n_triangles_raw"] = result.n_triangles_raw;
    out["n_triangles_valid"] = result.n_triangles_valid;
    out["n_holes_detected"] = result.n_holes_detected;
    out["n_holes_filled_as_speckle"] = result.n_holes_filled_as_speckle;
    out["n_holes_rejected"] = result.n_holes_rejected;
    return out;
}

py::list build_observation_masks_py(
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    py::array_t<int, py::array::c_style | py::array::forcecast> camera_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> observation_uv,
    const py::dict& options_raw)
{
    auto cam_buf = camera_indices.request();
    auto uv_buf = observation_uv.request();
    if (cam_buf.ndim != 1) {
        throw std::runtime_error("camera_indices must have shape (n,).");
    }
    if (uv_buf.ndim != 2 || uv_buf.shape[1] != 2) {
        throw std::runtime_error("observation_uv must have shape (n, 2).");
    }
    if (cam_buf.shape[0] != uv_buf.shape[0]) {
        throw std::runtime_error("camera_indices and observation_uv must contain the same number of observations.");
    }
    const auto cam_rows = camera_indices.unchecked<1>();
    const auto uv_rows = observation_uv.unchecked<2>();
    std::vector<int> cams(static_cast<std::size_t>(cam_buf.shape[0]));
    std::vector<Eigen::Vector2d> uv(static_cast<std::size_t>(uv_buf.shape[0]));
    for (py::ssize_t i = 0; i < cam_buf.shape[0]; ++i) {
        cams[static_cast<std::size_t>(i)] = cam_rows(i);
        uv[static_cast<std::size_t>(i)] = Eigen::Vector2d(uv_rows(i, 0), uv_rows(i, 1));
    }
    const auto options = observation_mask_options_from_dict(options_raw);
    py::gil_scoped_release release;
    auto results = dic::build_observation_masks(widths, heights, cams, uv, options);
    py::gil_scoped_acquire acquire;
    py::list out;
    for (const auto& result : results) {
        out.append(observation_mask_result_to_dict(result));
    }
    return out;
}

dic::Image py_image_from_object(py::object obj)
{
    if (py::isinstance<py::array_t<float>>(obj)) {
        return core_image_from_numpy(py::cast<py::array_t<float>>(obj));
    }
    if (py::isinstance<py::array_t<double>>(obj)) {
        return core_image_from_numpy64(py::cast<py::array_t<double>>(obj));
    }
    throw std::runtime_error("image must be a 2D numpy array (float32 or float64).");
}

} // namespace

void bind_core(py::module_& m)
{
    auto sub = m.def_submodule("core");

    py::enum_<dic::ImageNormalization>(sub, "ImageNormalization")
        .value("None_", dic::ImageNormalization::None)
        .value("MaxIntensity", dic::ImageNormalization::MaxIntensity)
        .value("GlobalMeanStd", dic::ImageNormalization::GlobalMeanStd)
        .value("RoiMeanStd", dic::ImageNormalization::RoiMeanStd);

    py::class_<dic::Image>(sub, "Image")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def_property_readonly("width", &dic::Image::width)
        .def_property_readonly("height", &dic::Image::height)
        .def("empty", &dic::Image::empty)
        .def("to_numpy", &image_to_numpy);

    py::class_<dic::Mask>(sub, "Mask")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def(py::init<const std::string&>())
        .def_property_readonly("width", &dic::Mask::width)
        .def_property_readonly("height", &dic::Mask::height)
        .def("empty", &dic::Mask::empty)
        .def("to_numpy", &mask_to_numpy);

    sub.def("image_from_numpy", [](py::object image) { return py_image_from_object(image); }, py::arg("image"));
    sub.def("mask_from_numpy", &core_mask_from_numpy_u8, py::arg("mask"));
    sub.def("load_image", [](const std::string& path) { return dic::Image(path); }, py::arg("path"));
    sub.def("load_mask", [](const std::string& path) { return dic::Mask(path); }, py::arg("path"));
    sub.def("build_observation_masks",
        &build_observation_masks_py,
        py::arg("widths"),
        py::arg("heights"),
        py::arg("camera_indices"),
        py::arg("observation_uv"),
        py::arg("options") = py::dict());
    sub.def("normalize_image",
        [](py::object image, const std::string& method) {
            dic::ImageNormalization normalization = dic::ImageNormalization::None;
            if (method == "max" || method == "max_intensity") {
                normalization = dic::ImageNormalization::MaxIntensity;
            } else if (method == "global_mean_std") {
                normalization = dic::ImageNormalization::GlobalMeanStd;
            } else if (method != "none") {
                throw std::runtime_error("Unsupported normalization method: " + method);
            }
            return image_to_numpy(dic::normalize_image(py_image_from_object(image), normalization));
        },
        py::arg("image"),
        py::arg("method") = "none");
}
